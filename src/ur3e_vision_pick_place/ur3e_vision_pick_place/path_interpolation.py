#!/usr/bin/env python3
"""
Path Interpolation Node — Cartesian Space

Linear position + SLERP orientation, time-scaled with a trapezoidal
profile. IK converts each cartesian waypoint to joints.

No math lives here — this file is just ROS wiring:
  - Cartesian interpolation from helper_functions/path_interpolation.py
  - IK from helper_functions/kinematics.py
  - Current EE pose read from the FK node via /end_effector_pose

Input:  /path_target_pose (PoseStamped)
Output: JointTrajectory to controller

Author: Tejas
"""

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory

from ur3e_vision_pick_place.helper_functions.kinematics import compute_ik, load_pinocchio
from ur3e_vision_pick_place.helper_functions.path_interpolation import (
    interpolate_cartesian_path, quat_to_rotation_matrix,
)
from ur3e_vision_pick_place.robot_config import JOINT_NAMES
from ur3e_vision_pick_place.ros_utils import profile_to_trajectory_msg


class PathInterpolation(Node):
    def __init__(self):
        super().__init__('path_interpolation')

        self.joint_names = JOINT_NAMES

        # Load Pinocchio (same function IK node uses)
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Pinocchio loaded: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            return

        # Profile parameters
        self.vmax = 0.3
        self.amax = 0.3
        self.dt = 0.05

        # State
        self.current_pose = None
        self.current_q = np.zeros(6)
        self.joints_received = False

        # Sub: current EE pose from FK node
        self.create_subscription(
            PoseStamped, '/end_effector_pose', self.ee_pose_cb, 10)
        # Sub: current joints for IK seed
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        # Sub: target pose
        self.create_subscription(
            PoseStamped, '/path_target_pose', self.target_cb, 10)

        # Pub: trajectory to controller
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.get_logger().info('Path Interpolation Node Ready!')
        self.get_logger().info('  FK pose from: /end_effector_pose')
        self.get_logger().info('  Interpolation from: helper_functions.path_interpolation')
        self.get_logger().info('  IK from: helper_functions.kinematics.compute_ik()')
        self.get_logger().info('  Send target to: /path_target_pose')

    # ── Callbacks ─────────────────────────────────────────────────

    def ee_pose_cb(self, msg):
        self.current_pose = msg

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            for i, name in enumerate(self.joint_names):
                self.current_q[i] = positions[name]
            self.joints_received = True

    def target_cb(self, msg):
        if self.current_pose is None:
            self.get_logger().warn('No EE pose yet — is FK node running?')
            return
        if not self.joints_received:
            self.get_logger().warn('No joint states yet!')
            return
        self.plan_and_execute(msg)

    # ── Planning ──────────────────────────────────────────────────

    def plan_and_execute(self, target_msg):
        # Start pose (from FK node topic) and end pose (the request)
        sp = self.current_pose.pose
        ep = target_msg.pose
        start_pos = np.array([sp.position.x, sp.position.y, sp.position.z])
        start_quat = np.array([sp.orientation.x, sp.orientation.y,
                               sp.orientation.z, sp.orientation.w])
        end_pos = np.array([ep.position.x, ep.position.y, ep.position.z])
        end_quat = np.array([ep.orientation.x, ep.orientation.y,
                             ep.orientation.z, ep.orientation.w])

        self.get_logger().info(
            f'Position distance: {np.linalg.norm(end_pos - start_pos):.4f} m')

        # Straight-line + SLERP path, trapezoid time-scaled (pure math)
        times, positions, quaternions = interpolate_cartesian_path(
            start_pos, start_quat, end_pos, end_quat,
            self.vmax, self.amax, self.dt)

        if len(times) == 1 and np.allclose(positions[0], start_pos, atol=1e-6):
            self.get_logger().info('Already at target.')
            return

        self.get_logger().info(
            f'Trajectory: {len(times)} waypoints, {times[-1]:.2f}s')

        # IK each cartesian waypoint, seeding from the previous solution
        # so the whole path stays on one IK branch.
        q_seed = self.current_q.copy()
        joint_times = []
        joint_positions = []
        ik_failures = 0

        for i in range(len(times)):
            q_sol = compute_ik(
                self.model, self.data, self.ee_frame_id,
                positions[i], quat_to_rotation_matrix(quaternions[i]), q_seed,
                joint_names=JOINT_NAMES)

            if q_sol is None:
                ik_failures += 1
                if ik_failures > 5:
                    self.get_logger().error(
                        f'Too many IK failures ({ik_failures}), aborting.')
                    return
                continue

            q_seed = q_sol.copy()
            joint_times.append(times[i])
            joint_positions.append(q_sol)

        if not joint_positions:
            self.get_logger().error('No valid waypoints!')
            return

        joint_times = np.asarray(joint_times)
        joint_positions = np.asarray(joint_positions)

        # Per-joint velocities by central finite difference. Without
        # them the controller interpolates positions at constant
        # velocity, so speed jumps at every waypoint -> jerky motion.
        # The real timestamps are used, which stays valid even when IK
        # failures left non-uniform spacing; endpoints are pinned to 0.
        velocities = self._finite_difference_velocities(joint_times, joint_positions)

        traj_msg = profile_to_trajectory_msg(
            self.joint_names, joint_times, joint_positions, velocities)
        self.traj_pub.publish(traj_msg)
        self.get_logger().info(
            f'Published {len(traj_msg.points)} points, '
            f'{ik_failures} IK failures skipped')

    @staticmethod
    def _finite_difference_velocities(times, positions):
        """Central-difference joint velocities, zero at both endpoints."""
        n = len(times)
        velocities = np.zeros_like(positions)
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 1e-9:
                velocities[i] = (positions[i + 1] - positions[i - 1]) / dt
        return velocities


def main(args=None):
    rclpy.init(args=args)
    node = PathInterpolation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
