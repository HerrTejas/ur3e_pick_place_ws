#!/usr/bin/env python3
"""
Path Interpolation Node — Cartesian Space

Linear position + SLERP orientation, time-scaled with
trapezoidal profile. IK converts each waypoint to joints.

No code duplication:
  - IK math imported from inverse_kinematics.py
  - Trapezoid math imported from trapezoidal_planner.py
  - Current EE pose read from FK node via /end_effector_pose topic

Input:  /path_target_pose (PoseStamped)
Output: JointTrajectory to controller

Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Import math from existing nodes — no duplication
from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik
from ur3e_vision_pick_place.trapezoidal_planner import TrajectoryProfile


class PathInterpolation(Node):
    def __init__(self):
        super().__init__('path_interpolation')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Load Pinocchio (same function IK node uses)
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Pinocchio loaded: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            return

        # Trapezoid profile (same class trapezoidal_planner uses)
        self.profile = TrajectoryProfile()

        # Profile parameters
        self.vmax = 0.5
        self.amax = 0.5
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
        self.get_logger().info('  IK from: inverse_kinematics.compute_ik()')
        self.get_logger().info('  Profile from: trapezoidal_planner.TrajectoryProfile')
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
        # Start pose (from FK node topic)
        sp = self.current_pose.pose
        start_pos = np.array([sp.position.x, sp.position.y, sp.position.z])
        r_start = R.from_quat([
            sp.orientation.x, sp.orientation.y,
            sp.orientation.z, sp.orientation.w])

        # End pose
        ep = target_msg.pose
        end_pos = np.array([ep.position.x, ep.position.y, ep.position.z])
        r_end = R.from_quat([
            ep.orientation.x, ep.orientation.y,
            ep.orientation.z, ep.orientation.w])

        # Path lengths
        L_pos = np.linalg.norm(end_pos - start_pos)
        L_ori = (r_start.inv() * r_end).magnitude()

        self.get_logger().info(f'Position distance: {L_pos:.4f} m')
        self.get_logger().info(f'Orientation distance: {L_ori:.4f} rad')

        if max(L_pos, L_ori) < 1e-6:
            self.get_logger().info('Already at target.')
            return

        # Synchronized trapezoidal profile
        t_array, s_scaled, _ = self.profile.trapezoid_multi(
            [L_pos, L_ori], self.vmax, self.amax, self.dt)

        s_pos = s_scaled[0]
        s_ori = s_scaled[1]

        self.get_logger().info(
            f'Trajectory: {len(t_array)} waypoints, {t_array[-1]:.2f}s')

        # Setup SLERP
        rots = R.concatenate([r_start, r_end])
        slerp = Slerp([0.0, 1.0], rots)

        # Interpolate + IK
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        q_seed = self.current_q.copy()
        ik_failures = 0

        for i in range(len(t_array)):
            t_pos = (s_pos[i] / L_pos) if L_pos > 1e-6 else 0.0
            t_ori = (s_ori[i] / L_ori) if L_ori > 1e-6 else 0.0
            t_interp = min(max(t_pos, t_ori), 1.0)

            # Cartesian interpolation: linear pos + SLERP orientation
            pos = (1.0 - t_interp) * start_pos + t_interp * end_pos
            r_interp = slerp(t_interp)

            # IK (same function the IK node uses)
            q_sol = compute_ik(
                self.model, self.data, self.ee_frame_id,
                pos, r_interp.as_matrix(), q_seed)

            if q_sol is None:
                ik_failures += 1
                if ik_failures > 5:
                    self.get_logger().error(
                        f'Too many IK failures ({ik_failures}), aborting.')
                    return
                continue

            q_seed = q_sol.copy()

            point = JointTrajectoryPoint()
            point.positions = q_sol.tolist()
            t = t_array[i]
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9))
            traj_msg.points.append(point)

        if not traj_msg.points:
            self.get_logger().error('No valid waypoints!')
            return

        self.traj_pub.publish(traj_msg)
        self.get_logger().info(
            f'Published {len(traj_msg.points)} points, '
            f'{ik_failures} IK failures skipped')


def main(args=None):
    rclpy.init(args=args)
    node = PathInterpolation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()