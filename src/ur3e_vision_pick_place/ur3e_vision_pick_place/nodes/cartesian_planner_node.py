#!/usr/bin/env python3
"""Cartesian Planner Node.

Subscribes:  /path_target_pose    (geometry_msgs/PoseStamped) — target EE pose
Subscribes:  /end_effector_pose   (geometry_msgs/PoseStamped) — current EE pose
Subscribes:  /joint_states        (sensor_msgs/JointState) — current joints
Publishes:   /scaled_joint_trajectory_controller/joint_trajectory
                                   (trajectory_msgs/JointTrajectory)

When a target pose arrives, generates a Cartesian path from the
current EE pose to the target (linear position, SLERP orientation),
solves IK at each waypoint, and publishes the joint trajectory.

IK guards reject:
  - per-joint jumps > 0.4 rad (single-joint flips)
  - whole-arm step norm > 1.0 rad (branch shifts where every joint
    moves moderately but no single joint trips per-joint guard)

All math is in helper_functions.path_interpolation and helper_functions.kinematics.
"""

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from ur3e_vision_pick_place.helper_functions.kinematics import (
    load_pinocchio,
    compute_ik,
)
from ur3e_vision_pick_place.helper_functions.path_interpolation import (
    cartesian_path,
)
from ur3e_vision_pick_place.helper_functions.trajectory_utils import (
    joint_step_max,
    joint_step_norm,
)


class CartesianPlannerNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    # Profile parameters for the Cartesian path
    VMAX = 0.2
    AMAX = 0.2
    DT = 0.05
    T_MIN = 2.0
    DURATION_SCALE = 10.0

    # IK guards
    MAX_PER_JOINT_JUMP = 0.4   # ~23 deg on any one joint
    MAX_NORM_JUMP = 1.0        # whole-arm branch shift
    MAX_IK_FAILURES = 5        # consecutive convergence failures
    MAX_IK_JUMPS = 10          # total guard rejections

    def __init__(self):
        super().__init__('cartesian_planner')

        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Pinocchio loaded: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        self.current_pose = None
        self.current_q = None

        self.create_subscription(
            PoseStamped, '/end_effector_pose', self.ee_pose_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(
            PoseStamped, '/path_target_pose', self.target_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.get_logger().info('Cartesian Planner Ready')
        self.get_logger().info('  Send target to /path_target_pose')

    def ee_pose_cb(self, msg):
        self.current_pose = msg

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_q = np.array(
                [positions[n] for n in self.JOINT_NAMES])

    def target_cb(self, target_msg):
        if self.current_pose is None:
            self.get_logger().warn(
                'No EE pose yet — is forward_kinematics_node running?')
            return
        if self.current_q is None:
            self.get_logger().warn('No joint states yet')
            return

        # Extract poses
        sp = self.current_pose.pose
        start_pos = np.array([sp.position.x, sp.position.y, sp.position.z])
        start_quat = np.array([sp.orientation.x, sp.orientation.y,
                               sp.orientation.z, sp.orientation.w])

        ep = target_msg.pose
        end_pos = np.array([ep.position.x, ep.position.y, ep.position.z])
        end_quat = np.array([ep.orientation.x, ep.orientation.y,
                             ep.orientation.z, ep.orientation.w])

        # Generate Cartesian path (no IK yet)
        path = cartesian_path(
            start_pos, start_quat, end_pos, end_quat,
            vmax=self.VMAX, amax=self.AMAX, dt=self.DT,
            t_min=self.T_MIN, duration_scale=self.DURATION_SCALE)

        if len(path) < 2:
            self.get_logger().info('Already at target')
            return

        L_pos = float(np.linalg.norm(end_pos - start_pos))
        self.get_logger().info(
            f'Path: {len(path)} waypoints, '
            f'distance {L_pos:.3f}m, duration {path[-1][0]:.2f}s')

        # Solve IK along the path. Seed each step with the previous solution
        # to keep us in the same kinematic branch.
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.JOINT_NAMES

        # First waypoint = current joints exactly, zero velocity
        first_point = JointTrajectoryPoint()
        first_point.positions = self.current_q.tolist()
        first_point.velocities = [0.0] * 6
        first_point.time_from_start = Duration(sec=0, nanosec=0)
        traj_msg.points.append(first_point)

        q_seed = self.current_q.copy()
        consec_ik_fail = 0
        total_jumps = 0

        for i in range(1, len(path)):
            t, pos, rot = path[i]

            q_sol = compute_ik(
                self.model, self.data, self.ee_frame_id,
                pos, rot, q_seed)

            if q_sol is None:
                consec_ik_fail += 1
                if consec_ik_fail > self.MAX_IK_FAILURES:
                    self.get_logger().error(
                        f'Too many IK failures ({consec_ik_fail}), aborting')
                    return
                continue
            consec_ik_fail = 0

            # Two complementary guards
            max_step = joint_step_max(q_seed, q_sol)
            norm_step = joint_step_norm(q_seed, q_sol)
            if (max_step > self.MAX_PER_JOINT_JUMP
                    or norm_step > self.MAX_NORM_JUMP):
                total_jumps += 1
                self.get_logger().warn(
                    f'IK jump skipped: max={np.degrees(max_step):.1f}°, '
                    f'norm={norm_step:.2f} rad')
                if total_jumps > self.MAX_IK_JUMPS:
                    self.get_logger().error('Too many IK jumps, aborting')
                    return
                continue

            q_seed = q_sol.copy()

            point = JointTrajectoryPoint()
            point.positions = q_sol.tolist()
            point.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t - int(t)) * 1e9))
            traj_msg.points.append(point)

        if len(traj_msg.points) < 2:
            self.get_logger().error('Not enough valid waypoints')
            return

        self.traj_pub.publish(traj_msg)
        self.get_logger().info(
            f'Published {len(traj_msg.points)} waypoints, '
            f'{total_jumps} jumps skipped')


def main(args=None):
    rclpy.init(args=args)
    node = CartesianPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
