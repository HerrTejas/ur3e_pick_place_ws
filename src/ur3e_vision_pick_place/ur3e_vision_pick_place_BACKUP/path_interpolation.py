#!/usr/bin/env python3
"""
Path Interpolation Node — Cartesian Space

Linear position + SLERP orientation, time-scaled with
trapezoidal profile on POSITION DISTANCE ONLY.
Orientation follows the same time parameter via SLERP —
never dominates timing, always synchronized.

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

from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik
from ur3e_vision_pick_place.trapezoidal_planner import TrajectoryProfile


class PathInterpolation(Node):
    def __init__(self):
        super().__init__('path_interpolation')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Pinocchio loaded: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            return

        self.profile = TrajectoryProfile()

        # Profile parameters
        self.vmax = 0.2
        self.amax = 0.2
        self.dt = 0.05

        # Duration scaling: t = max(t_min, duration_scale * distance)
        self.t_min = 2.0
        self.duration_scale = 10.0

        # State
        self.current_pose = None
        self.current_q = np.zeros(6)
        self.joints_received = False

        self.create_subscription(
            PoseStamped, '/end_effector_pose', self.ee_pose_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(
            PoseStamped, '/path_target_pose', self.target_cb, 10)

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.get_logger().info('Path Interpolation Node Ready!')
        self.get_logger().info('  FK pose from: /end_effector_pose')
        self.get_logger().info('  IK from: inverse_kinematics.compute_ik()')
        self.get_logger().info('  Profile from: trapezoidal_planner.TrajectoryProfile')
        self.get_logger().info('  Send target to: /path_target_pose')

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

    def plan_and_execute(self, target_msg):
        # Start pose (from FK node)
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

        # Distances
        L_pos = np.linalg.norm(end_pos - start_pos)
        L_ori = (r_start.inv() * r_end).magnitude()

        self.get_logger().info(f'Position distance: {L_pos:.4f} m')
        self.get_logger().info(f'Orientation distance: {L_ori:.4f} rad')

        if L_pos < 1e-6 and L_ori < 1e-6:
            self.get_logger().info('Already at target.')
            return

        # Choose primary distance for time-scaling
        # Position controls timing. Orientation just follows via SLERP.
        # If position is ~zero (pure rotation), use orientation instead.
        if L_pos > 1e-6:
            L_primary = L_pos
        else:
            L_primary = L_ori

        # Desired duration: proportional to distance, with minimum
        desired_duration = max(self.t_min, self.duration_scale * L_primary)

        # Scale vmax/amax if profile would be too short
        # Check: L / vmax is roughly the cruise-phase time
        if L_primary / self.vmax < desired_duration * 0.75:
            vmax = L_primary / (desired_duration * 0.75)
            amax = vmax / (desired_duration * 0.25)
        else:
            vmax = self.vmax
            amax = self.amax

        # Generate trapezoidal profile on primary distance only
        t_array, s_array, t_total = self.profile.trapezoid_time_scaled(
            L_primary, vmax, amax, self.dt)

        self.get_logger().info(
            f'Trajectory: {len(t_array)} waypoints, {t_total:.2f}s')

        # Setup SLERP
        rots = R.concatenate([r_start, r_end])
        slerp = Slerp([0.0, 1.0], rots)

        # Build trajectory
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        # First point: current joints exactly, velocity = 0
        first_point = JointTrajectoryPoint()
        first_point.positions = self.current_q[:6].tolist()
        first_point.velocities = [0.0] * 6
        first_point.time_from_start = Duration(sec=0, nanosec=0)
        traj_msg.points.append(first_point)

        q_seed = self.current_q.copy()
        ik_failures = 0

        for i in range(len(t_array)):
            if i == 0:
                continue

            # Single normalized parameter: 0 → 1
            # Position and orientation both use this same parameter
            t_interp = min(s_array[i] / L_primary, 1.0) if L_primary > 1e-6 else 1.0

            # Interpolate position (linear) and orientation (SLERP)
            pos = (1.0 - t_interp) * start_pos + t_interp * end_pos
            r_interp = slerp(t_interp)

            # IK
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

            # Check per-joint wrapped angular distance
            # Raw difference is WRONG for joints (misses wraparound)
            # Per-joint check catches single-joint spikes that norm misses
            diffs = [
                (q_sol[j] - q_seed[j] + np.pi) % (2 * np.pi) - np.pi
                for j in range(6)
            ]
            max_joint_jump = max(abs(d) for d in diffs)
            norm_jump = float(np.linalg.norm(diffs))

            if max_joint_jump > 0.4 or norm_jump > 1.0:
                ik_failures += 1
                self.get_logger().warn(
                    f'IK jump: joint max {np.degrees(max_joint_jump):.1f}°, skipping')
                if ik_failures > 10:
                    self.get_logger().error('Too many IK jumps, aborting.')
                    return
                continue

            q_seed = q_sol.copy()

            point = JointTrajectoryPoint()

            #check if norm is greater than 1 and return

            point = JointTrajectoryPoint()
            point.positions = q_sol.tolist()
            t = t_array[i]
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9))
            traj_msg.points.append(point)

        if len(traj_msg.points) < 2:
            self.get_logger().error('Not enough valid waypoints!')
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