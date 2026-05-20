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
import threading
from scipy.spatial.transform import Rotation as R, Slerp
from ament_index_python.packages import get_package_share_directory
import os
import subprocess

# Import math from existing nodes — no duplication
from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik
from ur3e_vision_pick_place.trapezoidal_planner import TrajectoryProfile
from ur3e_vision_pick_place.inverse_kinematics import unwrap_solution


class PathInterpolation(Node):
    def __init__(self):
        super().__init__('path_interpolation')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        pkg_path = get_package_share_directory("ur_description")

        xacro_path = os.path.join(pkg_path, "urdf", "ur.urdf.xacro")
        urdf_path = "/tmp/ur3e.urdf"

        # Convert xacro → urdf
        subprocess.run([
            "xacro",
            xacro_path,
            "name:=ur",
            "ur_type:=ur3e",
            "prefix:=",
        ], stdout=open(urdf_path, "w"), check=True)

        # Load Pinocchio (same function IK node uses)
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio(urdf_path)
            self.get_logger().info(f'Pinocchio loaded: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            return

        # Trapezoid profile (same class trapezoidal_planner uses)
        self.profile = TrajectoryProfile()

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
            '/joint_trajectory_controller/joint_trajectory', 10)

        # Pub: debug joint commands (per-step) for PlotJuggler comparison
        self.cmd_joints_pub = self.create_publisher(
            JointState, '/commanded_joint_states', 10)

        # State for debug replay timer
        self._debug_points = []
        self._debug_times = []
        self._debug_idx = 0
        self._debug_timer = None

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

            if not self.joints_received:
                # One-time diagnostic: confirm names and ordering are correct
                self.get_logger().info(
                    f'[JOINT INIT] All joint names from /joint_states: {list(msg.name)}'
                )
                self.get_logger().info(
                    f'[JOINT INIT] current_q mapping (should match robot pose):'
                )
                for i, name in enumerate(self.joint_names):
                    self.get_logger().info(
                        f'  current_q[{i}] = {self.current_q[i]:.4f} rad  ← {name}'
                    )

            self.joints_received = True

    def target_cb(self, msg):
        if self.current_pose is None:
            self.get_logger().warn('No EE pose yet — is FK node running?')
            return
        if not self.joints_received:
            self.get_logger().warn('No joint states yet!')
            return
        self.plan_and_execute(msg)


    # Poll /joint_states until actual position matches expected goal, THEN seed IK

    def wait_for_settle(self, goal_q, timeout=5.0, tol=0.05):
        start = self.get_clock().now()
        while (self.get_clock().now() - start).nanoseconds < timeout * 1e9:
            if np.allclose(self.current_q[:6], goal_q, atol=tol):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)  # ✅ lets callbacks fire
        self.get_logger().warn('Settle timeout — proceeding anyway')
        return False


    # ── Planning ──────────────────────────────────────────────────

    def plan_and_execute(self, target_msg):
        self.wait_for_settle(self.current_q.copy())

        # Diagnostic: confirm IK seed matches the EE pose we're planning from
        self.get_logger().info(
            f'[SEED CHECK] current_q (IK seed): {np.round(self.current_q, 4).tolist()}'
        )
        sp = self.current_pose.pose
        self.get_logger().info(
            f'[SEED CHECK] FK start pose: pos=[{sp.position.x:.4f}, {sp.position.y:.4f}, {sp.position.z:.4f}]'
        )
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
        self.get_logger().info(
            f'Target pos: [{end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f}]  '
            f'dist_from_base: {np.linalg.norm(end_pos):.4f} m  '
            f'(UR3e max ~0.50 m from shoulder)'
        )

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
        q_last_good = self.current_q.copy()  # last successfully solved config
        ik_failures = 0
        # Abort only if >30% of waypoints fail, prevents over-eager abort
        max_failures = max(6, len(t_array) // 3)

        # Fallback seeds: diverse configs covering high-z / extended poses
        fallback_seeds = [
            np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]),   # canonical elbow-up
            np.array([0.0, -np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0.0]),  # mid-reach
            np.array([0.0, -1.0, 0.5, -1.0, -np.pi/2, 0.0]),       # higher-z bias
            np.zeros(6),
        ]

        for i in range(len(t_array)):
            # ── BUG FIX: use t_pos for position, t_ori for orientation separately
            # Previously both used a single t_interp=max(t_pos,t_ori) which
            # created geometrically inconsistent intermediate poses → IK failures
            t_pos_s = (s_pos[i] / L_pos) if L_pos > 1e-6 else 1.0
            t_ori_s = (s_ori[i] / L_ori) if L_ori > 1e-6 else 1.0
            t_pos_s = min(max(t_pos_s, 0.0), 1.0)
            t_ori_s = min(max(t_ori_s, 0.0), 1.0)

            # Cartesian interpolation: linear pos + SLERP orientation (INDEPENDENT)
            pos = (1.0 - t_pos_s) * start_pos + t_pos_s * end_pos
            r_interp = slerp(t_ori_s)
            rot_mat = r_interp.as_matrix()

            # IK — primary seed is last solved config (warm start)
            q_sol = compute_ik(
                self.model, self.data, self.ee_frame_id,
                pos, rot_mat, q_seed)

            # Fallback: try diverse seeds with more aggressive solver settings
            if q_sol is None:
                for fb_seed in fallback_seeds:
                    q_sol = compute_ik(
                        self.model, self.data, self.ee_frame_id,
                        pos, rot_mat, fb_seed,
                        max_iter=800)
                    if q_sol is not None:
                        self.get_logger().info(
                            f'IK recovered at waypoint {i} using fallback seed')
                        break

            if q_sol is None:
                ik_failures += 1
                self.get_logger().warn(
                    f'IK FAILED at waypoint {i}/{len(t_array)}: '
                    f'target_pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] '
                    f't_pos={t_pos_s:.3f} t_ori={t_ori_s:.3f} '
                    f'seed_q={np.round(q_seed, 3).tolist()}'
                )
                if ik_failures > max_failures:
                    self.get_logger().error(
                        f'Too many IK failures ({ik_failures}/{len(t_array)}), aborting.')
                    return
                # Keep seed at last GOOD solution — do NOT drift it toward
                # current_q (start config), that pulls the seed backwards
                q_seed = q_last_good.copy()
                continue

            q_sol = unwrap_solution(q_sol, q_seed)
            q_seed = q_sol.copy()
            q_last_good = q_sol.copy()

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

        # ── Debug: replay commanded joints on /commanded_joint_states ──
        # Publishes each waypoint's joint values at the correct wall-clock time
        # so PlotJuggler can compare commanded vs /joint_states side-by-side.
        self._start_debug_replay(traj_msg)


    def _start_debug_replay(self, traj_msg):
        """
        Publish commanded joints to /commanded_joint_states at wall-clock
        intervals matching time_from_start, so PlotJuggler can overlay them
        against /joint_states for each joint individually.
        """
        # Cancel any previous replay
        if self._debug_timer is not None:
            self._debug_timer.cancel()
            self._debug_timer = None

        # Build flat lists of (delay_sec, positions)
        self._debug_points = [list(p.positions) for p in traj_msg.points]
        self._debug_times = [
            p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
            for p in traj_msg.points
        ]
        self._debug_idx = 0

        # Publish first point immediately, then schedule the rest via timer
        self._publish_debug_point()

    def _publish_debug_point(self):
        if self._debug_idx >= len(self._debug_points):
            if self._debug_timer is not None:
                self._debug_timer.cancel()
                self._debug_timer = None
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self._debug_points[self._debug_idx]

        self.cmd_joints_pub.publish(msg)

        self._debug_idx += 1
        if self._debug_idx >= len(self._debug_points):
            if self._debug_timer is not None:
                self._debug_timer.cancel()
                self._debug_timer = None
            return

        # Time until next point
        dt_next = self._debug_times[self._debug_idx] - self._debug_times[self._debug_idx - 1]
        dt_next = max(dt_next, 0.001)  # at least 1 ms

        self._debug_timer = self.create_timer(dt_next, self._debug_timer_cb)

    def _debug_timer_cb(self):
        # One-shot: cancel immediately, then publish + schedule next
        if self._debug_timer is not None:
            self._debug_timer.cancel()
            self._debug_timer = None
        self._publish_debug_point()


def main(args=None):
    rclpy.init(args=args)
    node = PathInterpolation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()