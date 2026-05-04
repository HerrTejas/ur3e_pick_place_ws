#!/usr/bin/env python3
"""
Pick and Place Controller — Industrial-Style State Machine

Cartesian for vertical moves.
Joint-space for lateral moves.
HOLD states at every motion-type boundary.
Dynamic MID_SAFE before HOME to prevent arm stretch.

States:
  IDLE → PRE_GRASP → LOWER → CLOSE_GRIP → HOLD → LIFT →
  HOLD → MOVE_TO_PLACE (joint) → HOLD → PRE_PLACE (Cartesian) →
  LOWER_PLACE → OPEN_GRIP → HOLD → RETREAT →
  HOLD → MID_SAFE (joint, dynamic) → HOLD → HOME (joint) → IDLE

Author: Tejas
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import tf2_ros
import tf2_geometry_msgs
from rclpy.duration import Duration as RclDuration
import numpy as np
import pinocchio as pin

from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik


class PickAndPlaceController(Node):
    def __init__(self):
        super().__init__('pick_and_place_controller')

        # ── State machine ─────────────────────────────────────────
        self.state = 'IDLE'

        # ── TF2 ───────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Pinocchio ─────────────────────────────────────────────
        try:
            self.pin_model, self.pin_data, self.pin_ee_id = load_pinocchio()
            self.get_logger().info('Pinocchio loaded for place IK')
        except Exception as e:
            self.get_logger().error(f'Failed to load Pinocchio: {e}')
            return

        # ── Robot config ──────────────────────────────────────────
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.MID_SAFE = [1.255, -1.2, 0.5, -1.0, -1.57, -0.3]

        # Place positions per color
        self.place_positions = {
            'red':   [0.15, 0.25],
            'green': [0.0, 0.25],
            'blue':  [-0.15, 0.25],
        }

        # Grasp orientation
        self.grasp_orientation = {
            'x': 0.999, 'y': -0.008, 'z': 0.010, 'w': -0.033
        }
        self.grasp_rot = pin.Quaternion(
            self.grasp_orientation['w'],
            self.grasp_orientation['x'],
            self.grasp_orientation['y'],
            self.grasp_orientation['z']
        ).toRotationMatrix()

        # Heights
        self.pre_grasp_height = 0.10
        self.grasp_height = -0.01
        self.lift_height = 0.10
        self.place_drop_height = 0.03

        # tool0 → fingertip offset
        self.tool0_to_fingertip = 0.125

        # ── Arrival — N consecutive cycles ────────────────────────
        self.position_tolerance = 0.005
        self.joint_tolerance = 0.05
        self.velocity_tolerance = 0.02
        self.arrival_cycles_required = 5
        self.arrival_count = 0

        # IK cost threshold
        self.ik_cost_threshold = 3.0

        # ── State data ────────────────────────────────────────────
        self.active_color = None
        self.target_object_base = None
        self.current_target_xyz = None
        self.current_target_joints = None
        self.current_ee_pose = None
        self.current_positions = None
        self.current_velocities = None
        self.gripper_timer = None
        self.command_sent = False
        self.post_hold_state = None
        self.place_target_xyz = None

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(
            String, '/pick_color', self.pick_color_cb, 10)

        self.colors = ['red', 'green', 'blue']
        for color in self.colors:
            self.create_subscription(
                PointStamped, f'/detected_object/{color}',
                lambda msg, c=color: self.detection_cb(msg, c), 10)

        self.create_subscription(
            PoseStamped, '/end_effector_pose', self.ee_pose_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)

        # ── Publishers ────────────────────────────────────────────
        self.cartesian_pub = self.create_publisher(
            PoseStamped, '/path_target_pose', 10)
        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/cmd_joint_positions', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # ── Arrival checker at 10 Hz ──────────────────────────────
        self.create_timer(0.1, self.check_arrival)

        self.get_logger().info('Pick and Place Controller Ready!')
        self.get_logger().info('  Send color to /pick_color to start')

    # ══════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def joint_cost(self, q_current, q_target):
        return sum(abs(self.wrap_angle(q_target[j] - q_current[j]))
                   for j in range(6))

    # ══════════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════════

    def ee_pose_cb(self, msg):
        self.current_ee_pose = msg

    def joint_state_cb(self, msg):
        positions = {}
        velocities = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    velocities[name] = msg.velocity[i]
        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]
        if len(velocities) == 6:
            self.current_velocities = [velocities[name] for name in self.joint_names]

    def pick_color_cb(self, msg):
        if self.state != 'IDLE':
            self.get_logger().warn(f'Busy with {self.state}, ignoring.')
            return

        color = msg.data.strip().lower()
        if color not in self.colors:
            self.get_logger().warn(f'Unknown color: {color}')
            return

        self.active_color = color
        self.target_object_base = None
        self.place_target_xyz = None
        self.get_logger().info(f'')
        self.get_logger().info(f'{"="*50}')
        self.get_logger().info(f'  PICK AND PLACE: {color.upper()}')
        self.get_logger().info(f'{"="*50}')
        self.transition_to('DETECTING')

    def detection_cb(self, msg, color):
        if self.state != 'DETECTING':
            return
        if color != self.active_color:
            return

        try:
            msg.header.stamp = rclpy.time.Time().to_msg()
            point_base = self.tf_buffer.transform(
                msg, 'base_link', timeout=RclDuration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}')
            return

        self.target_object_base = [
            point_base.point.x,
            point_base.point.y,
            point_base.point.z
        ]
        self.get_logger().info(
            f'  Detected {color}: ({self.target_object_base[0]:.3f}, '
            f'{self.target_object_base[1]:.3f}, '
            f'{self.target_object_base[2]:.3f})')

        self.transition_to('PRE_GRASP')

    # ══════════════════════════════════════════════════════════════
    #  State transitions
    # ══════════════════════════════════════════════════════════════

    def transition_to(self, new_state):
        self.state = new_state
        self.command_sent = False
        self.current_target_xyz = None
        self.current_target_joints = None
        self.arrival_count = 0
        self.get_logger().info(f'  State → {self.state}')
        self.execute_current_state()

    def execute_current_state(self):
        if self.command_sent:
            return

        obj = self.target_object_base
        z_off = self.tool0_to_fingertip

        if self.state == 'PRE_GRASP':
            self.move_cartesian(
                obj[0], obj[1],
                obj[2] + z_off + self.pre_grasp_height)

        elif self.state == 'LOWER':
            self.move_cartesian(
                obj[0], obj[1],
                obj[2] + z_off + self.grasp_height)

        elif self.state == 'CLOSE_GRIP':
            self.close_gripper()
            self.gripper_timer = self.create_timer(
                2.0, self.gripper_done_cb)

        elif self.state == 'LIFT':
            self.move_cartesian(
                obj[0], obj[1],
                obj[2] + z_off + self.lift_height)

        elif self.state == 'HOLD':
            self.get_logger().info(
                f'  Stabilizing before {self.post_hold_state}...')
            self.gripper_timer = self.create_timer(
                1.5, self.hold_done_cb)

        elif self.state == 'MOVE_TO_PLACE':
            px, py = self.place_positions[self.active_color]
            place_z = obj[2] + z_off + self.lift_height

            self.place_target_xyz = [px, py, place_z]

            target_pos = np.array([px, py, place_z])
            seed = np.array(self.current_positions[:6])

            q_sol = compute_ik(
                self.pin_model, self.pin_data, self.pin_ee_id,
                target_pos, self.grasp_rot, seed)

            if q_sol is None:
                self.get_logger().error('IK failed for place position!')
                self.transition_to('IDLE')
                return

            cost = self.joint_cost(seed, q_sol)
            self.get_logger().info(f'  IK cost: {cost:.2f} rad')

            if cost > self.ik_cost_threshold:
                self.get_logger().error(
                    f'IK cost too high ({cost:.2f} > {self.ik_cost_threshold})!')
                self.transition_to('IDLE')
                return

            self.get_logger().info(
                f'  Joint-space to: ({px:.3f}, {py:.3f}, {place_z:.3f})')
            self.move_joints(q_sol.tolist())

        elif self.state == 'PRE_PLACE':
            px, py, pz = self.place_target_xyz
            self.move_cartesian(px, py, pz)

        elif self.state == 'LOWER_PLACE':
            px, py = self.place_positions[self.active_color]
            self.move_cartesian(
                px, py,
                obj[2] + z_off + self.place_drop_height)

        elif self.state == 'OPEN_GRIP':
            self.open_gripper()
            self.gripper_timer = self.create_timer(
                2.0, self.gripper_done_cb)

        elif self.state == 'RETREAT':
            px, py = self.place_positions[self.active_color]
            self.move_cartesian(
                px, py,
                obj[2] + z_off + self.lift_height)

        elif self.state == 'MID_SAFE':
            self.get_logger().info('  Moving to MID_SAFE')
            self.move_joints(self.MID_SAFE)

        elif self.state == 'HOME':
            self.get_logger().info('  Moving HOME')
            self.move_joints(self.HOME)

        elif self.state == 'IDLE':
            self.active_color = None
            self.target_object_base = None
            self.place_target_xyz = None
            self.get_logger().info(f'{"="*50}')
            self.get_logger().info(f'  COMPLETE')
            self.get_logger().info(f'{"="*50}')
            self.get_logger().info('Waiting for next /pick_color command...')

        self.command_sent = True

    # ── Timer callbacks ───────────────────────────────────────────

    def hold_done_cb(self):
        if self.gripper_timer is not None:
            self.gripper_timer.cancel()
            self.gripper_timer = None
        if self.post_hold_state is not None:
            self.transition_to(self.post_hold_state)
            self.post_hold_state = None

    def gripper_done_cb(self):
        if self.gripper_timer is not None:
            self.gripper_timer.cancel()
            self.gripper_timer = None

        if self.state == 'CLOSE_GRIP':
            self.post_hold_state = 'LIFT'
            self.transition_to('HOLD')
        elif self.state == 'OPEN_GRIP':
            self.post_hold_state = 'RETREAT'
            self.transition_to('HOLD')

    # ══════════════════════════════════════════════════════════════
    #  Arrival checker — N consecutive cycles
    # ══════════════════════════════════════════════════════════════

    def check_arrival(self):
        # Adaptive HOLD
        if self.state == 'HOLD':
            if self.current_velocities is not None:
                max_vel = max(abs(v) for v in self.current_velocities)
                if max_vel < self.velocity_tolerance:
                    self.arrival_count += 1
                    if self.arrival_count >= self.arrival_cycles_required:
                        self.arrival_count = 0
                        if self.gripper_timer is not None:
                            self.gripper_timer.cancel()
                            self.gripper_timer = None
                        self.get_logger().info('  Settled (adaptive)')
                        if self.post_hold_state is not None:
                            next_s = self.post_hold_state
                            self.post_hold_state = None
                            self.transition_to(next_s)
                else:
                    self.arrival_count = 0
            return

        if self.state in ['IDLE', 'DETECTING', 'CLOSE_GRIP', 'OPEN_GRIP']:
            return

        arrived = False

        # Cartesian check
        if self.current_target_xyz is not None and self.current_ee_pose is not None:
            ee = self.current_ee_pose.pose.position
            tx, ty, tz = self.current_target_xyz

            pos_error = np.sqrt(
                (ee.x - tx) ** 2 +
                (ee.y - ty) ** 2 +
                (ee.z - tz) ** 2)

            if pos_error < self.position_tolerance:
                arrived = True

        # Joint check (wrapped)
        if self.current_target_joints is not None and self.current_positions is not None:
            max_error = max(
                abs(self.wrap_angle(
                    self.current_positions[j] - self.current_target_joints[j]))
                for j in range(6))

            if max_error < self.joint_tolerance:
                arrived = True

        # Velocity check
        if arrived and self.current_velocities is not None:
            max_vel = max(abs(v) for v in self.current_velocities)
            if max_vel > self.velocity_tolerance:
                arrived = False

        # N consecutive cycles
        if arrived:
            self.arrival_count += 1
            if self.arrival_count >= self.arrival_cycles_required:
                self.arrival_count = 0
                self.get_logger().info(
                    f'  Arrived ({self.arrival_cycles_required} cycles)')
                self.next_state()
        else:
            self.arrival_count = 0

    def next_state(self):
        transitions = {
            'PRE_GRASP': 'LOWER',
            'LOWER': 'CLOSE_GRIP',
            'LIFT': ('HOLD', 'MOVE_TO_PLACE'),
            'MOVE_TO_PLACE': ('HOLD', 'PRE_PLACE'),
            'PRE_PLACE': 'LOWER_PLACE',
            'LOWER_PLACE': 'OPEN_GRIP',
            'RETREAT': ('HOLD', 'MID_SAFE'),
            'MID_SAFE': ('HOLD', 'HOME'),
            'HOME': 'IDLE',
        }

        next_s = transitions.get(self.state)
        if next_s is None:
            return

        if isinstance(next_s, tuple):
            self.post_hold_state = next_s[1]
            self.transition_to('HOLD')
        else:
            self.transition_to(next_s)

    # ══════════════════════════════════════════════════════════════
    #  Actions
    # ══════════════════════════════════════════════════════════════

    def move_cartesian(self, x, y, z):
        self.current_target_xyz = [x, y, z]
        self.current_target_joints = None

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'

        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z

        pose_msg.pose.orientation.x = self.grasp_orientation['x']
        pose_msg.pose.orientation.y = self.grasp_orientation['y']
        pose_msg.pose.orientation.z = self.grasp_orientation['z']
        pose_msg.pose.orientation.w = self.grasp_orientation['w']

        self.cartesian_pub.publish(pose_msg)
        self.get_logger().info(f'  Cartesian to: ({x:.3f}, {y:.3f}, {z:.3f})')

    def move_joints(self, joints):
        self.current_target_joints = joints
        self.current_target_xyz = None

        msg = Float64MultiArray()
        msg.data = joints
        self.joint_pub.publish(msg)

    def close_gripper(self):
        traj = JointTrajectory()
        traj.joint_names = ['rh_r1_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.7]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.gripper_pub.publish(traj)
        self.get_logger().info('  Gripper: CLOSING')

    def open_gripper(self):
        traj = JointTrajectory()
        traj.joint_names = ['rh_r1_joint']
        point = JointTrajectoryPoint()
        point.positions = [0.0]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points = [point]
        self.gripper_pub.publish(traj)
        self.get_logger().info('  Gripper: OPENING')


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()