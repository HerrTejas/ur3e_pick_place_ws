#!/usr/bin/env python3
"""Pick and Place Controller — state machine.

Coordinates the pick-and-place cycle. Does NOT do kinematics or
trajectory generation directly — delegates to the planner nodes.

Architecture:
  Long moves (HOME -> object, object -> place, mid-safe, home):
    Send target pose to /target_ee_pose, get joints back on /ik_solution,
    forward joints to /cmd_joint_positions (joint planner).
    This sequence is "joint-space with IK done up front."

  Vertical moves (LOWER, LIFT, RETREAT):
    Send target pose to /path_target_pose (Cartesian planner).
    Cartesian path stays in the same kinematic branch on Z-only motions.

This controller imports zero kinematics. It only knows poses, joint
arrays, and topic names.
"""

import numpy as np
import rclpy
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclDuration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — needed for buffer.transform()

from ur3e_vision_pick_place.helper_functions.trajectory_utils import (
    wrap_angle,
)


class PickAndPlaceController(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    MID_SAFE = [1.255, -1.2, 0.5, -1.0, -1.57, -0.3]

    PLACE_POSITIONS = {
        'red':   [0.15, 0.25],
        'green': [0.0, 0.25],
        'blue':  [-0.15, 0.25],
    }

    # Top-down grasp orientation (gripper pointing straight down)
    GRASP_ORIENTATION = {
        'x': 0.999, 'y': -0.008, 'z': 0.010, 'w': -0.033
    }

    # Heights (meters above object base)
    PRE_GRASP_HEIGHT = 0.10
    GRASP_HEIGHT = -0.01
    LIFT_HEIGHT = 0.10
    PLACE_DROP_HEIGHT = 0.03
    TOOL0_TO_FINGERTIP = 0.125

    # Arrival tolerances
    POSITION_TOLERANCE = 0.005
    JOINT_TOLERANCE = 0.05
    VELOCITY_TOLERANCE = 0.02
    ARRIVAL_CYCLES_REQUIRED = 5

    # IK request timeout
    IK_TIMEOUT_SEC = 2.0

    def __init__(self):
        super().__init__('pick_and_place_controller')

        # State machine
        self.state = 'IDLE'
        self.command_sent = False
        self.arrival_count = 0
        self.post_hold_state = None
        self.hold_timer = None

        # Active pick
        self.active_color = None
        self.target_object_base = None

        # Current motion target (for arrival check)
        self.current_target_xyz = None
        self.current_target_joints = None

        # State data from sensors
        self.current_ee_pose = None
        self.current_positions = None
        self.current_velocities = None

        # IK request tracking
        self.pending_ik_purpose = None  # 'PRE_GRASP' or 'MOVE_TO_PLACE'
        self.ik_request_time = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            String, '/pick_color', self.pick_color_cb, 10)
        for color in self.PLACE_POSITIONS:
            self.create_subscription(
                PointStamped, f'/detected_object/{color}',
                lambda msg, c=color: self.detection_cb(msg, c), 10)
        self.create_subscription(
            PoseStamped, '/end_effector_pose', self.ee_pose_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/ik_solution', self.ik_solution_cb, 10)

        # Publishers — to planner nodes and gripper
        self.ik_request_pub = self.create_publisher(
            PoseStamped, '/target_ee_pose', 10)
        self.cartesian_pub = self.create_publisher(
            PoseStamped, '/path_target_pose', 10)
        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/cmd_joint_positions', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # 10 Hz arrival checker
        self.create_timer(0.1, self.check_arrival)

        self.get_logger().info('Pick and Place Controller Ready')
        self.get_logger().info('  Send color to /pick_color')

    # ════════════════════════════════════════════════════════════
    #  Sensor callbacks
    # ════════════════════════════════════════════════════════════

    def ee_pose_cb(self, msg):
        self.current_ee_pose = msg

    def joint_state_cb(self, msg):
        positions = {}
        velocities = {}
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES:
                positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    velocities[name] = msg.velocity[i]
        if len(positions) == 6:
            self.current_positions = [positions[n] for n in self.JOINT_NAMES]
        if len(velocities) == 6:
            self.current_velocities = [velocities[n] for n in self.JOINT_NAMES]

    def ik_solution_cb(self, msg):
        """Receive IK solution and forward to joint planner."""
        if self.pending_ik_purpose is None:
            return  # Solution arrived but we weren't waiting

        if len(msg.data) != 6:
            self.get_logger().error(
                f'IK solution has {len(msg.data)} values, expected 6')
            self.pending_ik_purpose = None
            self.transition_to('IDLE')
            return

        joints = list(msg.data)
        self.get_logger().info(
            f'  Got IK solution for {self.pending_ik_purpose}: '
            f'{[f"{j:.2f}" for j in joints]}')

        self.pending_ik_purpose = None
        self.move_joints(joints)

    # ════════════════════════════════════════════════════════════
    #  Pick command + detection
    # ════════════════════════════════════════════════════════════

    def pick_color_cb(self, msg):
        if self.state != 'IDLE':
            self.get_logger().warn(f'Busy with {self.state}, ignoring')
            return
        color = msg.data.strip().lower()
        if color not in self.PLACE_POSITIONS:
            self.get_logger().warn(f'Unknown color: {color}')
            return
        self.active_color = color
        self.target_object_base = None
        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'  PICK AND PLACE: {color.upper()}')
        self.get_logger().info('=' * 50)
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
            f'  Detected {color}: '
            f'({self.target_object_base[0]:.3f}, '
            f'{self.target_object_base[1]:.3f}, '
            f'{self.target_object_base[2]:.3f})')
        self.transition_to('PRE_GRASP')

    # ════════════════════════════════════════════════════════════
    #  State machine
    # ════════════════════════════════════════════════════════════

    def transition_to(self, new_state):
        self.state = new_state
        self.command_sent = False
        self.current_target_xyz = None
        self.current_target_joints = None
        self.arrival_count = 0
        self.get_logger().info(f'  State -> {self.state}')
        self.execute_current_state()

    def execute_current_state(self):
        if self.command_sent:
            return

        obj = self.target_object_base
        z_off = self.TOOL0_TO_FINGERTIP

        if self.state == 'PRE_GRASP':
            # Long move: ask IK node for joints, then move via joint planner
            z = obj[2] + z_off + self.PRE_GRASP_HEIGHT
            self.request_ik(obj[0], obj[1], z, purpose='PRE_GRASP')

        elif self.state == 'LOWER':
            # Vertical move: Cartesian planner
            self.move_cartesian(
                obj[0], obj[1],
                obj[2] + z_off + self.GRASP_HEIGHT)

        elif self.state == 'CLOSE_GRIP':
            self.close_gripper()
            self.hold_timer = self.create_timer(2.0, self.gripper_done_cb)

        elif self.state == 'LIFT':
            # Vertical move: Cartesian
            self.move_cartesian(
                obj[0], obj[1],
                obj[2] + z_off + self.LIFT_HEIGHT)

        elif self.state == 'HOLD':
            self.get_logger().info(
                f'  Stabilizing before {self.post_hold_state}...')
            self.hold_timer = self.create_timer(1.5, self.hold_done_cb)

        elif self.state == 'MOVE_TO_PLACE':
            # Long move: IK -> joint planner
            px, py = self.PLACE_POSITIONS[self.active_color]
            place_z = obj[2] + z_off + self.LIFT_HEIGHT
            self.request_ik(px, py, place_z, purpose='MOVE_TO_PLACE')

        elif self.state == 'LOWER_PLACE':
            # Vertical move: Cartesian
            px, py = self.PLACE_POSITIONS[self.active_color]
            self.move_cartesian(
                px, py,
                obj[2] + z_off + self.PLACE_DROP_HEIGHT)

        elif self.state == 'OPEN_GRIP':
            self.open_gripper()
            self.hold_timer = self.create_timer(2.0, self.gripper_done_cb)

        elif self.state == 'RETREAT':
            # Vertical move: Cartesian
            px, py = self.PLACE_POSITIONS[self.active_color]
            self.move_cartesian(
                px, py,
                obj[2] + z_off + self.LIFT_HEIGHT)

        elif self.state == 'MID_SAFE':
            # Joint-space, but to a fixed config — no IK needed
            self.get_logger().info('  Moving to MID_SAFE')
            self.move_joints(self.MID_SAFE)

        elif self.state == 'HOME':
            self.get_logger().info('  Moving HOME')
            self.move_joints(self.HOME)

        elif self.state == 'IDLE':
            self.active_color = None
            self.target_object_base = None
            self.get_logger().info('=' * 50)
            self.get_logger().info('  COMPLETE')
            self.get_logger().info('=' * 50)
            self.get_logger().info('Waiting for next /pick_color command')

        self.command_sent = True

    def next_state(self):
        """State transitions on arrival."""
        # Tuple value = (intermediate HOLD, real next state)
        transitions = {
            'PRE_GRASP':     ('HOLD', 'LOWER'),
            'LOWER':         'CLOSE_GRIP',
            'LIFT':          ('HOLD', 'MOVE_TO_PLACE'),
            'MOVE_TO_PLACE': ('HOLD', 'LOWER_PLACE'),
            'LOWER_PLACE':   'OPEN_GRIP',
            'RETREAT':       ('HOLD', 'MID_SAFE'),
            'MID_SAFE':      ('HOLD', 'HOME'),
            'HOME':          'IDLE',
        }
        nxt = transitions.get(self.state)
        if nxt is None:
            return
        if isinstance(nxt, tuple):
            self.post_hold_state = nxt[1]
            self.transition_to('HOLD')
        else:
            self.transition_to(nxt)

    # ════════════════════════════════════════════════════════════
    #  Timer-driven transitions
    # ════════════════════════════════════════════════════════════

    def hold_done_cb(self):
        if self.hold_timer is not None:
            self.hold_timer.cancel()
            self.hold_timer = None
        if self.post_hold_state is not None:
            nxt = self.post_hold_state
            self.post_hold_state = None
            self.transition_to(nxt)

    def gripper_done_cb(self):
        if self.hold_timer is not None:
            self.hold_timer.cancel()
            self.hold_timer = None
        if self.state == 'CLOSE_GRIP':
            self.post_hold_state = 'LIFT'
            self.transition_to('HOLD')
        elif self.state == 'OPEN_GRIP':
            self.post_hold_state = 'RETREAT'
            self.transition_to('HOLD')

    # ════════════════════════════════════════════════════════════
    #  Arrival checker
    # ════════════════════════════════════════════════════════════

    def check_arrival(self):
        # IK timeout — if we asked for IK and nothing came back, abort
        if self.pending_ik_purpose is not None:
            elapsed = (self.get_clock().now() - self.ik_request_time
                       ).nanoseconds / 1e9
            if elapsed > self.IK_TIMEOUT_SEC:
                self.get_logger().error(
                    f'IK timeout ({elapsed:.1f}s) for '
                    f'{self.pending_ik_purpose}')
                self.pending_ik_purpose = None
                self.transition_to('IDLE')
            return

        # HOLD: settle until velocity drops below threshold for N cycles
        if self.state == 'HOLD':
            if self.current_velocities is None:
                return
            max_vel = max(abs(v) for v in self.current_velocities)
            if max_vel < self.VELOCITY_TOLERANCE:
                self.arrival_count += 1
                if self.arrival_count >= self.ARRIVAL_CYCLES_REQUIRED:
                    self.arrival_count = 0
                    if self.hold_timer is not None:
                        self.hold_timer.cancel()
                        self.hold_timer = None
                    self.get_logger().info('  Settled')
                    if self.post_hold_state is not None:
                        nxt = self.post_hold_state
                        self.post_hold_state = None
                        self.transition_to(nxt)
            else:
                self.arrival_count = 0
            return

        # Skip arrival check in non-motion states
        if self.state in ['IDLE', 'DETECTING', 'CLOSE_GRIP', 'OPEN_GRIP']:
            return

        arrived = False

        # Cartesian arrival: position error < tolerance
        if (self.current_target_xyz is not None
                and self.current_ee_pose is not None):
            ee = self.current_ee_pose.pose.position
            tx, ty, tz = self.current_target_xyz
            err = np.sqrt((ee.x - tx)**2 + (ee.y - ty)**2 + (ee.z - tz)**2)
            if err < self.POSITION_TOLERANCE:
                arrived = True

        # Joint arrival: max wrapped joint error < tolerance
        if (self.current_target_joints is not None
                and self.current_positions is not None):
            max_err = max(
                abs(wrap_angle(
                    self.current_positions[j]
                    - self.current_target_joints[j]))
                for j in range(6))
            if max_err < self.JOINT_TOLERANCE:
                arrived = True

        # Must also be slow when "arrived"
        if arrived and self.current_velocities is not None:
            max_vel = max(abs(v) for v in self.current_velocities)
            if max_vel > self.VELOCITY_TOLERANCE:
                arrived = False

        if arrived:
            self.arrival_count += 1
            if self.arrival_count >= self.ARRIVAL_CYCLES_REQUIRED:
                self.arrival_count = 0
                self.get_logger().info('  Arrived')
                self.next_state()
        else:
            self.arrival_count = 0

    # ════════════════════════════════════════════════════════════
    #  Action helpers
    # ════════════════════════════════════════════════════════════

    def make_pose_msg(self, x, y, z):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = self.GRASP_ORIENTATION['x']
        msg.pose.orientation.y = self.GRASP_ORIENTATION['y']
        msg.pose.orientation.z = self.GRASP_ORIENTATION['z']
        msg.pose.orientation.w = self.GRASP_ORIENTATION['w']
        return msg

    def request_ik(self, x, y, z, purpose: str):
        """Send target pose to IK node. Solution arrives on /ik_solution."""
        self.pending_ik_purpose = purpose
        self.ik_request_time = self.get_clock().now()
        msg = self.make_pose_msg(x, y, z)
        self.ik_request_pub.publish(msg)
        self.get_logger().info(
            f'  IK request for {purpose}: ({x:.3f}, {y:.3f}, {z:.3f})')

    def move_cartesian(self, x, y, z):
        """Send Cartesian target to path planner."""
        self.current_target_xyz = [x, y, z]
        self.current_target_joints = None
        msg = self.make_pose_msg(x, y, z)
        self.cartesian_pub.publish(msg)
        self.get_logger().info(
            f'  Cartesian to: ({x:.3f}, {y:.3f}, {z:.3f})')

    def move_joints(self, joints):
        """Send 6-DOF target to joint planner."""
        self.current_target_joints = joints
        self.current_target_xyz = None
        msg = Float64MultiArray()
        msg.data = list(joints)
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
