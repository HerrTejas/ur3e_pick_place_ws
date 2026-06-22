#!/usr/bin/env python3
"""Vision Pick and Place Node.

Runs the full pipeline end-to-end for a single color, specified with
the ``target_color`` ROS2 parameter:

    1. Wait for a detection from object_detector.py on
       ``/detected_object/{target_color}`` (camera frame).
    2. Transform it to ``base_link`` with TF2 (same approach as
       frame_transformer.py) and attach the tested downward grasp
       orientation.
    3. Solve IK (inverse_kinematics.compute_ik, seeded from the live
       joint state) for a pre-grasp / grasp / lift waypoint above the
       object.
    4. Drive the arm + gripper through those waypoints via
       FollowJointTrajectory actions (same action-client pattern as
       pick_and_place.py), then place at that color's calibrated place
       location and return home.

This node composes the existing nodes' pieces rather than
re-implementing them — no detection, TF or IK math is duplicated here.

Run:
    ros2 run ur3e_vision_pick_place forward_kinematics
    ros2 run ur3e_vision_pick_place object_detector
    ros2 run ur3e_vision_pick_place vision_pick_and_place \
        --ros-args -p target_color:=red

Author: Tejas
"""

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
import rclpy.time
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # needed for buffer.transform() to work with geometry_msgs

from ur3e_vision_pick_place.inverse_kinematics import compute_ik, load_pinocchio
from ur3e_vision_pick_place.robot_config import GRIPPER_JOINTS, HOME, JOINT_NAMES

#: Downward-facing gripper orientation, tested with the red box grasp
#: (see frame_transformer.py for the original calibration).
GRASP_ORIENTATION = {'x': 0.999, 'y': -0.008, 'z': 0.010, 'w': -0.033}

#: Cartesian offsets (metres) above the detected point, applied along
#: the world Z axis before/after the actual grasp.
PRE_GRASP_HEIGHT_M = 0.15
GRASP_CLEARANCE_M = 0.02
LIFT_HEIGHT_M = 0.20

#: Calibrated joint-space place locations per color (rad), reused from
#: pick_and_place.py's hard-coded object positions.
PLACE_POSITIONS: Dict[str, Dict[str, List[float]]] = {
    'red': {
        'place_pre': [0.3, -0.94, 0.8, -1.9, -1.57, -0.3],
        'place_down': [0.3, -0.94, 1.35, -1.9, -1.57, -0.3],
    },
    'green': {
        'place_pre': [0.5, -0.94, 0.8, -1.9, -1.57, -0.3],
        'place_down': [0.5, -0.94, 1.35, -1.9, -1.57, -0.3],
    },
    'blue': {
        'place_pre': [1.2, -0.94, 0.8, -1.9, -1.57, -0.3],
        'place_down': [1.2, -0.94, 1.35, -1.9, -1.57, -0.3],
    },
}

GRIPPER_OPEN: List[float] = [0.0]
GRIPPER_CLOSE: List[float] = [0.7]


class VisionPickAndPlace(Node):
    """Detect, pick, and place one object of the requested color."""

    def __init__(self) -> None:
        super().__init__('vision_pick_and_place')

        self.declare_parameter('target_color', 'red')
        target_color = self.get_parameter('target_color').get_parameter_value().string_value
        if target_color not in PLACE_POSITIONS:
            self.get_logger().error(
                f"Unknown target_color '{target_color}'. "
                f"Use one of: {list(PLACE_POSITIONS.keys())}")
            raise ValueError(f"Unknown target_color '{target_color}'")
        self.target_color = target_color

        # Pinocchio model, for compute_ik
        self.model, self.data, self.ee_frame_id = load_pinocchio()

        # TF2, to move the detected point from camera frame to base_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Live joint state, used to seed IK and as the action-goal start
        self.current_q = np.zeros(self.model.nq)
        self.joints_received = False
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # Latest detection for our color, in camera frame
        self.detected_point: Optional[PointStamped] = None
        self.create_subscription(
            PointStamped, f'/detected_object/{self.target_color}',
            self._detection_cb, 10)

        # Action clients, same controllers pick_and_place.py drives
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory')

        self.get_logger().info(f'Vision Pick and Place ready — target color: {self.target_color}')

    def _joint_state_cb(self, msg: JointState) -> None:
        """Track the current joint positions, used to seed IK."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in JOINT_NAMES:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            for i, name in enumerate(JOINT_NAMES):
                self.current_q[i] = positions[name]
            self.joints_received = True

    def _detection_cb(self, msg: PointStamped) -> None:
        """Store the latest detection of the target color."""
        self.detected_point = msg

    def wait_for_detection(self, timeout_sec: float = 30.0) -> bool:
        """Spin until a detection and a joint state are available.

        Args:
            timeout_sec: Maximum time to wait, seconds.

        Returns:
            True if both a detection and a joint state arrived in time.
        """
        self.get_logger().info(
            f'Waiting for /detected_object/{self.target_color} and /joint_states...')
        end_time = self.get_clock().now() + RclDuration(seconds=timeout_sec)
        while rclpy.ok() and self.get_clock().now() < end_time:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.detected_point is not None and self.joints_received:
                return True
        return False

    def transform_to_base_link(self, point: PointStamped) -> Optional[npt.NDArray[np.float64]]:
        """Transform a camera-frame point into base_link.

        Args:
            point: Detected object position, camera frame.

        Returns:
            (3,) position in base_link, or None if the TF lookup failed.
        """
        try:
            point.header.stamp = rclpy.time.Time().to_msg()
            point_base = self.tf_buffer.transform(
                point, 'base_link', timeout=RclDuration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF transform to base_link failed: {e}')
            return None
        return np.array([point_base.point.x, point_base.point.y, point_base.point.z])

    def solve_ik(self, position: npt.NDArray[np.float64]) -> Optional[npt.NDArray[np.float64]]:
        """Solve IK for a cartesian position with the fixed grasp orientation.

        Args:
            position: (3,) target position, base_link, metres.

        Returns:
            (6,) joint solution seeded from the current joints, or None
            if the solver couldn't converge.
        """
        quat = GRASP_ORIENTATION
        rot = pin.Quaternion(quat['w'], quat['x'], quat['y'], quat['z']).toRotationMatrix()
        return compute_ik(
            self.model, self.data, self.ee_frame_id,
            position, rot, self.current_q[:6])

    def move_arm(self, positions: List[float], duration: float = 2.0) -> bool:
        """Send the arm to a joint-space goal and wait for completion.

        Args:
            positions: (6,) target joint positions, radians.
            duration: Time to reach the goal, seconds.

        Returns:
            True if the action server accepted and completed the goal.
        """
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1) * 1e9))
        trajectory.points = [point]
        goal.trajectory = trajectory

        self.get_logger().info(f'Moving arm to: {[f"{p:.2f}" for p in positions]}')
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Arm goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.current_q[:6] = positions
        return True

    def move_gripper(self, positions: List[float], duration: float = 0.5) -> bool:
        """Send the gripper to a goal and wait for completion.

        Args:
            positions: Target gripper joint position(s).
            duration: Time to reach the goal, seconds.

        Returns:
            True if the action server accepted and completed the goal.
        """
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = GRIPPER_JOINTS

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1) * 1e9))
        trajectory.points = [point]
        goal.trajectory = trajectory

        action = 'Opening' if positions[0] < 0.3 else 'Closing'
        self.get_logger().info(f'{action} gripper...')
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def run(self) -> bool:
        """Execute the full detect -> pick -> place -> home sequence.

        Returns:
            True if every stage (detection, IK, and all motions)
            succeeded.
        """
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()

        if not self.wait_for_detection():
            self.get_logger().error(
                f'Timed out waiting for {self.target_color} detection / joint states.')
            return False

        base_position = self.transform_to_base_link(self.detected_point)
        if base_position is None:
            return False

        pre_grasp_q = self.solve_ik(base_position + [0, 0, PRE_GRASP_HEIGHT_M])
        grasp_q = self.solve_ik(base_position + [0, 0, GRASP_CLEARANCE_M])
        lift_q = self.solve_ik(base_position + [0, 0, LIFT_HEIGHT_M])
        if pre_grasp_q is None or grasp_q is None or lift_q is None:
            self.get_logger().error('IK failed to find a pick solution — aborting.')
            return False

        place = PLACE_POSITIONS[self.target_color]

        self.get_logger().info('=' * 50)
        self.get_logger().info(f'VISION PICK AND PLACE: {self.target_color}')
        self.get_logger().info('=' * 50)

        self.get_logger().info('--- HOME ---')
        self.move_arm(HOME)
        self.move_gripper(GRIPPER_OPEN)

        self.get_logger().info('--- PICK ---')
        self.move_arm(pre_grasp_q.tolist())
        self.move_arm(grasp_q.tolist())
        self.move_gripper(GRIPPER_CLOSE)
        self.move_arm(lift_q.tolist())

        self.get_logger().info('--- PLACE ---')
        self.move_arm(place['place_pre'])
        self.move_arm(place['place_down'])
        self.move_gripper(GRIPPER_OPEN)
        self.move_arm(place['place_pre'])

        self.get_logger().info('--- RETURN HOME ---')
        self.move_arm(HOME)

        self.get_logger().info('VISION PICK AND PLACE COMPLETE!')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = VisionPickAndPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
