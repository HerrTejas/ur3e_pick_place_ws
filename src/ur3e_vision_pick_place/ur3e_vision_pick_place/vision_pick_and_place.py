#!/usr/bin/env python3
"""Vision Pick and Place Node.

Runs the full pipeline end-to-end for a single color, specified with
the ``target_color`` ROS2 parameter:

    1. Wait for a detection from object_detector.py on
       ``/detected_object/{target_color}`` (camera frame).
    2. Transform it to ``base_link`` with TF2 (same approach as
       frame_transformer.py) and attach the tested downward grasp
       orientation.
    3. Solve IK (helper_functions.kinematics.compute_ik, seeded from
       the live joint state) for a pre-grasp / grasp / lift waypoint
       above the object.
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
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # needed for buffer.transform() to work with geometry_msgs

from ur3e_vision_pick_place.helper_functions.kinematics import compute_ik, load_pinocchio
from ur3e_vision_pick_place.helper_functions.trajectory_profile import (
    joint_trapezoid, shortest_angular_distance,
)
from ur3e_vision_pick_place.robot_config import GRIPPER_JOINTS, HOME, JOINT_NAMES
from ur3e_vision_pick_place.ros_utils import profile_to_trajectory_msg, seconds_to_duration

#: Downward-facing gripper orientation, tested with the red box grasp
#: (see frame_transformer.py for the original calibration).
GRASP_ORIENTATION = {'x': 0.999, 'y': -0.008, 'z': 0.010, 'w': -0.033}

#: Cartesian Z offsets (metres) from the detected point, applied along
#: base_link Z. The detector reports the object's *top face*, so the
#: grasp offset is negative: the fingertips (EE frame = rh_p12_rn_ee)
#: descend to roughly mid-height of the 6 cm box for a secure grip.
#: Pre-grasp/lift stay well above. Tune GRASP_CLEARANCE_M per object
#: height if you change the boxes.
PRE_GRASP_HEIGHT_M = 0.15
GRASP_CLEARANCE_M = -0.03
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

#: Motion profile limits for arm moves. Duration is scaled so the
#: fastest joint stays under MAX_JOINT_VEL, never shorter than MIN_MOVE_SEC.
#: Kept deliberately gentle: a lower cruise speed and a longer floor time
#: soften acceleration, so contact with a box (pick/place) doesn't spike
#: the physics and the arm never whips through a large transit.
MAX_JOINT_VEL = 0.35  # rad/s
MIN_MOVE_SEC = 3.0    # s
TRAJ_DT = 0.05        # s, sample period of the generated profile


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

    def solve_ik(
        self, position: npt.NDArray[np.float64],
        seed: Optional[npt.NDArray[np.float64]] = None,
    ) -> Optional[npt.NDArray[np.float64]]:
        """Solve IK for a cartesian position with the fixed grasp orientation.

        Args:
            position: (3,) target position, base_link, metres.
            seed: (6,) IK seed. Defaults to the current joints; pass the
                previous waypoint's solution to chain pre-grasp -> grasp
                -> lift so each stays on the same IK branch (avoids the
                arm flipping configuration between nearby points).

        Returns:
            (6,) joint solution, or None if the solver couldn't converge.
        """
        if seed is None:
            seed = self.current_q[:6]
        quat = GRASP_ORIENTATION
        rot = pin.Quaternion(quat['w'], quat['x'], quat['y'], quat['z']).toRotationMatrix()
        # joint_names selects just the 6 arm joints — the URDF also
        # contains the gripper's revolute joints.
        return compute_ik(
            self.model, self.data, self.ee_frame_id,
            position, rot, seed, joint_names=JOINT_NAMES)

    def _send_and_wait(self, client, goal, expected_sec: float, label: str) -> bool:
        """Send a trajectory goal and wait, with a timeout and status check.

        Returns False (instead of blocking forever) if the goal is
        rejected, the controller never returns a result within the
        expected motion time plus a margin, or the motion ends in any
        state other than SUCCEEDED — e.g. when the arm jams against an
        object and the controller can't reach the goal. Without this the
        node hung indefinitely on a stuck/aborted motion.

        Args:
            client: The FollowJointTrajectory action client to use.
            goal: The populated goal message.
            expected_sec: Planned motion duration, seconds.
            label: Human-readable name for log messages.

        Returns:
            True only if the motion completed successfully.
        """
        timeout = expected_sec + 5.0  # margin for accel/comms/settling

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'{label}: goal rejected or send timed out.')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        result = result_future.result()
        if result is None:
            self.get_logger().error(
                f'{label}: no result within {timeout:.1f}s — likely stuck/collided. '
                'Cancelling and aborting.')
            goal_handle.cancel_goal_async()
            return False
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f'{label}: motion did not succeed (status {result.status}). Aborting.')
            return False
        return True

    def move_arm(self, positions: List[float]) -> bool:
        """Send the arm to a joint-space goal and wait for completion.

        Args:
            positions: (6,) target joint positions, radians.

        Returns:
            True if the action server accepted and completed the goal.
        """
        # Snap each target joint to its nearest co-terminal angle (within
        # +/-pi of where the joint is now). joint_trapezoid interpolates
        # linearly start->goal, so a raw target more than half a turn away
        # (e.g. a hard-coded place pose vs. the current wrist winding) makes
        # that joint spin the LONG way around — the wild sweep between pick
        # and place. Taking the shortest rotation keeps every move direct.
        start = self.current_q[:6]
        goal = np.array([
            start[j] + shortest_angular_distance(start[j], positions[j])
            for j in range(6)
        ])

        # Full position+velocity trapezoid instead of a single target
        # point: a lone point makes the controller pick its own (constant
        # velocity) interpolation, so speed jumps 0->v at the start and
        # v->0 at the end — the jerk you feel.
        times, traj_positions, velocities = joint_trapezoid(
            start, goal, MAX_JOINT_VEL, TRAJ_DT, MIN_MOVE_SEC)

        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory = profile_to_trajectory_msg(
            JOINT_NAMES, times, traj_positions, velocities)
        expected_sec = float(times[-1])

        # Diagnostic: the largest per-joint move and the planned duration.
        # A move that jerks "super fast" is either (a) unexpectedly large
        # here — an IK branch flip sending a joint most of a turn — or
        # (b) large here but the arm still finishes way before expected_sec,
        # meaning the controller is racing the trajectory (a clock / speed-
        # scaling problem, not our profile). Compare this against how long
        # the move actually takes on screen.
        max_move = float(np.max(np.abs(goal - start)))
        which = JOINT_NAMES[int(np.argmax(np.abs(goal - start)))]
        self.get_logger().info(
            f'Moving arm to: {[f"{p:.2f}" for p in positions]} | '
            f'max joint move {max_move:.2f} rad ({which}), '
            f'planned {expected_sec:.1f}s')
        if not self._send_and_wait(self.arm_client, arm_goal, expected_sec, 'Arm'):
            return False
        self.current_q[:6] = goal  # only trust the goal once it succeeded
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
        point.time_from_start = seconds_to_duration(duration)
        trajectory.points = [point]
        goal.trajectory = trajectory

        action = 'Opening' if positions[0] < 0.3 else 'Closing'
        self.get_logger().info(f'{action} gripper...')
        return self._send_and_wait(self.gripper_client, goal, duration, 'Gripper')

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

        # Chain seeds: each waypoint is solved from the previous solution
        # so the three pick poses stay on the same IK branch instead of
        # being solved independently from HOME (which let them land on
        # different configurations and sweep the arm through the object).
        pre_grasp_q = self.solve_ik(base_position + [0, 0, PRE_GRASP_HEIGHT_M])
        if pre_grasp_q is None:
            self.get_logger().error('IK failed for pre-grasp — aborting.')
            return False
        grasp_q = self.solve_ik(base_position + [0, 0, GRASP_CLEARANCE_M], seed=pre_grasp_q)
        lift_q = self.solve_ik(base_position + [0, 0, LIFT_HEIGHT_M], seed=grasp_q)
        if grasp_q is None or lift_q is None:
            self.get_logger().error('IK failed for grasp/lift — aborting.')
            return False

        place = PLACE_POSITIONS[self.target_color]

        # Freeze the wrist (joint 6 / wrist_3) at the grasp value for the
        # whole carry. A top-down grasp is symmetric about the vertical
        # axis, so the gripper's yaw is irrelevant for placing — but the
        # hard-coded place poses use wrist_3 = -0.30 while IK gives each
        # grasp a very different wrist_3 (e.g. blue -2.24). Moving between
        # them spun the wrist ~2 rad mid-carry and flung the block out.
        # Overriding only wrist_3 leaves the place *position* unchanged
        # (that joint just rotates the last link about the approach axis).
        grasp_wrist3 = float(grasp_q[5])
        place_pre = list(place['place_pre'])
        place_down = list(place['place_down'])
        place_pre[5] = grasp_wrist3
        place_down[5] = grasp_wrist3

        self.get_logger().info('=' * 50)
        self.get_logger().info(f'VISION PICK AND PLACE: {self.target_color}')
        self.get_logger().info('=' * 50)

        # Each motion is a checkpoint: bail out the moment one fails (a
        # stuck/aborted move or rejected goal) instead of pushing on and
        # piling more motion onto a robot that's already off-target.
        steps = [
            ('--- HOME ---',         lambda: self.move_arm(HOME)),
            (None,                   lambda: self.move_gripper(GRIPPER_OPEN)),
            ('--- PICK ---',         lambda: self.move_arm(pre_grasp_q.tolist())),
            (None,                   lambda: self.move_arm(grasp_q.tolist())),
            (None,                   lambda: self.move_gripper(GRIPPER_CLOSE)),
            (None,                   lambda: self.move_arm(lift_q.tolist())),
            ('--- PLACE ---',        lambda: self.move_arm(place_pre)),
            (None,                   lambda: self.move_arm(place_down)),
            (None,                   lambda: self.move_gripper(GRIPPER_OPEN)),
            (None,                   lambda: self.move_arm(place_pre)),
            ('--- RETURN HOME ---',  lambda: self.move_arm(HOME)),
        ]
        for header, action in steps:
            if header:
                self.get_logger().info(header)
            if not action():
                self.get_logger().error('Sequence aborted — a motion failed.')
                return False

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
