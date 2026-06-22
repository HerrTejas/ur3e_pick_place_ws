#!/usr/bin/env python3
"""
Pick and Place Node
Hard-coded positions with clear stages for easy MoveIt transition later
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
from typing import Dict, List, Optional

from ur3e_vision_pick_place.robot_config import GRIPPER_JOINTS, HOME, JOINT_NAMES


class PickAndPlace(Node):
    """Hard-coded joint-space pick-and-place for 3 known objects.

    No vision — object positions were found manually with
    joint_tester.py. See vision_pick_and_place.py for the
    camera-driven version that picks whichever color is requested.
    """

    def __init__(self) -> None:
        super().__init__('pick_and_place')
        
        # Action clients for arm and gripper
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory'
        )
        
        # Joint names
        self.arm_joints = JOINT_NAMES
        self.gripper_joints = GRIPPER_JOINTS

        # ============================================
        # HARD-CODED POSITIONS (found manually)
        # ============================================

        # Home position
        self.HOME = HOME
        
        # Object positions dictionary with individual place locations
        self.objects: Dict[str, Dict[str, List[float]]] = {
            'red_box': {
                'grasp':      [1.255, -0.98, 1.4, -1.8, -1.61, -0.3],
                'pre_grasp':  [1.255, -0.94, 1.2, -1.9, -1.57, -0.3],
                'lift':       [1.255, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_pre':  [0.3, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [0.3, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
            'green_cylinder': {
                'grasp':      [1.6, -1.06, 1.5, -2.068, -1.5, 0],
                'pre_grasp':  [1.55, -1.0, 1.2, -2.068, -1.5, 0],
                'lift':       [1.6, -1.0, 0.8, -2.068, -1.5, 0],
                'place_pre':  [0.5, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [0.5, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
            'blue_box': {
                'grasp':      [0.95, -0.95, 1.4, -1.9, -1.65, -0.55],
                'pre_grasp':  [0.95, -0.95, 1.2, -1.8, -1.58, -0.55],
                'lift':       [0.93, -0.9, 0.8, -2.1, -1.65, -0.7],
                'place_pre':  [1.2, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [1.2, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
        }
        
        # Gripper positions
        self.GRIPPER_OPEN = [0.0]
        self.GRIPPER_CLOSE = [0.7]
        
        # Movement duration (seconds)
        self.move_duration = 2.0
        self.gripper_duration = 0.5  # Reduced for faster operation
        
        self.get_logger().info('Pick and Place node initialized!')
        self.get_logger().info('Waiting for action servers...')
        
        # Wait for servers
        self.arm_client.wait_for_server()
        self.get_logger().info('Arm controller ready!')
        
        self.gripper_client.wait_for_server()
        self.get_logger().info('Gripper controller ready!')
        
        self.get_logger().info('Ready to execute pick and place!')
        self.get_logger().info(f'Available objects: {list(self.objects.keys())}')

    def move_arm(self, positions: List[float], duration: Optional[float] = None) -> bool:
        """Move the arm to a joint-space goal and wait for completion.

        Args:
            positions: (6,) target joint positions, radians.
            duration: Time to reach the goal, seconds. Defaults to
                ``self.move_duration``.

        Returns:
            True if the action server accepted and completed the goal.
        """
        if duration is None:
            duration = self.move_duration
            
        goal = FollowJointTrajectory.Goal()
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
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
        
        self.get_logger().info('Arm movement complete!')
        return True

    def move_gripper(self, positions: List[float], duration: Optional[float] = None) -> bool:
        """Move the gripper to a goal and wait for completion.

        Args:
            positions: Target gripper joint position(s).
            duration: Time to reach the goal, seconds. Defaults to
                ``self.gripper_duration``.

        Returns:
            True if the action server accepted and completed the goal.
        """
        if duration is None:
            duration = self.gripper_duration
            
        goal = FollowJointTrajectory.Goal()
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        trajectory.points = [point]
        goal.trajectory = trajectory
        
        action = "Opening" if positions[0] < 0.3 else "Closing"
        self.get_logger().info(f'{action} gripper...')
        
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected!')
            return False
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        self.get_logger().info('Gripper movement complete!')
        return True

    def pick_object(self, object_name: str) -> bool:
        """Run the pre-grasp -> approach -> grasp -> lift sequence.

        Args:
            object_name: Key into ``self.objects``.

        Returns:
            True if the object was a known name and the sequence ran.
        """
        if object_name not in self.objects:
            self.get_logger().error(f'Unknown object: {object_name}')
            return False
        
        obj = self.objects[object_name]
        
        self.get_logger().info(f'\n>>> PICKING: {object_name} <<<')
        
        # PRE-GRASP
        self.get_logger().info('Moving to PRE-GRASP...')
        self.move_arm(obj['pre_grasp'])
        
        # APPROACH
        self.get_logger().info('Approaching object...')
        self.move_arm(obj['grasp'])
        
        # GRASP
        self.get_logger().info('Grasping...')
        self.move_gripper(self.GRIPPER_CLOSE)
        
        # LIFT
        self.get_logger().info('Lifting...')
        self.move_arm(obj['lift'])
        
        return True

    def place_object(self, object_name: str) -> bool:
        """Run the move-to-place -> lower -> release -> retreat sequence.

        Args:
            object_name: Key into ``self.objects``.

        Returns:
            True once the sequence completes.
        """
        obj = self.objects[object_name]
        
        self.get_logger().info(f'\n>>> PLACING: {object_name} <<<')
        
        # MOVE TO PLACE
        self.get_logger().info('Moving to place position...')
        self.move_arm(obj['place_pre'])
        
        # LOWER
        self.get_logger().info('Lowering...')
        self.move_arm(obj['place_down'])
        
        # RELEASE
        self.get_logger().info('Releasing...')
        self.move_gripper(self.GRIPPER_OPEN)
        
        # RETREAT
        self.get_logger().info('Retreating...')
        self.move_arm(obj['place_pre'])
        
        return True

    def execute_pick_place(self, object_name: str = 'red_box') -> bool:
        """Run the full home -> pick -> place -> home cycle for one object.

        Args:
            object_name: Key into ``self.objects``.

        Returns:
            True if every stage completed.
        """
        
        self.get_logger().info('='*50)
        self.get_logger().info(f'PICK AND PLACE: {object_name}')
        self.get_logger().info('='*50)
        
        # Stage 1: HOME
        self.get_logger().info('\n--- STAGE 1: HOME ---')
        self.move_arm(self.HOME)
        self.move_gripper(self.GRIPPER_OPEN)
        
        # Stage 2-5: PICK
        self.get_logger().info('\n--- STAGE 2-5: PICK ---')
        if not self.pick_object(object_name):
            return False
        
        # Stage 6-7: PLACE
        self.get_logger().info('\n--- STAGE 6-7: PLACE ---')
        self.place_object(object_name)
        
        # Stage 8: HOME
        self.get_logger().info('\n--- STAGE 8: RETURN HOME ---')
        self.move_arm(self.HOME)
        
        self.get_logger().info('='*50)
        self.get_logger().info('PICK AND PLACE COMPLETE!')
        self.get_logger().info('='*50)
        
        return True

    def execute_all_objects(self) -> None:
        """Run execute_pick_place() for every known object, in turn."""
        for obj_name in self.objects.keys():
            self.execute_pick_place(obj_name)
            time.sleep(0.5)


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    
    try:
        # Pick a specific object (change this to test different objects)
        # Options: 'red_box', 'green_cylinder', 'blue_box'
        # node.execute_pick_place('red_box')
        
        # Or pick all objects:
        node.execute_all_objects()
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()