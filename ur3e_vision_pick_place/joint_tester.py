#!/usr/bin/env python3
"""
Joint Tester - Manually find joint positions for pick and place
Run this, then adjust values to find correct positions
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


class JointTester(Node):
    def __init__(self):
        super().__init__('joint_tester')
        
        # Current joint positions
        self.current_positions = {}
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Action client for arm
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        
        # Action client for gripper
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory'
        )
        
        # Joint names
        self.arm_joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.gripper_joints = ['rh_r1_joint']
        
        self.get_logger().info('Joint Tester ready!')
        self.get_logger().info('Waiting for action servers...')
        
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        
        self.get_logger().info('Ready! Use move_arm() and move_gripper() methods.')
        
    def joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.current_positions[name] = msg.position[i]
    
    def print_current_positions(self):
        """Print current joint positions"""
        self.get_logger().info('\n=== CURRENT JOINT POSITIONS ===')
        arm_pos = []
        for joint in self.arm_joints:
            pos = self.current_positions.get(joint, 0.0)
            arm_pos.append(pos)
            self.get_logger().info(f'{joint}: {pos:.4f}')
        
        self.get_logger().info(f'\nCopy-paste format:')
        self.get_logger().info(f'[{", ".join([f"{p:.4f}" for p in arm_pos])}]')
        
        gripper_pos = self.current_positions.get('rh_r1_joint', 0.0)
        self.get_logger().info(f'\nGripper (rh_r1_joint): {gripper_pos:.4f}')
        
    def move_arm(self, positions, duration=3.0):
        """Move arm to position"""
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
            self.get_logger().error('Goal rejected!')
            return False
            
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        self.get_logger().info('Movement complete!')
        self.print_current_positions()
        return True
    
    def move_gripper(self, position, duration=1.0):
        """Move gripper (0.0 = open, 0.7 = closed)"""
        goal = FollowJointTrajectory.Goal()
        
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints
        
        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        trajectory.points = [point]
        goal.trajectory = trajectory
        
        self.get_logger().info(f'Moving gripper to: {position}')
        
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


def main(args=None):
    rclpy.init(args=args)
    node = JointTester()
    
    print("\n" + "="*60)
    print("JOINT TESTER - Find positions for pick and place")
    print("="*60)
    print("\nCommands:")
    print("  node.print_current_positions()  - Show current positions")
    print("  node.move_arm([j1,j2,j3,j4,j5,j6])  - Move arm")
    print("  node.move_gripper(0.0)  - Open gripper")
    print("  node.move_gripper(0.7)  - Close gripper")
    print("\nStarting positions to try:")
    print("  HOME:      [0, -1.57, 0, -1.57, 0, 0]")
    print("  LOOK_DOWN: [0, -0.5, 0.5, -1.57, -1.57, 0]")
    print("="*60 + "\n")
    
    # Keep node alive for interactive use
    try:
        # Print initial positions
        rclpy.spin_once(node, timeout_sec=1.0)
        node.print_current_positions()
        
        # Interactive loop
        while rclpy.ok():
            try:
                cmd = input("\nEnter command (or 'q' to quit): ")
                if cmd.lower() == 'q':
                    break
                elif cmd.strip():
                    exec(cmd)
            except Exception as e:
                print(f"Error: {e}")
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
