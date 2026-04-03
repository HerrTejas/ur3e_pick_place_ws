#!/usr/bin/env python3
"""
Forward Kinematics Node

Forward Kinematics (FK) calculates the end-effector pose given joint angles.

Input:  Joint angles [θ1, θ2, θ3, θ4, θ5, θ6] (radians)
Output: End-effector pose (X, Y, Z position + orientation)

How FK Works:
─────────────
Each joint has a transformation matrix that describes how it moves.
We multiply all matrices together to get the final end-effector pose.

    T_total = T_base × T_joint1 × T_joint2 × T_joint3 × T_joint4 × T_joint5 × T_joint6

Where each T_joint depends on the joint angle θ.

We use Pinocchio library which handles all this math for us!

Author: Tejas (Learning Exercise)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
import pinocchio as pin


class ForwardKinematics(Node):
    def __init__(self):
        super().__init__('forward_kinematics')
        
        # ==========================================
        # PINOCCHIO SETUP
        # ==========================================
        
        # Load robot model from URDF
        urdf_path = "/tmp/ur3e.urdf"
        try:
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            self.get_logger().info(f'Loaded robot model: {self.model.name}')
            self.get_logger().info(f'Number of joints: {self.model.nq}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf first!')
            return
        
        # Get end-effector frame ID
        self.ee_frame_name = "tool0"
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        self.get_logger().info(f'End-effector frame: {self.ee_frame_name} (ID: {self.ee_frame_id})')
        
        # ==========================================
        # JOINT NAMES
        # ==========================================
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # ==========================================
        # ROS2 SETUP
        # ==========================================
        
        # Subscriber: Get joint positions
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher: Publish end-effector pose
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/end_effector_pose',
            10
        )
        
        # Store current positions
        self.current_positions = None
        
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('FORWARD KINEMATICS NODE READY')
        self.get_logger().info('='*60)
        self.get_logger().info('Subscribing to: /joint_states')
        self.get_logger().info('Publishing to:  /end_effector_pose')
        self.get_logger().info('')
    
    def joint_state_callback(self, msg):
        """
        Called when joint states are received.
        Computes FK and publishes end-effector pose.
        """
        # Extract arm joint positions in correct order
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        
        # Check if we have all joints
        if len(positions) != len(self.joint_names):
            return
        
        # Create ordered joint array
        q = np.zeros(self.model.nq)
        for i, name in enumerate(self.joint_names):
            q[i] = positions[name]
        
        self.current_positions = q[:6]
        
        # ==========================================
        # FORWARD KINEMATICS CALCULATION
        # ==========================================
        
        # Step 1: Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        
        # Step 2: Update frame placements
        pin.updateFramePlacements(self.model, self.data)
        
        # Step 3: Get end-effector pose
        ee_pose = self.data.oMf[self.ee_frame_id]
        
        # ==========================================
        # EXTRACT POSITION AND ORIENTATION
        # ==========================================
        
        # Position (X, Y, Z)
        position = ee_pose.translation
        
        # Orientation as rotation matrix
        rotation_matrix = ee_pose.rotation
        
        # Convert rotation matrix to quaternion
        quaternion = pin.Quaternion(rotation_matrix)
        
        # Convert rotation matrix to Roll-Pitch-Yaw (for human readability)
        rpy = pin.rpy.matrixToRpy(rotation_matrix)
        
        # ==========================================
        # PUBLISH POSE
        # ==========================================
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"
        
        # Position
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        
        # Orientation (as quaternion)
        pose_msg.pose.orientation.x = quaternion.x
        pose_msg.pose.orientation.y = quaternion.y
        pose_msg.pose.orientation.z = quaternion.z
        pose_msg.pose.orientation.w = quaternion.w
        
        self.pose_pub.publish(pose_msg)
    
    def compute_fk(self, joint_angles):
        """
        Compute FK for given joint angles.
        
        Args:
            joint_angles: List of 6 joint angles in radians
            
        Returns:
            Dictionary with position, rotation matrix, quaternion, and RPY
        """
        # Create joint array
        q = np.zeros(self.model.nq)
        for i in range(6):
            q[i] = joint_angles[i]
        
        # Compute FK
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get end-effector pose
        ee_pose = self.data.oMf[self.ee_frame_id]
        
        # Extract data
        position = ee_pose.translation
        rotation = ee_pose.rotation
        quaternion = pin.Quaternion(rotation)
        rpy = pin.rpy.matrixToRpy(rotation)
        
        return {
            'position': position,
            'rotation_matrix': rotation,
            'quaternion': [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
            'rpy': rpy  # Roll, Pitch, Yaw in radians
        }
    
    def print_fk(self, joint_angles):
        """
        Compute and print FK result in a nice format.
        """
        result = self.compute_fk(joint_angles)
        
        print('\n' + '='*60)
        print('FORWARD KINEMATICS RESULT')
        print('='*60)
        
        print('\nINPUT - Joint Angles (radians):')
        for i, name in enumerate(self.joint_names):
            print(f'  {name}: {joint_angles[i]:.4f} rad ({np.degrees(joint_angles[i]):.2f}°)')
        
        print('\nOUTPUT - End Effector Pose:')
        print(f'\n  Position (meters):')
        print(f'    X: {result["position"][0]:.4f} m')
        print(f'    Y: {result["position"][1]:.4f} m')
        print(f'    Z: {result["position"][2]:.4f} m')
        
        print(f'\n  Orientation (RPY in degrees):')
        print(f'    Roll:  {np.degrees(result["rpy"][0]):.2f}°')
        print(f'    Pitch: {np.degrees(result["rpy"][1]):.2f}°')
        print(f'    Yaw:   {np.degrees(result["rpy"][2]):.2f}°')
        
        print(f'\n  Orientation (Quaternion):')
        print(f'    x: {result["quaternion"][0]:.4f}')
        print(f'    y: {result["quaternion"][1]:.4f}')
        print(f'    z: {result["quaternion"][2]:.4f}')
        print(f'    w: {result["quaternion"][3]:.4f}')
        
        print('\n' + '='*60)
        
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematics()
    
    # ==========================================
    # TEST FK WITH KNOWN POSITIONS
    # ==========================================
    
    HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]
    
    print('\n' + '#'*60)
    print('FORWARD KINEMATICS DEMONSTRATION')
    print('#'*60)
    
    print('\n>>> Computing FK for HOME position...')
    node.print_fk(HOME)
    
    print('\n>>> Computing FK for RED_BOX_GRASP position...')
    node.print_fk(RED_BOX_GRASP)
    
    # ==========================================
    # INTERACTIVE MODE
    # ==========================================
    
    print('\n' + '#'*60)
    print('INTERACTIVE MODE')
    print('#'*60)
    print('\nYou can now enter custom joint angles.')
    print('Format: j1 j2 j3 j4 j5 j6 (space separated, in radians)')
    print('Example: 0 -1.57 0 -1.57 0 0')
    print('Type "q" to quit.\n')
    
    try:
        while rclpy.ok():
            user_input = input('Enter joint angles (or "q" to quit): ')
            
            if user_input.lower() == 'q':
                break
            
            try:
                angles = [float(x) for x in user_input.split()]
                if len(angles) != 6:
                    print('Error: Please enter exactly 6 values!')
                    continue
                
                node.print_fk(angles)
                
            except ValueError:
                print('Error: Invalid input! Enter numbers separated by spaces.')
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
