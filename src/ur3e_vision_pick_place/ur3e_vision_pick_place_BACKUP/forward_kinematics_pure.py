#!/usr/bin/env python3
"""
Forward Kinematics - Pure Python (DH Parameters)

- joint_state_cb: ONLY stores current joint positions
- Timer: Computes FK using DH parameters and publishes end-effector pose

Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np


class ForwardKinematicsPure(Node):
    def __init__(self):
        super().__init__('forward_kinematics_pure')
        
        # Joint names
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # UR3e DH Parameters: [d, a, alpha]
        self.dh_params = [
            [0.15185,    0,          np.pi/2 ],   # Joint 1
            [0,          -0.24355,   0       ],   # Joint 2
            [0,          -0.2132,    0       ],   # Joint 3
            [0.13105,    0,          np.pi/2 ],   # Joint 4
            [0.08535,    0,          -np.pi/2],   # Joint 5
            [0.0921,     0,          0       ],   # Joint 6
        ]
        
        # Base frame correction (180° rotation around Z)
        self.T_base_correction = np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])
        
        # Current joint positions
        self.current_positions = None
        
        # Subscriber: Joint states (only stores data)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        
        # Publisher: End-effector pose
        self.pose_pub = self.create_publisher(PoseStamped, '/end_effector_pose_pure', 10)
        
        # Timer: Publish FK at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_cb)
        
        self.get_logger().info('Forward Kinematics Pure Node Ready!')
        self.get_logger().info('Publishing to /end_effector_pose_pure at 10 Hz')
    
    def joint_state_cb(self, msg):
        """ONLY store current joint positions."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        
        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]
    
    def dh_matrix(self, theta, d, a, alpha):
        """Compute DH transformation matrix for one joint."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,     st*sa,      a*ct],
            [st,     ct*ca,    -ct*sa,      a*st],
            [0,      sa,        ca,         d   ],
            [0,      0,         0,          1   ]
        ])
    
    def rotation_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return x, y, z, w
    
    def timer_cb(self):
        """Compute FK using DH parameters and publish."""
        if self.current_positions is None:
            return
        
        # Compute FK: T_total = T1 * T2 * T3 * T4 * T5 * T6
        T_total = np.eye(4)
        
        for i in range(6):
            theta = self.current_positions[i]
            d = self.dh_params[i][0]
            a = self.dh_params[i][1]
            alpha = self.dh_params[i][2]
            
            T_i = self.dh_matrix(theta, d, a, alpha)
            T_total = T_total @ T_i
        
        # Apply base frame correction
        T_total = self.T_base_correction @ T_total
        
        # Extract position and orientation
        position = T_total[:3, 3]
        rotation = T_total[:3, :3]
        qx, qy, qz, qw = self.rotation_to_quaternion(rotation)
        
        # Publish
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"
        
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsPure()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
