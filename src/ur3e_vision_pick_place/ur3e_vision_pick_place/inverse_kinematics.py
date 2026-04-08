#!/usr/bin/env python3
"""
Inverse Kinematics Node using Pinocchio

Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import pinocchio as pin


class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')
        
        # Joint names
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Load Pinocchio model
        urdf_path = "/tmp/ur3e.urdf"
        try:
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            self.ee_frame_id = self.model.getFrameId("tool0")
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
            self.get_logger().info(f'Model nq: {self.model.nq}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf')
            return
        
        # IK parameters
        self.max_iterations = 200
        self.tolerance = 1e-4
        self.damping = 1e-6
        
        # State
        self.current_q = np.zeros(self.model.nq)
        self.joints_received = False
        self.target_pose = None
        
        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/end_effector_pose', self.pose_cb, 10)
        
        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, '/ik_solution', 10)
        
        # Timer
        self.timer = self.create_timer(0.1, self.timer_cb)
        
        self.get_logger().info('Inverse Kinematics Node Ready!')
        self.get_logger().info('Subscribing to /end_effector_pose')
    
    def joint_state_cb(self, msg):
        """Store current joint positions as initial guess."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        
        if len(positions) == 6:
            for i, name in enumerate(self.joint_names):
                self.current_q[i] = positions[name]
            self.joints_received = True
    
    def pose_cb(self, msg):
        """Store target end-effector pose."""
        self.target_pose = msg.pose
    
    def compute_ik(self, target_pose):
        """Compute IK using damped least squares."""
        # Extract target position
        target_position = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z
        ])
        
        # Extract target quaternion
        target_quaternion = np.array([
            target_pose.orientation.x,
            target_pose.orientation.y,
            target_pose.orientation.z,
            target_pose.orientation.w
        ])
        
        # Create target SE3
        target_rotation = pin.Quaternion(
            target_quaternion[3],
            target_quaternion[0],
            target_quaternion[1],
            target_quaternion[2]
        ).toRotationMatrix()
        
        target_se3 = pin.SE3(target_rotation, target_position)
        
        # Initial guess (from actual robot position)
        q = self.current_q.copy()
        
        # Iterative IK
        for i in range(self.max_iterations):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Current end-effector pose
            current_se3 = self.data.oMf[self.ee_frame_id]
            
            # Error
            error = pin.log6(current_se3.inverse() * target_se3).vector
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < self.tolerance:
                self.get_logger().info(f'Converged at iteration {i}, error: {error_norm:.6f}')
                self.current_q = q.copy()
                return q[:6].tolist()
            
            # Compute Jacobian
            J = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            
            # Only use first 6 columns
            J_arm = J[:, :6]
            
            # Damped least squares
            JtJ = J_arm.T @ J_arm + self.damping * np.eye(6)
            delta_q = np.linalg.solve(JtJ, J_arm.T @ error)
            
            # Update arm joints
            q[:6] = q[:6] + delta_q
        
        self.get_logger().warn(f'IK did not converge! Final error: {error_norm:.6f}')
        return None
    
    def timer_cb(self):
        """Compute IK and publish joint angles."""
        if self.target_pose is None:
            return
        
        if not self.joints_received:
            self.get_logger().warn('Waiting for joint states...')
            return
        
        q = self.compute_ik(self.target_pose)
        
        if q is not None:
            msg = Float64MultiArray()
            msg.data = q
            self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()