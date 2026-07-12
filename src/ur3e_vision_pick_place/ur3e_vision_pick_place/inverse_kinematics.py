#!/usr/bin/env python3
"""
Inverse Kinematics Node using Pinocchio

The IK math lives in helper_functions/kinematics.py. This file is just
ROS wiring that calls it.

Author: Tejas
"""

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from ur3e_vision_pick_place.helper_functions.kinematics import compute_ik, load_pinocchio
from ur3e_vision_pick_place.robot_config import JOINT_NAMES


class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')

        self.joint_names = JOINT_NAMES

        # Load Pinocchio
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
            self.get_logger().info(f'Model nq: {self.model.nq}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf')
            return

        # State
        self.current_q = np.zeros(self.model.nq)
        self.joints_received = False
        self.target_pose = None

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/target_ee_pose', self.pose_cb, 10)

        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, '/ik_solution', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Inverse Kinematics Node Ready!')
        self.get_logger().info('Subscribing to /target_ee_pose')

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            for i, name in enumerate(self.joint_names):
                self.current_q[i] = positions[name]
            self.joints_received = True

    def pose_cb(self, msg):
        self.target_pose = msg.pose

    def timer_cb(self):
        if self.target_pose is None or not self.joints_received:
            return

        # Extract target
        pos = np.array([
            self.target_pose.position.x,
            self.target_pose.position.y,
            self.target_pose.position.z
        ])
        quat = [
            self.target_pose.orientation.x,
            self.target_pose.orientation.y,
            self.target_pose.orientation.z,
            self.target_pose.orientation.w
        ]
        rot = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()

        # Call the standalone function. joint_names selects just the 6
        # arm joints — the URDF also contains the gripper's joints.
        q = compute_ik(self.model, self.data, self.ee_frame_id,
                       pos, rot, self.current_q[:6], joint_names=JOINT_NAMES)

        if q is not None:
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            self.current_q[:6] = q
        else:
            self.get_logger().warn('IK failed to converge for target pose')


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
