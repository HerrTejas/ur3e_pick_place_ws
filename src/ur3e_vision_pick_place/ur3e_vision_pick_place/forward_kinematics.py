#!/usr/bin/env python3
"""
Forward Kinematics Node using Pinocchio

The FK math lives in helper_functions/kinematics.py. This file is just
ROS wiring: joint_state_cb stores current joint positions, the timer
computes FK and publishes the end-effector pose.

Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import pinocchio as pin

from ur3e_vision_pick_place.helper_functions.kinematics import compute_fk, load_pinocchio
from ur3e_vision_pick_place.robot_config import JOINT_NAMES


class ForwardKinematics(Node):
    def __init__(self) -> None:
        super().__init__('forward_kinematics')

        # Joint names
        self.joint_names = JOINT_NAMES

        # Current joint positions
        self.current_positions = None

        # Load Pinocchio model
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf')
            return

        # Subscriber: Joint states (only stores data)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        # Publisher: End-effector pose
        self.pose_pub = self.create_publisher(PoseStamped, '/end_effector_pose', 10)

        # Timer: Publish FK at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Forward Kinematics Node Ready!')
        self.get_logger().info('Publishing to /end_effector_pose at 10 Hz')

    def joint_state_cb(self, msg: JointState) -> None:
        """ONLY store current joint positions."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]

        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]

    def timer_cb(self) -> None:
        """Compute FK and publish end-effector pose."""
        if self.current_positions is None:
            return

        ee_pose = compute_fk(self.model, self.data, self.ee_frame_id, self.current_positions)
        position = ee_pose.translation
        quaternion = pin.Quaternion(ee_pose.rotation)

        # Publish
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        pose_msg.pose.orientation.x = quaternion.x
        pose_msg.pose.orientation.y = quaternion.y
        pose_msg.pose.orientation.z = quaternion.z
        pose_msg.pose.orientation.w = quaternion.w

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
