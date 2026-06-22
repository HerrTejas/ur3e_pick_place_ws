#!/usr/bin/env python3
"""
Forward Kinematics - Pure Python (DH Parameters)

The DH math lives in helper_functions/dh_kinematics.py. This file is
just ROS wiring: joint_state_cb stores current joint positions, the
timer computes FK and publishes the end-effector pose.

Author: Tejas
"""

from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from ur3e_vision_pick_place.helper_functions.dh_kinematics import compute_fk_dh, rotation_to_quaternion
from ur3e_vision_pick_place.robot_config import JOINT_NAMES


class ForwardKinematicsPure(Node):
    """FK via hand-rolled DH parameters, as a cross-check on the
    Pinocchio-based forward_kinematics.py."""

    def __init__(self) -> None:
        super().__init__('forward_kinematics_pure')

        # Joint names
        self.joint_names = JOINT_NAMES

        # Current joint positions
        self.current_positions: Optional[List[float]] = None

        # Subscriber: Joint states (only stores data)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        # Publisher: End-effector pose
        self.pose_pub = self.create_publisher(PoseStamped, '/end_effector_pose_pure', 10)

        # Timer: Publish FK at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Forward Kinematics Pure Node Ready!')
        self.get_logger().info('Publishing to /end_effector_pose_pure at 10 Hz')

    def joint_state_cb(self, msg: JointState) -> None:
        """ONLY store current joint positions."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]

        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]

    def timer_cb(self) -> None:
        """Compute FK using DH parameters and publish the EE pose."""
        if self.current_positions is None:
            return

        position, rotation = compute_fk_dh(self.current_positions)
        qx, qy, qz, qw = rotation_to_quaternion(rotation)

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
