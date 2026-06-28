#!/usr/bin/env python3
"""Forward Kinematics Node.

Subscribes:  /joint_states    (sensor_msgs/JointState)
Publishes:   /end_effector_pose (geometry_msgs/PoseStamped) at 10 Hz

Pure ROS wiring. All math is in helper_functions.kinematics.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from ur3e_vision_pick_place.helper_functions.kinematics import (
    load_pinocchio,
    compute_fk,
)


class ForwardKinematicsNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    def __init__(self):
        super().__init__('forward_kinematics')

        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf')
            raise

        self.current_q = None

        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/end_effector_pose', 10)

        self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Forward Kinematics Ready (10 Hz)')

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_q = np.array(
                [positions[n] for n in self.JOINT_NAMES])

    def timer_cb(self):
        if self.current_q is None:
            return

        position, quaternion = compute_fk(
            self.model, self.data, self.ee_frame_id, self.current_q)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
