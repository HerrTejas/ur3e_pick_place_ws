#!/usr/bin/env python3
"""
Frame Transformer Node
Transforms detected object from camera frame to base_link,
builds a PoseStamped, and publishes to /path_target_pose.

Author: Tejas Murkute
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
from std_msgs.msg import String
import tf2_geometry_msgs # needed for buffer.transform() to work with geometry_msgs
from rclpy.duration import Duration
import rclpy.time

class FrameTransformer(Node):
    def __init__(self):
        super().__init__('frame_transformer')

        # TF2 buffer and listner
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listner = tf2_ros.TransformListener(self.tf_buffer, self)

        # Fixed grasp orientation
        #
        # Node 1 gives us a 3D point (position only — x, y, z).
        # But path_interpolation needs a full POSE (position + orientation).
        # We need to add an orientation: "which way should the gripper face?"
        #
        # For grasping from above, we use the same downward-facing orientation
        # you already tested with the red box grasp:
        #   orientation: x=0.995, y=-0.006, z=-0.010, w=0.097
        #
        # This quaternion means "gripper pointing almost straight down."
        # Later you could compute this dynamically based on the object,
        # but fixed works for pick-and-place from above.
        # self.grasp_orientation = {
        #     'x': 0.995,
        #     'y': -0.006,
        #     'z': -0.010,
        #     'w': 0.097
        # 
        self.grasp_orientation = {
            'x': 0.999,
            'y': -0.008,
            'z': 0.010,
            'w': -0.033
        }

        # Active color — None means "do nothing, wait for command"
        self.active_color = None

        # Trigger subscriber — send "red", "green", or "blue" to pick
        self.create_subscription(
            String, '/pick_color',
            self.pick_color_cb, 10)

        # Subscribe to all three color topics from Node 1
        self.colors = ['red', 'green', 'blue']
        for color in self.colors:
            self.create_subscription(PointStamped, f'/detected_object/{color}',
                                     lambda msg, c=color: self.point_cb(msg, c), 10)
            
        # Publisher - goes directly to path interpolation node
        self.pose_pub = self.create_publisher(PoseStamped, '/path_target_pose', 10)

        self.get_logger().info('Frame Transformer Ready!')
        self.get_logger().info(' SUbscribing to /detected_object/red, green, blue')
        self.get_logger().info(' Publishing to /path_target_pose')

    def pick_color_cb(self, msg):
        """Activate picking for a specific color."""
        color = msg.data.strip().lower()
        if color not in self.colors:
            self.get_logger().warn(f'Unknown color: {color}. Use red, green, or blue.')
            return
        self.active_color = color
        self.get_logger().info(f'Command received: pick {self.active_color}')

    def point_cb(self, msg, color):
        """Transform point from camera frame to base_link, publish as PoseStamped."""
        if self.active_color is None:
            return
        if color != self.active_color:
            return

        # Immediately block further publishes before doing any work
        picked_color = self.active_color
        self.active_color = None

        try:
            msg.header.stamp = rclpy.time.Time().to_msg()
            point_base = self.tf_buffer.transform(
                msg, 'base_link', timeout=Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'TF failed for {picked_color}: {e}')
            return

        pose_msg = PoseStamped()
        pose_msg.header.stamp = point_base.header.stamp
        pose_msg.header.frame_id = 'base_link'

        pose_msg.pose.position.x = point_base.point.x
        pose_msg.pose.position.y = point_base.point.y
        pose_msg.pose.position.z = point_base.point.z + 0.10

        pose_msg.pose.orientation.x = self.grasp_orientation['x']
        pose_msg.pose.orientation.y = self.grasp_orientation['y']
        pose_msg.pose.orientation.z = self.grasp_orientation['z']
        pose_msg.pose.orientation.w = self.grasp_orientation['w']

        self.pose_pub.publish(pose_msg)
        self.get_logger().info(
            f'Sent {picked_color} target in base_link: '
            f'({pose_msg.pose.position.x:.3f}, '
            f'{pose_msg.pose.position.y:.3f}, '
            f'{pose_msg.pose.position.z:.3f})')
        self.get_logger().info('Waiting for next /pick_color command...')

def main(args=None):
    rclpy.init(args=args)
    node = FrameTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()