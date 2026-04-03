#!/usr/bin/env python3
"""
Color Tuner - Shows HSV values at mouse position
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorTuner(Node):
    def __init__(self):
        super().__init__('color_tuner')
        
        self.bridge = CvBridge()
        self.current_image = None
        self.hsv_image = None
        
        self.image_sub = self.create_subscription(
            Image,
            '/gripper_camera/image',
            self.image_callback,
            10
        )
        
        # Create window
        cv2.namedWindow('Color Tuner')
        cv2.setMouseCallback('Color Tuner', self.mouse_callback)
        
        # Timer to update display
        self.timer = self.create_timer(0.1, self.display_image)
        
        self.get_logger().info('Color Tuner started!')
        self.get_logger().info('Hover mouse over colors to see HSV values')

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def mouse_callback(self, event, x, y, flags, param):
        if self.hsv_image is not None and event == cv2.EVENT_MOUSEMOVE:
            # Get HSV value at mouse position
            h, s, v = self.hsv_image[y, x]
            self.get_logger().info(f'Position ({x}, {y}) - H: {h}, S: {s}, V: {v}')

    def display_image(self):
        if self.current_image is not None:
            cv2.imshow('Color Tuner', self.current_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ColorTuner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
