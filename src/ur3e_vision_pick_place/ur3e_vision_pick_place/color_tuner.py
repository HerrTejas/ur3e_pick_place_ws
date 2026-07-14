#!/usr/bin/env python3
"""
Color Tuner

Interactive dev tool: shows the HSV value under the mouse cursor over
the live camera feed, for picking color_ranges thresholds used by
object_detector.py / color_detector_v2.py.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorTuner(Node):
    """Interactive HSV value probe for camera-feed color tuning."""

    def __init__(self) -> None:
        super().__init__('color_tuner')
        
        self.bridge = CvBridge()
        self.current_image = None
        self.hsv_image = None
        
        self.image_sub = self.create_subscription(
            Image,
            '/overhead_camera/image',
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

    def image_callback(self, msg: Image) -> None:
        """Cache the latest frame as both BGR and HSV.

        Args:
            msg: RGB image, bgr8 encoding.
        """
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.hsv_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Log the HSV value under the cursor on mouse move.

        Args:
            event: OpenCV mouse event type.
            x: Cursor column in the displayed image.
            y: Cursor row in the displayed image.
            flags: OpenCV mouse event flags (unused).
            param: OpenCV callback userdata (unused).
        """
        if self.hsv_image is not None and event == cv2.EVENT_MOUSEMOVE:
            # Get HSV value at mouse position
            h, s, v = self.hsv_image[y, x]
            self.get_logger().info(f'Position ({x}, {y}) - H: {h}, S: {s}, V: {v}')

    def display_image(self) -> None:
        """Render the latest cached frame to the OpenCV window."""
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
