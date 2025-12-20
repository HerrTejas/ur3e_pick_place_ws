#!/usr/bin/env python3
"""
Color Detector Node - Step 2: Detect blue color
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/gripper_camera/image',
            self.image_callback,
            10
        )
        
        # Publish debug image (with detection boxes)
        self.debug_pub = self.create_publisher(Image, '/color_detector/debug_image', 10)
        
        # Blue color range in HSV
        # H: 100-130 (blue hue)
        # S: 100-255 (medium to high saturation)
        # V: 100-255 (medium to high brightness)
        self.blue_lower = np.array([100, 100, 100])
        self.blue_upper = np.array([130, 255, 255])
        
        self.get_logger().info('Color Detector started!')

    def image_callback(self, msg):
        # Step 1: Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Step 2: Convert BGR to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Step 3: Create mask for blue color
        blue_mask = cv2.inRange(hsv_image, self.blue_lower, self.blue_upper)
        
        # Step 4: Find contours (shapes) in the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 5: Process each contour
        debug_image = cv_image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Ignore small detections (noise)
            if area > 500:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Draw rectangle around object
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw label
                cv2.putText(debug_image, 'BLUE', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw center point
                cv2.circle(debug_image, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # Log detection
                self.get_logger().info(f'Blue object at ({center_x}, {center_y}), area: {area}')
        
        # Step 6: Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
