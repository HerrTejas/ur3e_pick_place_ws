#!/usr/bin/env python3
"""
Color Detector V2 - With morphological cleanup for shadow removal
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorDetectorV2(Node):
    def __init__(self):
        super().__init__('color_detector_v2')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/gripper_camera/image',
            self.image_callback,
            10
        )
        
        # Publish debug image
        self.debug_pub = self.create_publisher(Image, '/color_detector/debug_image', 10)
        
        # Publish detected objects info
        self.detection_pub = self.create_publisher(String, '/detected_objects', 10)
        
        # Define color ranges in HSV
        self.colors = {
            'red': {
                'lower1': np.array([0, 100, 100]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),
                'upper2': np.array([180, 255, 255]),
                'bgr': (0, 0, 255)
            },
            'green': {
                'lower': np.array([55, 100, 100]),
                'upper': np.array([65, 255, 255]),
                'bgr': (0, 255, 0)
            },
            'blue': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255]),
                'bgr': (255, 0, 0)
            }
        }
        
        # Kernel for morphological operations
        self.kernel = np.ones((5, 5), np.uint8)
        
        self.get_logger().info('Color Detector V2 started!')

    def detect_color(self, hsv_image, color_name):
        """Create mask for a specific color with cleanup"""
        color_info = self.colors[color_name]
        
        if color_name == 'red':
            mask1 = cv2.inRange(hsv_image, color_info['lower1'], color_info['upper1'])
            mask2 = cv2.inRange(hsv_image, color_info['lower2'], color_info['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, color_info['lower'], color_info['upper'])
        
        # Morphological operations to clean up
        # Erosion: removes small noise and thin shadow connections
        # Dilation: restores main object size
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        
        return mask

    def is_circular(self, contour):
        """Check if contour is circular"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        self.get_logger().info(f'Circularity: {circularity:.3f}')
        return circularity > 0.8

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create debug image
        debug_image = cv_image.copy()
        
        # Store all detections
        all_detections = []
        
        # Detect each color
        for color_name in self.colors:
            mask = self.detect_color(hsv_image, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bgr_color = self.colors[color_name]['bgr']
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if self.is_circular(contour):
                        # Use minimum enclosing circle
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        center_x, center_y = int(cx), int(cy)
                        cv2.circle(debug_image, (center_x, center_y), int(radius), bgr_color, 2)
                        shape = 'circle'
                    else:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), bgr_color, 2)
                        shape = 'rectangle'
                    
                    # Draw label and center
                    cv2.putText(debug_image, color_name.upper(), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
                    cv2.circle(debug_image, (center_x, center_y), 5, bgr_color, -1)
                    
                    all_detections.append({
                        'color': color_name,
                        'shape': shape,
                        'center_x': center_x,
                        'center_y': center_y,
                        'area': area
                    })
        
        # Publish detections
        if all_detections:
            detected = [f"{d['color']}({d['shape']})" for d in all_detections]
            self.get_logger().info(f'Detected: {detected}')
            
            detection_msg = String()
            detection_msg.data = str(all_detections)
            self.detection_pub.publish(detection_msg)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectorV2()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
