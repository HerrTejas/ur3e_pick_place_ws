#!/usr/bin/env python3
"""
Color Detector Node - Step 4: Different shapes for different objects
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
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
        
        self.get_logger().info('Color Detector started! Detecting: red, green, blue')

    def detect_color(self, hsv_image, color_name):
        """Create mask for a specific color"""
        color_info = self.colors[color_name]
        
        if color_name == 'red':
            mask1 = cv2.inRange(hsv_image, color_info['lower1'], color_info['upper1'])
            mask2 = cv2.inRange(hsv_image, color_info['lower2'], color_info['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, color_info['lower'], color_info['upper'])
        
        return mask

    def is_circular(self, contour):
        """Check if contour is circular using circularity formula"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return False
        
        # Circularity = 4 * pi * area / perimeter^2
        # Perfect circle = 1.0, square ≈ 0.785
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        #Printing Actual Circularity value to see what we are getting
        self.get_logger().info(f'Circularity: {circularity:.3f}')
        
        # If circularity > 0.8, consider it a circle
        return circularity < 0.7

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
            # Get mask for this color
            mask = self.detect_color(hsv_image, color_name)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get BGR color for drawing
            bgr_color = self.colors[color_name]['bgr']
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                self.get_logger().info(f'{color_name}: area = {area}')
                
                if area > 300:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Check if circular or rectangular
                    if self.is_circular(contour):
                        # Draw circle around circular object
                        radius = max(w, h) // 2
                        cv2.circle(debug_image, (center_x, center_y), radius, bgr_color, 2)
                        shape = 'circle'
                    else:
                        # Draw rectangle around rectangular object
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), bgr_color, 2)
                        shape = 'rectangle'
                    
                    # Draw label above object
                    cv2.putText(debug_image, color_name.upper(), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
                    
                    # Draw center point
                    cv2.circle(debug_image, (center_x, center_y), 5, bgr_color, -1)
                    
                    # Add to detections
                    all_detections.append({
                        'color': color_name,
                        'shape': shape,
                        'center_x': center_x,
                        'center_y': center_y,
                        'area': area
                    })
        
        # Log and publish detections
        if all_detections:
            # Log to terminal
            detected = [f"{d['color']}({d['shape']})" for d in all_detections]
            self.get_logger().info(f'Detected: {detected}')
            
            # Publish detection info
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
