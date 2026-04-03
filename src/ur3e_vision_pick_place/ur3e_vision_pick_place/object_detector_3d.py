#!/usr/bin/env python3
"""
Object Detector 3D - Detects colored objects and publishes 3D world coordinates
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class ObjectDetector3D(Node):
    def __init__(self):
        super().__init__('object_detector_3d')
        
        self.bridge = CvBridge()
        
        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = None
        
        # Known heights
        self.table_height = 0.41
        self.object_height = 0.04
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/gripper_camera/camera_info',
            self.camera_info_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/gripper_camera/image',
            self.image_callback, 10
        )
        
        # Publishers
        self.debug_pub = self.create_publisher(Image, '/object_detector_3d/debug_image', 10)
        self.detection_pub = self.create_publisher(String, '/detected_objects_3d', 10)
        
        # Color ranges in HSV
        self.colors = {
            'red': {
                'lower1': np.array([0, 100, 100]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([160, 100, 100]),
                'upper2': np.array([180, 255, 255]),
                'bgr': (0, 0, 255)
            },
            'green': {
                'lower': np.array([40, 100, 100]),
                'upper': np.array([80, 255, 255]),
                'bgr': (0, 255, 0)
            },
            'blue': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255]),
                'bgr': (255, 0, 0)
            }
        }
        
        self.kernel = np.ones((5, 5), np.uint8)
        
        self.get_logger().info('Object Detector 3D started!')

    def camera_info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(f'Camera: fx={self.fx:.2f}, fy={self.fy:.2f}, frame={self.camera_frame}')

    def pixel_to_world(self, u, v):
        """Convert pixel coordinates to world frame 3D position"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            camera_z = transform.transform.translation.z
            depth = camera_z - self.table_height - (self.object_height / 2)
            
            if depth <= 0:
                return None
            
            # Backproject to camera frame
            cam_x = (u - self.cx) * depth / self.fx
            cam_y = (v - self.cy) * depth / self.fy
            cam_z = depth
            
            # Transform to world frame
            point = PointStamped()
            point.header.frame_id = self.camera_frame
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = cam_x
            point.point.y = cam_y
            point.point.z = cam_z
            
            world_point = do_transform_point(point, transform)
            return [world_point.point.x, world_point.point.y, world_point.point.z]
            
        except Exception as e:
            return None

    def image_callback(self, msg):
        if self.fx is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            return
        
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        debug_image = cv_image.copy()
        
        detections = {}
        
        for color_name, color_info in self.colors.items():
            # Create mask
            if 'lower1' in color_info:
                mask1 = cv2.inRange(hsv, color_info['lower1'], color_info['upper1'])
                mask2 = cv2.inRange(hsv, color_info['lower2'], color_info['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            
            # Cleanup
            mask = cv2.erode(mask, self.kernel, iterations=1)
            mask = cv2.dilate(mask, self.kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    center_u = x + w // 2
                    center_v = y + h // 2
                    
                    world_pos = self.pixel_to_world(center_u, center_v)
                    
                    if world_pos is not None:
                        detections[color_name] = {
                            'x': round(world_pos[0], 4),
                            'y': round(world_pos[1], 4),
                            'z': round(world_pos[2], 4),
                            'pixel_u': center_u,
                            'pixel_v': center_v
                        }
                        
                        # Draw on debug image
                        bgr = color_info['bgr']
                        cv2.rectangle(debug_image, (x, y), (x+w, y+h), bgr, 2)
                        cv2.circle(debug_image, (center_u, center_v), 5, bgr, -1)
                        label = f'{color_name}: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})'
                        cv2.putText(debug_image, label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)
        
        # Publish detections as JSON
        if detections:
            msg_out = String()
            msg_out.data = json.dumps(detections)
            self.detection_pub.publish(msg_out)
        
        # Publish debug image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, 'bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector3D()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
