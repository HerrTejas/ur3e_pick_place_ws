#!/usr/bin/env python3
"""
Object Detector 3D - Detects colored objects and calculates 3D world coordinates
Uses pinhole camera model with backprojection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class ObjectDetector3D(Node):
    def __init__(self):
        super().__init__('object_detector_3d')
        
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from camera_info)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = None
        
        # Known heights
        self.table_height = 0.41  # Table surface Z in world frame
        self.object_height = 0.04  # Approximate object height (4cm)
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribe to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/gripper_camera/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image,
            '/gripper_camera/image',
            self.image_callback,
            10
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
        
        # Morphological kernel
        self.kernel = np.ones((5, 5), np.uint8)
        
        self.get_logger().info('Object Detector 3D started!')
        self.get_logger().info('Waiting for camera info...')

    def camera_info_callback(self, msg):
        """Extract camera intrinsics from camera info"""
        if self.fx is None:
            # K matrix is row-major: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_frame = msg.header.frame_id
            
            self.get_logger().info(f'Camera intrinsics received:')
            self.get_logger().info(f'  fx={self.fx:.2f}, fy={self.fy:.2f}')
            self.get_logger().info(f'  cx={self.cx:.2f}, cy={self.cy:.2f}')
            self.get_logger().info(f'  frame={self.camera_frame}')

    def pixel_to_3d_camera_frame(self, u, v, depth):
        """
        Backproject pixel (u,v) to 3D point in camera frame
        
        Pinhole camera model (inverse):
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def transform_to_world(self, x, y, z):
        """Transform point from camera frame to world frame using TF"""
        try:
            # Create point in camera frame
            point = PointStamped()
            point.header.frame_id = self.camera_frame
            point.header.stamp = self.get_clock().now().to_msg()
            point.point.x = x
            point.point.y = y
            point.point.z = z
            
            # Get transform from camera to world
            transform = self.tf_buffer.lookup_transform(
                'world',
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Transform point
            point_world = do_transform_point(point, transform)
            
            return point_world.point.x, point_world.point.y, point_world.point.z
            
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None, None, None

    def calculate_depth_from_table(self):
        """
        Calculate depth (Z in camera frame) to table surface
        
        Camera is at Z=0.694m in world frame
        Table is at Z=0.41m in world frame
        Since camera looks down, depth ≈ camera_z - table_z
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            camera_z = transform.transform.translation.z
            # Depth to table (accounting for object height)
            depth = camera_z - self.table_height - (self.object_height / 2)
            return depth
            
        except Exception as e:
            self.get_logger().warn(f'Could not calculate depth: {e}')
            return 0.25  # Default fallback

    def detect_color(self, hsv_image, color_name):
        """Create mask for a specific color with morphological cleanup"""
        color_info = self.colors[color_name]
        
        if color_name == 'red':
            mask1 = cv2.inRange(hsv_image, color_info['lower1'], color_info['upper1'])
            mask2 = cv2.inRange(hsv_image, color_info['lower2'], color_info['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, color_info['lower'], color_info['upper'])
        
        # Cleanup
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
        return circularity > 0.8

    def image_callback(self, msg):
        # Wait for camera intrinsics
        if self.fx is None:
            return
        
        # Convert image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        debug_image = cv_image.copy()
        
        # Calculate depth to table
        depth = self.calculate_depth_from_table()
        
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
                    
                    # Get center pixel
                    if self.is_circular(contour):
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        center_u, center_v = int(cx), int(cy)
                        cv2.circle(debug_image, (center_u, center_v), int(radius), bgr_color, 2)
                        shape = 'circle'
                    else:
                        center_u = x + w // 2
                        center_v = y + h // 2
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), bgr_color, 2)
                        shape = 'rectangle'
                    
                    # Backproject to 3D camera frame
                    cam_x, cam_y, cam_z = self.pixel_to_3d_camera_frame(center_u, center_v, depth)
                    
                    # Transform to world frame
                    world_x, world_y, world_z = self.transform_to_world(cam_x, cam_y, cam_z)
                    
                    if world_x is not None:
                        # Draw label with 3D coordinates
                        label = f'{color_name.upper()}'
                        coords = f'({world_x:.2f}, {world_y:.2f}, {world_z:.2f})'
                        cv2.putText(debug_image, label, (x, y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
                        cv2.putText(debug_image, coords, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1)
                        
                        # Draw center
                        cv2.circle(debug_image, (center_u, center_v), 5, bgr_color, -1)
                        
                        all_detections.append({
                            'color': color_name,
                            'shape': shape,
                            'pixel': (center_u, center_v),
                            'world_position': {
                                'x': round(world_x, 3),
                                'y': round(world_y, 3),
                                'z': round(world_z, 3)
                            }
                        })
        
        # Log and publish
        if all_detections:
            for det in all_detections:
                pos = det['world_position']
                self.get_logger().info(
                    f"{det['color']}: pixel={det['pixel']} -> "
                    f"world=({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})"
                )
            
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
