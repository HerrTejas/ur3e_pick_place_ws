#!/usr/bin/env python3
"""
Object Detector Node

Detects colored objects (red, green, blue) in the RGB image, reads
depth at the detected pixel, and converts to a 3D point in the camera
frame using the pinhole model. Publishes one PointStamped per color on
``/detected_object/{color}`` plus a debug image with detections drawn.

Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

from ur3e_vision_pick_place.helper_functions.color_detection import (
    COLOR_RANGES_HSV, DRAW_COLORS_BGR, build_color_mask,
    find_largest_blob, pixel_to_camera,
)

#: Plausible depth range (metres) for a table object seen by the
#: gripper camera. Readings outside this are rejected: anything below
#: MIN is point-blank/near-clip noise (e.g. the camera parked on top of
#: the object, or the gripper occluding the lens), which would localize
#: the object at the camera and send the arm into a collision.
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 2.0


class ObjectDetector(Node):
    """HSV color + depth based 3D object detector."""

    def __init__(self) -> None:
        super().__init__('object_detector')

        self.bridge = CvBridge()

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Latest depth image
        self.depth_image = None

        # HSV ranges + debug draw colors, shared via helper_functions
        self.color_ranges = COLOR_RANGES_HSV
        self.draw_colors = DRAW_COLORS_BGR

        # Subscribers
        self.create_subscription(
            CameraInfo, '/gripper_camera/camera_info',
            self.camera_info_cb, 10)
        self.create_subscription(
            Image, '/gripper_camera/depth_image',
            self.depth_cb, 10)
        self.create_subscription(
            Image, '/gripper_camera/image',
            self.rgb_cb, 10)

        # One publisher per color
        self.point_pubs = {}
        for color in self.color_ranges:
            self.point_pubs[color] = self.create_publisher(
                PointStamped, f'/detected_object/{color}', 10)

        # Debug image publisher
        self.debug_pub = self.create_publisher(Image, '/detected_objects_debug', 10)

        self.get_logger().info('Object Detector Ready!')
        self.get_logger().info('Detecting: red, green, blue')
        self.get_logger().info('Debug image on: /detected_objects_debug')

    def camera_info_cb(self, msg: CameraInfo) -> None:
        """Extract intrinsics once from the K matrix.

        Args:
            msg: Camera intrinsics, fx/fy/cx/cy read from msg.k.
        """
        if self.fx is not None:
            return

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, '
            f'cx={self.cx:.2f}, cy={self.cy:.2f}')

    def depth_cb(self, msg: Image) -> None:
        """Store the latest depth image.

        Args:
            msg: Depth image, 32FC1 encoding (metres per pixel).
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_cb(self, msg: Image) -> None:
        """Detect objects in RGB, look up depth, publish 3D points.

        For each configured color, finds the largest matching contour,
        looks up its depth, backprojects to a camera-frame 3D point via
        the pinhole model, and publishes it on
        ``/detected_object/{color}``. Also publishes an annotated debug
        image on ``/detected_objects_debug``.

        Args:
            msg: RGB image, bgr8 encoding.
        """
        if self.fx is None or self.depth_image is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        for color, ranges in self.color_ranges.items():
            # HSV mask + largest blob (pure OpenCV, helper_functions)
            mask = build_color_mask(hsv, ranges)
            blob = find_largest_blob(mask, min_area=100.0)
            if blob is None:
                continue
            u, v, largest = blob

            # Depth lookup
            if v >= self.depth_image.shape[0] or u >= self.depth_image.shape[1]:
                continue
            Z = float(self.depth_image[v, u])
            if Z <= 0.0 or np.isnan(Z) or np.isinf(Z):
                continue
            if not (MIN_VALID_DEPTH_M <= Z <= MAX_VALID_DEPTH_M):
                self.get_logger().warn(
                    f'{color}: implausible depth {Z:.3f}m at px ({u},{v}) '
                    f'(outside [{MIN_VALID_DEPTH_M}, {MAX_VALID_DEPTH_M}]m) — '
                    'camera too close / occluded? Skipping this detection.')
                continue

            # Pinhole backprojection (helper_functions)
            X, Y, Z = pixel_to_camera(u, v, Z, self.fx, self.fy, self.cx, self.cy)

            # Draw on debug image
            bgr = self.draw_colors[color]
            cv2.circle(cv_image, (u, v), 10, bgr, 2)
            cv2.drawContours(cv_image, [largest], -1, bgr, 1)
            cv2.putText(cv_image, f'{color}', (u + 15, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            cv2.putText(cv_image, f'({X:.3f}, {Y:.3f}, {Z:.3f})',
                        (u + 15, v + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr, 1)
            cv2.putText(cv_image, f'px:({u}, {v})',
                        (u + 15, v + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr, 1)

            # Publish 3D point
            point_msg = PointStamped()
            point_msg.header.stamp = msg.header.stamp
            point_msg.header.frame_id = 'gripper_camera_optical_link'
            point_msg.point.x = X
            point_msg.point.y = Y
            point_msg.point.z = Z

            self.point_pubs[color].publish(point_msg)
            self.get_logger().info(
                f'{color}: pixel ({u},{v}), depth {Z:.3f}m → '
                f'camera ({X:.3f}, {Y:.3f}, {Z:.3f})')

        # Publish debug image with all detections drawn
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f'Debug image publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()