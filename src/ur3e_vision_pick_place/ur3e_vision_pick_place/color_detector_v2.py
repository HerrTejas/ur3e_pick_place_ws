#!/usr/bin/env python3
"""
Color Detector V2

2D-only color + shape detector with morphological shadow removal.
Debug/visualization tool — does not use depth, so it does not feed
the live pick pipeline (see object_detector.py for the 3D detector
that does).

The detection math (HSV masks, shape classification) lives in
helper_functions/color_detection.py; this file is just ROS wiring.
"""

from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

from ur3e_vision_pick_place.helper_functions.color_detection import (
    COLOR_RANGES_HSV, DRAW_COLORS_BGR, build_color_mask, classify_shape,
)


class ColorDetectorV2(Node):
    """2D color/shape detector with shadow-robust HSV masking."""

    def __init__(self) -> None:
        super().__init__('color_detector_v2')

        self.bridge = CvBridge()

        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/overhead_camera/image',
            self.image_callback,
            10
        )

        # Publish debug image
        self.debug_pub = self.create_publisher(Image, '/color_detector/debug_image', 10)

        # Publish detected objects info
        self.detection_pub = self.create_publisher(String, '/detected_objects', 10)

        self.get_logger().info('Color Detector V2 started!')

    def image_callback(self, msg: Image) -> None:
        """Detect colored shapes, draw a debug overlay, publish results.

        Args:
            msg: RGB image, bgr8 encoding.
        """
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
        all_detections: List[Dict[str, Any]] = []

        # Detect each color. denoise=True applies the erosion+dilation
        # (morphological opening) that drops thin shadow connections.
        for color_name, ranges in COLOR_RANGES_HSV.items():
            mask = build_color_mask(hsv_image, ranges, denoise=True)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bgr_color = DRAW_COLORS_BGR[color_name]

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    shape = classify_shape(contour)

                    if shape == 'circle':
                        # Use minimum enclosing circle
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        center_x, center_y = int(cx), int(cy)
                        cv2.circle(debug_image, (center_x, center_y), int(radius), bgr_color, 2)
                    else:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), bgr_color, 2)

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
