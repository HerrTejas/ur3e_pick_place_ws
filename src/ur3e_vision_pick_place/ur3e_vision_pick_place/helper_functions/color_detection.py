#!/usr/bin/env python3
"""HSV color detection and pinhole backprojection — pure OpenCV/numpy, no ROS.

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.color_detection import (
        COLOR_RANGES_HSV, DRAW_COLORS_BGR, build_color_mask,
        find_largest_blob, pixel_to_camera, classify_shape,
    )

Author: Tejas
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

#: HSV (lower, upper) ranges per color. Red needs two ranges because its
#: hue wraps around 0/180 in OpenCV's H channel.
COLOR_RANGES_HSV: Dict[str, List[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]] = {
    'red': [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    'green': [
        (np.array([35, 100, 70]), np.array([85, 255, 255])),
    ],
    'blue': [
        (np.array([100, 120, 70]), np.array([130, 255, 255])),
    ],
}

#: BGR colors used to draw each detection on debug images.
DRAW_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
}


def build_color_mask(
    hsv_image: npt.NDArray[np.uint8],
    ranges: List[Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    denoise: bool = False,
    kernel_size: int = 5,
) -> npt.NDArray[np.uint8]:
    """Build a binary mask of pixels inside any of the given HSV ranges.

    Args:
        hsv_image: (H, W, 3) image already converted to HSV.
        ranges: List of (lower, upper) HSV bounds; results are OR-ed.
        denoise: If True, apply erosion + dilation to drop speckle and
            shadow noise (morphological opening).
        kernel_size: Size of the square structuring element for denoise.

    Returns:
        (H, W) uint8 mask, 255 where the color matched.
    """
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = mask | cv2.inRange(hsv_image, lower, upper)

    if denoise:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def find_largest_blob(
    mask: npt.NDArray[np.uint8], min_area: float = 100.0,
) -> Optional[Tuple[int, int, npt.NDArray]]:
    """Find the centroid of the largest connected region in a mask.

    Args:
        mask: (H, W) binary mask (e.g. from :func:`build_color_mask`).
        min_area: Minimum contour area in pixels; smaller blobs are noise.

    Returns:
        ``(u, v, contour)`` — centroid pixel coordinates and the contour
        itself (for drawing), or None if nothing big enough was found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    m = cv2.moments(largest)
    if m['m00'] == 0:
        return None
    u = int(m['m10'] / m['m00'])
    v = int(m['m01'] / m['m00'])
    return u, v, largest


def pixel_to_camera(
    u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float,
) -> Tuple[float, float, float]:
    """Backproject a pixel with known depth to camera-frame 3D (pinhole model).

    Args:
        u, v: Pixel coordinates.
        depth: Depth along the optical axis, metres.
        fx, fy: Focal lengths, pixels.
        cx, cy: Principal point, pixels.

    Returns:
        ``(X, Y, Z)`` in the camera optical frame, metres.
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return x, y, depth


def classify_shape(contour: npt.NDArray, circularity_threshold: float = 0.8) -> str:
    """Classify a contour as 'circle' or 'rectangle' by circularity.

    Circularity = 4*pi*area / perimeter^2 — 1.0 for a perfect circle,
    ~0.785 for a square, lower for elongated shapes.

    Args:
        contour: OpenCV contour (from findContours).
        circularity_threshold: Above this the contour counts as a circle.

    Returns:
        'circle' or 'rectangle' ('unknown' for degenerate contours).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0 or area <= 0:
        return 'unknown'
    circularity = 4.0 * np.pi * area / (perimeter ** 2)
    return 'circle' if circularity > circularity_threshold else 'rectangle'
