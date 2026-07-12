#!/usr/bin/env python3
"""Unit tests for helper_functions/color_detection.py (needs cv2, no ROS).

    cd src/ur3e_vision_pick_place && python -m pytest test/test_color_detection.py
"""

import cv2
import numpy as np
import pytest

from ur3e_vision_pick_place.helper_functions.color_detection import (
    COLOR_RANGES_HSV, build_color_mask, classify_shape,
    find_largest_blob, pixel_to_camera,
)


def _bgr_image_with_rect(bgr_color, top_left, size):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    x, y = top_left
    w, h = size
    image[y:y + h, x:x + w] = bgr_color
    return image


@pytest.mark.parametrize('color,bgr', [
    ('red', (0, 0, 255)),
    ('green', (0, 255, 0)),
    ('blue', (255, 0, 0)),
])
def test_detects_each_color_centroid(color, bgr):
    image = _bgr_image_with_rect(bgr, top_left=(300, 200), size=(60, 40))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = build_color_mask(hsv, COLOR_RANGES_HSV[color])
    blob = find_largest_blob(mask, min_area=100.0)

    assert blob is not None
    u, v, _ = blob
    assert u == pytest.approx(300 + 30, abs=2)
    assert v == pytest.approx(200 + 20, abs=2)


def test_no_detection_on_empty_image():
    hsv = cv2.cvtColor(np.zeros((480, 640, 3), np.uint8), cv2.COLOR_BGR2HSV)
    for ranges in COLOR_RANGES_HSV.values():
        mask = build_color_mask(hsv, ranges)
        assert find_largest_blob(mask, min_area=100.0) is None


def test_min_area_filters_specks():
    image = _bgr_image_with_rect((0, 0, 255), top_left=(10, 10), size=(3, 3))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv, COLOR_RANGES_HSV['red'])
    assert find_largest_blob(mask, min_area=100.0) is None


def test_largest_blob_wins():
    image = _bgr_image_with_rect((0, 0, 255), top_left=(50, 50), size=(20, 20))
    image[300:380, 400:480] = (0, 0, 255)  # bigger red square
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv, COLOR_RANGES_HSV['red'])
    u, v, _ = find_largest_blob(mask, min_area=100.0)
    assert u == pytest.approx(440, abs=2)
    assert v == pytest.approx(340, abs=2)


def test_denoise_removes_speckle():
    image = _bgr_image_with_rect((0, 0, 255), top_left=(300, 200), size=(60, 40))
    # Sprinkle 1-pixel red noise.
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 480, 50)
    xs = rng.integers(0, 640, 50)
    image[ys, xs] = (0, 0, 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = build_color_mask(hsv, COLOR_RANGES_HSV['red'], denoise=True)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) > 50]
    assert len(big) == 1


def test_pixel_to_camera_roundtrip():
    fx = fy = 381.36
    cx, cy = 320.0, 240.0
    x, y, z = pixel_to_camera(400, 300, 0.5, fx, fy, cx, cy)
    assert z == pytest.approx(0.5)
    # Reproject back.
    assert 400 == pytest.approx(x * fx / z + cx)
    assert 300 == pytest.approx(y * fy / z + cy)
    # Principal point maps to the optical axis.
    assert pixel_to_camera(cx, cy, 1.0, fx, fy, cx, cy)[:2] == pytest.approx((0.0, 0.0))


def test_classify_shape():
    circle = cv2.findContours(
        cv2.circle(np.zeros((200, 200), np.uint8), (100, 100), 50, 255, -1),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    rect_img = np.zeros((200, 200), np.uint8)
    rect_img[50:90, 40:160] = 255
    rect = cv2.findContours(rect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]

    assert classify_shape(circle) == 'circle'
    assert classify_shape(rect) == 'rectangle'
