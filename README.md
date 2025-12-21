# UR3e Vision-Based Pick and Place

A ROS2 project demonstrating vision-based object detection and 3D localization using a UR3e robot arm with an attached gripper camera.

![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## Overview

This project implements a complete vision pipeline for robotic manipulation:
1. **Camera Integration** - RGB camera mounted on gripper
2. **Color Detection** - Detects red, green, and blue objects using HSV color space
3. **Shape Detection** - Distinguishes circles from rectangles using circularity
4. **3D Localization** - Converts 2D pixel coordinates to 3D world coordinates using pinhole camera model

## Features

- ✅ Custom Gazebo world with table and colored objects
- ✅ Gripper-mounted RGB camera with ROS2 integration
- ✅ Real-time color detection (red, green, blue)
- ✅ Shape classification (circle vs rectangle)
- ✅ Shadow removal using morphological operations
- ✅ Pinhole camera backprojection for 3D localization
- ✅ TF transform integration for world coordinates

## System Architecture
```
Camera Image → Color Detection → Backprojection → TF Transform → 3D World Position
     ↓              ↓                  ↓               ↓              ↓
  640x480      HSV Masking      Pinhole Model    Camera→World    (X, Y, Z)
   RGB         Contours         X=(u-cx)*Z/fx    Transform       in meters
```

## Prerequisites

- Ubuntu 24.04
- ROS2 Jazzy
- Gazebo Harmonic
- Python 3.12
- OpenCV

## Installation
```bash
# Create workspace
mkdir -p ~/ur3e_pick_place_ws/src
cd ~/ur3e_pick_place_ws/src

# Clone this repository
git clone https://github.com/HerrTejas/ur3e_pick_place_ws.git ur3e_vision_pick_place

# Install dependencies
cd ~/ur3e_pick_place_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --symlink-install

# Source
source install/setup.bash
```

## Usage

### Launch Simulation
```bash
ros2 launch ur3e_vision_pick_place ur3e_pick_place.launch.py
```

### Run Object Detector (2D)
```bash
ros2 run ur3e_vision_pick_place color_detector_v2
```

### Run 3D Object Detector
```bash
ros2 run ur3e_vision_pick_place object_detector_3d
```

### View Detection Results
```bash
ros2 run rqt_image_view rqt_image_view
# Select topic: /object_detector_3d/debug_image
```

### Color Tuning Tool
```bash
ros2 run ur3e_vision_pick_place color_tuner
# Hover mouse over objects to see HSV values
```

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/gripper_camera/image` | sensor_msgs/Image | Raw camera image |
| `/gripper_camera/camera_info` | sensor_msgs/CameraInfo | Camera intrinsics |
| `/object_detector_3d/debug_image` | sensor_msgs/Image | Image with detections |
| `/detected_objects_3d` | std_msgs/String | 3D positions of objects |

## Camera Specifications

| Property | Value |
|----------|-------|
| Resolution | 640 x 480 |
| Field of View | 80° |
| Frame Rate | 20 FPS |
| Focal Length | 381.36 px |

## Project Structure
```
ur3e_vision_pick_place/
├── launch/
│   └── ur3e_pick_place.launch.py
├── worlds/
│   └── pick_place_world.sdf
├── config/
│   └── gz_bridge.yaml
├── urdf/
│   ├── gripper_camera.xacro
│   └── gripper_camera.gazebo.xacro
├── ur3e_vision_pick_place/
│   ├── color_detector.py
│   ├── color_detector_v2.py
│   ├── color_tuner.py
│   └── object_detector_3d.py
├── package.xml
├── setup.py
└── README.md
```

## Technical Details

### Pinhole Camera Model

The 3D position is calculated using backprojection:
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth (known from table height)
```

Where:
- `(u, v)` = pixel coordinates
- `(cx, cy)` = principal point (320, 240)
- `(fx, fy)` = focal length (381.36)
- `Z` = depth to object

### Color Detection

Uses HSV color space for robust detection:
- **Red**: H=0-10, 160-180
- **Green**: H=55-65
- **Blue**: H=100-130

Shadow removal via morphological operations (erosion + dilation).

## Sample Output
```
[INFO] red: pixel=(320, 171) -> world=(0.000, 0.331, 0.430)
[INFO] green: pixel=(229, 168) -> world=(-0.063, 0.333, 0.430)
[INFO] blue: pixel=(409, 171) -> world=(0.062, 0.331, 0.430)
```

## Future Work

- [ ] MoveIt integration for motion planning
- [ ] Pick and place execution
- [ ] YOLO-based object detection
- [ ] Depth camera integration

## Author

**Tejas Murkute**

## License

MIT License
