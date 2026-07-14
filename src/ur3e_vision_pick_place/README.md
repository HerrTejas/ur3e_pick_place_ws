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

### Vision pipeline
```
Camera Image → Color Detection → Backprojection → TF Transform → 3D World Position
     ↓              ↓                  ↓               ↓              ↓
  640x480      HSV Masking      Pinhole Model    Camera→World    (X, Y, Z)
   RGB         Contours         X=(u-cx)*Z/fx    Transform       in meters
```

### Motion pipeline
```
3D Target → IK (seed-regularized DLS) → Trapezoidal Profile → JointTrajectory → Controller
               ↓                            ↓
        helper_functions/           helper_functions/
        kinematics.py               trajectory_profile.py
```

All math (kinematics, trajectory profiles, cartesian interpolation,
color detection) lives in `helper_functions/` as plain Python with **no
ROS imports** — every node file is thin ROS wiring around those
functions, and the math can be unit-tested without a robot or
simulator.

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

### Export the URDF for Pinocchio (required once per shell/session)

The kinematics helpers load the robot model from `/tmp/ur3e.urdf`
(see `robot_config.URDF_PATH`):

```bash
xacro $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro \
    ur_type:=ur3e name:=ur > /tmp/ur3e.urdf
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
ros2 run ur3e_vision_pick_place object_detector
```

### View Detection Results
```bash
# Detector + viewer together — rqt_image_view opens already showing the
# annotated detection image (no need to pick the topic by hand):
ros2 launch ur3e_vision_pick_place detector.launch.py

# Or just the viewer on its own:
ros2 run rqt_image_view rqt_image_view
# Select topic: /detected_objects_debug
```

### Run Full Vision Pick-and-Place
```bash
ros2 run ur3e_vision_pick_place forward_kinematics
ros2 run ur3e_vision_pick_place object_detector
ros2 run ur3e_vision_pick_place frame_transformer
ros2 run ur3e_vision_pick_place vision_pick_and_place --ros-args -p target_color:=red
```

### Color Tuning Tool
```bash
ros2 run ur3e_vision_pick_place color_tuner
# Hover mouse over objects to see HSV values
```

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/overhead_camera/image` | sensor_msgs/Image | Raw RGB image (fixed top-down view) |
| `/overhead_camera/depth_image` | sensor_msgs/Image | Depth image (grey = distance, not a bug) |
| `/overhead_camera/camera_info` | sensor_msgs/CameraInfo | Camera intrinsics |
| `/overhead_camera/points` | sensor_msgs/PointCloud2 | Depth point cloud (not viewable in rqt_image_view) |
| `/detected_objects_debug` | sensor_msgs/Image | Image with detections |
| `/detected_object/{red,green,blue}` | geometry_msgs/PointStamped | 3D position of each detected object, camera frame |
| `/path_target_pose` | geometry_msgs/PoseStamped | Pick target in base_link, published on /pick_color trigger |

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
│   # NOTE: the camera now lives in the rh_p12_rn_a_description package
│   # (overhead_camera, fixed to base_link) — see rh_p12_rn_a_gripper.xacro
│   # and rh_p12_rn_a.gazebo.
├── ur3e_vision_pick_place/
│   ├── helper_functions/          # pure math, NO ROS — importable anywhere
│   │   ├── kinematics.py          # Pinocchio FK/IK (seed-regularized DLS)
│   │   ├── dh_kinematics.py       # DH-parameter FK (cross-check)
│   │   ├── trajectory_profile.py  # trapezoidal profiles + angle utils
│   │   ├── path_interpolation.py  # linear + SLERP cartesian interpolation
│   │   └── color_detection.py     # HSV masks, blob finding, pinhole model
│   ├── robot_config.py            # joint names, limits, home, EE frame
│   ├── ros_utils.py               # numpy profile -> JointTrajectory msg
│   ├── color_detector_v2.py       # ROS nodes below: thin wiring only
│   ├── color_tuner.py
│   ├── object_detector.py
│   ├── frame_transformer.py
│   ├── forward_kinematics.py
│   ├── forward_kinematics_pure.py
│   ├── inverse_kinematics.py
│   ├── trapezoidal_planner.py
│   ├── path_interpolation.py
│   ├── pick_and_place.py
│   ├── vision_pick_and_place.py
│   └── joint_tester.py
├── test/
│   ├── test_helper_functions.py   # profile/interp/DH math (numpy only)
│   └── test_color_detection.py    # detection math (numpy + cv2 only)
├── package.xml
├── setup.py
└── README.md
```

## Running the Unit Tests

The math in `helper_functions/` is tested without ROS, Gazebo, or a
robot — plain pytest with numpy/OpenCV:

```bash
cd src/ur3e_vision_pick_place
python3 -m pytest test/test_helper_functions.py test/test_color_detection.py -v
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

Uses HSV color space for robust detection (ranges defined once in
`helper_functions/color_detection.py`):
- **Red**: H=0-10, 170-180
- **Green**: H=35-85
- **Blue**: H=100-130

Shadow removal via morphological operations (erosion + dilation).

## Sample Output
```
[INFO] red: pixel=(320, 171) -> world=(0.000, 0.331, 0.430)
[INFO] green: pixel=(229, 168) -> world=(-0.063, 0.333, 0.430)
[INFO] blue: pixel=(409, 171) -> world=(0.062, 0.331, 0.430)
```

## Future Work

- [x] Pick and place execution (vision_pick_and_place)
- [x] Depth camera integration (object_detector)
- [ ] MoveIt integration for motion planning
- [ ] YOLO-based object detection

## Author

**Tejas Murkute**

## License

MIT License
