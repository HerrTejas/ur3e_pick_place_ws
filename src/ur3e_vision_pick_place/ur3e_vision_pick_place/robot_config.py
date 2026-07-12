#!/usr/bin/env python3
"""Shared UR3e robot constants.

Single source of truth for joint names, URDF path, end-effector frame,
home pose and joint limits — previously copy-pasted across
forward_kinematics.py, inverse_kinematics.py, trapezoidal_planner.py,
path_interpolation.py, joint_tester.py and pick_and_place.py.

Author: Tejas
"""

from typing import List, Tuple

import numpy as np

#: Names of the 6 UR3e arm joints, in URDF/controller order.
JOINT_NAMES: List[str] = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

#: Single-joint gripper (rh_p12_rn_a) used for grasping.
GRIPPER_JOINTS: List[str] = ['rh_r1_joint']

#: Path the URDF is exported to (see README: `xacro ... > /tmp/ur3e.urdf`).
URDF_PATH: str = "/tmp/ur3e.urdf"

#: Pinocchio frame name used as the end-effector / TCP.
#: Must be the gripper's grasp point, NOT the wrist flange. tool0 sits
#: 0.125 m above the fingertips along the approach axis, so planning to
#: tool0 drove the fingers ~10 cm through the table on every grasp.
#: rh_p12_rn_ee is the gripper end frame (same orientation as tool0,
#: translated to the finger tips).
EE_FRAME: str = "rh_p12_rn_ee"

#: Safe joint-space home pose (rad), arms pointing down, elbow bent.
HOME: List[float] = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

#: Per-joint (min, max) position limits in radians, sourced from
#: Universal_Robots_ROS2_Description/config/ur3e/joint_limits.yaml
#: (UR3e User Manual, version 5.8). wrist_3 is the only joint with
#: more than one full turn of range; the others are hard end-stops.
#: NOTE: kept for reference/tests only — the IK solver now reads limits
#: from the URDF via the Pinocchio model, so it stays arm-agnostic.
JOINT_LIMITS_RAD: List[Tuple[float, float]] = [
    (np.deg2rad(-200.0), np.deg2rad(200.0)),  # shoulder_pan_joint
    (np.deg2rad(-180.0), np.deg2rad(180.0)),  # shoulder_lift_joint
    (np.deg2rad(-180.0), np.deg2rad(180.0)),  # elbow_joint
    (np.deg2rad(-180.0), np.deg2rad(180.0)),  # wrist_1_joint
    (np.deg2rad(-180.0), np.deg2rad(180.0)),  # wrist_2_joint
    (np.deg2rad(-360.0), np.deg2rad(360.0)),  # wrist_3_joint
]
