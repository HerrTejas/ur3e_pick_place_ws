#!/usr/bin/env python3
"""DH-parameter forward kinematics — pure math, no ROS.

A from-scratch cross-check on the Pinocchio-based FK in kinematics.py.

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.dh_kinematics import (
        UR3E_DH_PARAMS, BASE_FRAME_CORRECTION, dh_matrix, rotation_to_quaternion,
    )

Author: Tejas
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

#: UR3e DH parameters: [d, a, alpha] per joint.
UR3E_DH_PARAMS = [
    [0.15185,  0,        np.pi / 2],   # Joint 1
    [0,        -0.24355, 0],           # Joint 2
    [0,        -0.2132,  0],           # Joint 3
    [0.13105,  0,        np.pi / 2],   # Joint 4
    [0.08535,  0,        -np.pi / 2],  # Joint 5
    [0.0921,   0,        0],           # Joint 6
]

#: Base frame correction (180 deg rotation around Z).
BASE_FRAME_CORRECTION = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])


def dh_matrix(theta: float, d: float, a: float, alpha: float) -> npt.NDArray[np.float64]:
    """Compute the standard DH transformation matrix for one joint.

    Args:
        theta: Joint angle, radians.
        d: Link offset, metres.
        a: Link length, metres.
        alpha: Link twist, radians.

    Returns:
        (4, 4) homogeneous transform from this joint to the next.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])


def rotation_to_quaternion(
    R: npt.NDArray[np.float64],
) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a quaternion.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (x, y, z, w) quaternion components.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return x, y, z, w


def compute_fk_dh(
    q: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Chain the per-joint DH transforms into a base-to-EE pose.

    Args:
        q: (6,) joint positions, radians.

    Returns:
        ``(position, rotation)`` — (3,) EE position and (3, 3) EE
        rotation matrix, both in base_link.
    """
    T_total = np.eye(4)
    for i in range(6):
        d, a, alpha = UR3E_DH_PARAMS[i]
        T_total = T_total @ dh_matrix(q[i], d, a, alpha)

    T_total = BASE_FRAME_CORRECTION @ T_total
    return T_total[:3, 3], T_total[:3, :3]
