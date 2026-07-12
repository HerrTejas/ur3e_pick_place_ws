#!/usr/bin/env python3
"""Cartesian path interpolation — pure math, no ROS.

Linear position interpolation + quaternion SLERP, time-scaled with a
trapezoidal profile. Implemented with plain numpy (no scipy needed).

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.path_interpolation import (
        interpolate_cartesian_path, quat_slerp, quat_to_rotation_matrix,
    )

Quaternion convention throughout: ``[x, y, z, w]`` (ROS order).

Author: Tejas
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt

from ur3e_vision_pick_place.helper_functions.trajectory_profile import TrajectoryProfile


def quat_slerp(
    q0: npt.NDArray[np.float64], q1: npt.NDArray[np.float64], t: float,
) -> npt.NDArray[np.float64]:
    """Spherical linear interpolation between two unit quaternions.

    Args:
        q0: (4,) start quaternion, [x, y, z, w].
        q1: (4,) end quaternion, [x, y, z, w].
        t: Interpolation fraction in [0, 1].

    Returns:
        (4,) interpolated unit quaternion, [x, y, z, w].
    """
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    # q and -q are the same rotation: take the short arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Nearly parallel: fall back to normalized lerp (numerically stable).
    if dot > 0.9995:
        q = (1.0 - t) * q0 + t * q1
        return q / np.linalg.norm(q)

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def quat_to_rotation_matrix(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a quaternion to a 3x3 rotation matrix.

    Args:
        q: (4,) quaternion, [x, y, z, w]. Normalized internally.

    Returns:
        (3, 3) rotation matrix.
    """
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_rotation_distance(
    q0: npt.NDArray[np.float64], q1: npt.NDArray[np.float64],
) -> float:
    """Rotation angle (radians) needed to go from ``q0`` to ``q1``.

    Args:
        q0: (4,) start quaternion, [x, y, z, w].
        q1: (4,) end quaternion, [x, y, z, w].

    Returns:
        Geodesic angle in [0, pi].
    """
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def interpolate_cartesian_path(
    start_pos: npt.NDArray[np.float64],
    start_quat: npt.NDArray[np.float64],
    end_pos: npt.NDArray[np.float64],
    end_quat: npt.NDArray[np.float64],
    vmax: float,
    amax: float,
    dt: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate a straight-line cartesian path between two poses.

    Position moves along a straight line, orientation follows SLERP,
    and both are time-scaled together by one trapezoidal profile so
    they start and finish at the same instant.

    Args:
        start_pos: (3,) start position, metres.
        start_quat: (4,) start orientation, [x, y, z, w].
        end_pos: (3,) end position, metres.
        end_quat: (4,) end orientation, [x, y, z, w].
        vmax: Peak velocity of the dominant dimension (m/s or rad/s).
        amax: Peak acceleration of the dominant dimension.
        dt: Sample period, seconds.

    Returns:
        ``(times, positions, quaternions)`` — (N,) sample times, (N, 3)
        positions and (N, 4) unit quaternions ([x, y, z, w]) per sample.
        Returns single-sample arrays if start and end coincide.
    """
    start_pos = np.asarray(start_pos, dtype=float)
    end_pos = np.asarray(end_pos, dtype=float)
    start_quat = np.asarray(start_quat, dtype=float)
    end_quat = np.asarray(end_quat, dtype=float)

    L_pos = float(np.linalg.norm(end_pos - start_pos))
    L_ori = quat_rotation_distance(start_quat, end_quat)

    if max(L_pos, L_ori) < 1e-6:
        return (np.array([0.0]),
                end_pos[np.newaxis, :].copy(),
                (end_quat / np.linalg.norm(end_quat))[np.newaxis, :])

    # One shared trapezoid: position and orientation each get a row
    # scaled by their own distance over the same time base.
    profile = TrajectoryProfile()
    times, s_scaled, _ = profile.trapezoid_multi([L_pos, L_ori], vmax, amax, dt)
    s_pos, s_ori = s_scaled[0], s_scaled[1]

    positions = np.empty((len(times), 3))
    quaternions = np.empty((len(times), 4))
    for i in range(len(times)):
        f_pos = (s_pos[i] / L_pos) if L_pos > 1e-6 else 0.0
        f_ori = (s_ori[i] / L_ori) if L_ori > 1e-6 else 0.0
        # Both rows share the trapezoid's time base, so their fractions
        # agree; the max just picks whichever dimension actually moves.
        f = min(max(f_pos, f_ori), 1.0)

        positions[i] = (1.0 - f) * start_pos + f * end_pos
        quaternions[i] = quat_slerp(start_quat, end_quat, f)

    return times, positions, quaternions
