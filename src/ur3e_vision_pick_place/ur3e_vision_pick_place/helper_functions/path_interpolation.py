"""Cartesian path generation: linear position + SLERP orientation.

Pure math, no ROS, no IK. Generates a sequence of (time, position,
rotation_matrix) tuples that the caller will then run IK against.

This separation lets the IK loop live in the node where the robot
model lives, while the path math stays clean and testable.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _trapezoid_scalar(L: float, vmax: float, amax: float, dt: float):
    """Generate a 1-D trapezoidal arclength profile for distance L.

    Returns
    -------
    times : np.ndarray
    arclength : np.ndarray
        Distance traveled at each time, in [0, L].
    total_duration : float
    """
    if L < 1e-8:
        return np.array([0.0]), np.array([0.0]), 0.0

    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc * t_acc

    if 2 * d_acc > L:
        # Triangular profile (no cruise phase)
        t_acc = np.sqrt(L / amax)
        t_flat = 0.0
        t_total = 2 * t_acc
    else:
        d_flat = L - 2 * d_acc
        t_flat = d_flat / vmax
        t_total = 2 * t_acc + t_flat

    times = []
    arclengths = []
    t = 0.0
    while t <= t_total + 1e-9:
        if t < t_acc:
            s = 0.5 * amax * t * t
        elif t < t_acc + t_flat:
            s = d_acc + vmax * (t - t_acc)
        else:
            t_dec = t - (t_acc + t_flat)
            s = (d_acc + vmax * t_flat +
                 vmax * t_dec - 0.5 * amax * t_dec * t_dec)
        times.append(t)
        arclengths.append(min(s, L))
        t += dt

    return np.array(times), np.array(arclengths), t_total


def cartesian_path(start_pos: np.ndarray, start_quat: np.ndarray,
                   end_pos: np.ndarray, end_quat: np.ndarray,
                   vmax: float = 0.2, amax: float = 0.2,
                   dt: float = 0.05,
                   t_min: float = 2.0,
                   duration_scale: float = 10.0):
    """Generate a Cartesian path with synchronized position + orientation.

    Position interpolates linearly. Orientation interpolates via SLERP.
    Both follow the same normalized parameter (0..1), driven by a
    trapezoidal profile on the position arclength.

    For pure-rotation moves (position distance ~ 0), the trapezoid
    is run on the orientation magnitude instead.

    Parameters
    ----------
    start_pos, end_pos : np.ndarray, shape (3,)
        Position in base frame (meters).
    start_quat, end_quat : np.ndarray, shape (4,)
        Orientation as [x, y, z, w] quaternion.
    vmax : float
        Max linear velocity (m/s) — also used as max angular velocity (rad/s)
        for pure-rotation moves.
    amax : float
        Max linear acceleration (m/s^2).
    dt : float
        Time step between waypoints (seconds).
    t_min : float
        Minimum total duration (seconds).
    duration_scale : float
        Duration scale factor: time = max(t_min, duration_scale * distance).

    Returns
    -------
    waypoints : list of (time, position, rotation_matrix)
        - time: float, seconds
        - position: np.ndarray, shape (3,) — interpolated position
        - rotation_matrix: np.ndarray, shape (3, 3) — interpolated orientation
    """
    r_start = R.from_quat(start_quat)
    r_end = R.from_quat(end_quat)

    L_pos = float(np.linalg.norm(end_pos - start_pos))
    L_ori = float((r_start.inv() * r_end).magnitude())

    # Already at target
    if L_pos < 1e-6 and L_ori < 1e-6:
        return [(0.0, start_pos.copy(), r_start.as_matrix())]

    # Position drives timing unless we're doing a pure rotation
    L_primary = L_pos if L_pos > 1e-6 else L_ori

    # Stretch duration to at least t_min, scaled with distance
    desired_duration = max(t_min, duration_scale * L_primary)

    # If the profile would be too short, scale vmax/amax down so the
    # profile naturally takes desired_duration. Otherwise use raw vmax/amax.
    if L_primary / vmax < desired_duration * 0.75:
        vmax_eff = L_primary / (desired_duration * 0.75)
        amax_eff = vmax_eff / (desired_duration * 0.25)
    else:
        vmax_eff = vmax
        amax_eff = amax

    times, arclengths, _ = _trapezoid_scalar(L_primary, vmax_eff, amax_eff, dt)

    # SLERP setup
    rots = R.concatenate([r_start, r_end])
    slerp = Slerp([0.0, 1.0], rots)

    waypoints = []
    for t, s in zip(times, arclengths):
        # Normalized parameter in [0, 1]
        u = min(s / L_primary, 1.0) if L_primary > 1e-6 else 1.0

        position = (1.0 - u) * start_pos + u * end_pos
        rotation = slerp(u).as_matrix()

        waypoints.append((float(t), position, rotation))

    return waypoints
