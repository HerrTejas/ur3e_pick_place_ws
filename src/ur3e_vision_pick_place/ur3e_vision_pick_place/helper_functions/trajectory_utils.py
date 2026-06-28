"""Trajectory utility functions.

Pure math, no ROS. Used by both planners and the controller.
"""

import math
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def shortest_angular_distance(start: float, end: float) -> float:
    """Shortest signed angular distance from start to end.

    Returns a value in [-pi, pi]: positive = counter-clockwise.
    Prevents the robot from spinning the long way around.
    """
    return wrap_angle(end - start)


def joint_cost(q_current, q_target) -> float:
    """Sum of absolute wrapped joint distances.

    Useful as a "how much will the robot have to move" metric for
    comparing IK solutions or rejecting bad branches.
    """
    return float(sum(
        abs(shortest_angular_distance(q_current[j], q_target[j]))
        for j in range(len(q_current))
    ))


def joint_step_norm(q_current, q_next) -> float:
    """L2 norm of wrapped per-joint deltas.

    Catches IK branch shifts where every joint moves moderately
    but no single joint trips a per-joint threshold.
    """
    diffs = [shortest_angular_distance(q_current[j], q_next[j])
             for j in range(len(q_current))]
    return float(np.linalg.norm(diffs))


def joint_step_max(q_current, q_next) -> float:
    """Largest wrapped per-joint delta.

    Catches single-joint flips (wrist or elbow snap).
    """
    diffs = [shortest_angular_distance(q_current[j], q_next[j])
             for j in range(len(q_current))]
    return float(max(abs(d) for d in diffs))


def trajectory_is_finite(trajectory) -> bool:
    """Check trajectory has no inf or NaN values."""
    for point in trajectory.points:
        for position in point.positions:
            if math.isinf(position) or math.isnan(position):
                return False
        for velocity in point.velocities:
            if math.isinf(velocity) or math.isnan(velocity):
                return False
    return True


def has_velocities(trajectory) -> bool:
    """Check that velocities are defined for every point."""
    for point in trajectory.points:
        if len(point.velocities) != len(point.positions):
            return False
    return True
