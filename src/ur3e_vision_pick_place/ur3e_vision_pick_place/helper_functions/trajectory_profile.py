#!/usr/bin/env python3
"""Trapezoidal velocity profile generators — pure math, no ROS.

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.trajectory_profile import (
        TrajectoryProfile, joint_trapezoid, wrap_angle, shortest_angular_distance,
    )

Author: Tejas
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def shortest_angular_distance(start: float, end: float) -> float:
    """Signed shortest rotation from ``start`` to ``end``.

    Positive = counter-clockwise, negative = clockwise. Using this for
    joint deltas prevents the robot from spinning the long way around.
    """
    return wrap_angle(end - start)


def joint_trapezoid(
    start: npt.NDArray[np.float64],
    goal: npt.NDArray[np.float64],
    max_vel: float,
    dt: float,
    min_duration: float = 2.0,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Synchronized joint-space trapezoidal profile (25/50/25 split).

    All joints share one duration — chosen so the largest-moving joint
    cruises at exactly ``max_vel`` — and each joint scales its own cruise
    velocity by its distance, so every joint starts and stops together.
    Both endpoint velocities are zero, and the final sample lands exactly
    on ``goal`` (a lone-point or truncated trajectory is what makes the
    controller jerk: velocity would jump 0 -> v at the start and v -> 0
    at the end).

    Args:
        start: (J,) current joint positions, radians.
        goal: (J,) target joint positions, radians. Callers decide the
            branch/winding (e.g. via :func:`shortest_angular_distance`);
            this function moves linearly from start to goal.
        max_vel: Cruise velocity of the largest-moving joint, rad/s.
        dt: Sample period, seconds.
        min_duration: Lower bound on the move duration, seconds.

    Returns:
        ``(times, positions, velocities)`` — (N,) sample times and
        (N, J) position/velocity samples, ready to be packed into a
        trajectory message by the caller.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    deltas = goal - start
    max_distance = float(np.max(np.abs(deltas)))

    # Already there: a single zero-velocity sample.
    if max_distance < 1e-6:
        return (np.array([0.0]),
                goal[np.newaxis, :].copy(),
                np.zeros((1, len(goal))))

    # The cruise phase covers 75% of the "distance-equivalent" time
    # (accel and decel ramps average half speed over 25% each), so the
    # duration that makes the peak velocity exactly max_vel is
    # L / (0.75 * max_vel). The old distance/max_vel formula overshot
    # the requested limit by 33%.
    duration = max(min_duration, max_distance / (0.75 * max_vel))
    t_accel = duration * 0.25
    t_cruise = duration * 0.50

    # linspace is clamped to exactly `duration`: stepping with np.arange
    # could emit a sample past the end, where the decel parabola reverses
    # (overshoot then pull back) -> jerk at the goal.
    n = int(np.ceil(duration / dt))
    times = np.linspace(0.0, duration, n + 1)

    distances = np.abs(deltas)
    directions = np.where(deltas >= 0, 1.0, -1.0)
    # Per-joint cruise velocity/acceleration for the shared duration.
    v_max = distances / (0.75 * duration)
    accel = v_max / t_accel

    positions = np.empty((len(times), len(deltas)))
    velocities = np.empty_like(positions)

    for i, t in enumerate(times):
        if t <= t_accel:
            vel = accel * t
            dist = 0.5 * accel * t ** 2
        elif t <= t_accel + t_cruise:
            t_c = t - t_accel
            vel = v_max
            dist = 0.5 * accel * t_accel ** 2 + v_max * t_c
        else:
            t_d = t - t_accel - t_cruise
            vel = np.maximum(0.0, v_max - accel * t_d)
            dist = (0.5 * v_max * t_accel + v_max * t_cruise
                    + v_max * t_d - 0.5 * accel * t_d ** 2)
        positions[i] = start + directions * dist
        velocities[i] = directions * vel

    # Joints that don't move stay put (avoid 0/0 noise from accel).
    still = distances < 1e-9
    positions[:, still] = start[still]
    velocities[:, still] = 0.0

    # Pin the final sample exactly on target with zero velocity.
    positions[-1] = goal
    velocities[-1] = 0.0
    return times, positions, velocities


class TrajectoryProfile:
    """Trapezoidal velocity profile generator."""

    def trapezoid_time_scaled(
        self, L: float, vmax: float, amax: float, dt: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Generate a trapezoidal profile for a 1D distance.

        Args:
            L: Total distance to travel.
            vmax: Maximum cruise velocity.
            amax: Maximum acceleration/deceleration.
            dt: Sample period, seconds.

        Returns:
            ``(t_array, s_array, t_total)`` — sample times, travelled
            distance at each sample, and total profile duration. The
            last sample is always exactly ``(t_total, L)`` so consumers
            land on the target instead of stopping one step short.
        """
        if L < 1e-8:
            return np.array([0.0]), np.array([0.0]), 0.0

        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc ** 2

        if 2 * d_acc > L:
            t_acc = np.sqrt(L / amax)
            t_flat = 0.0
            t_total = 2 * t_acc
        else:
            d_flat = L - 2 * d_acc
            t_flat = d_flat / vmax
            t_total = 2 * t_acc + t_flat

        t_list = []
        s_list = []
        t = 0.0

        while t < t_total - 1e-9:
            if t < t_acc:
                s = 0.5 * amax * t ** 2
            elif t < t_acc + t_flat:
                s = d_acc + vmax * (t - t_acc)
            else:
                t_dec = t - (t_acc + t_flat)
                s = d_acc + vmax * t_flat + vmax * t_dec - 0.5 * amax * t_dec ** 2

            t_list.append(t)
            s_list.append(min(s, L))
            t += dt

        # Final sample exactly on (t_total, L): float stepping above can
        # otherwise end short of the target.
        t_list.append(t_total)
        s_list.append(L)

        return np.array(t_list), np.array(s_list), t_total

    def trapezoid_multi(
        self, L_array: List[float], vmax: float, amax: float, dt: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Synchronized trapezoid for multiple dimensions.

        All dimensions share the same time base, scaled by their own
        distance, so multi-joint/multi-axis motion starts and stops
        together.

        Args:
            L_array: Distance travelled by each dimension.
            vmax: Maximum cruise velocity (of the longest dimension).
            amax: Maximum acceleration/deceleration.
            dt: Sample period, seconds.

        Returns:
            ``(t_array, s_scaled, t_total)`` — sample times, one
            distance row per dimension, and total profile duration.
        """
        L_array = np.array(L_array)
        L_max = np.max(L_array)

        t_list, s_base, T = self.trapezoid_time_scaled(L_max, vmax, amax, dt)

        s_scaled = []
        for L in L_array:
            if L > 1e-8:
                s_scaled.append(s_base * (L / L_max))
            else:
                s_scaled.append(np.zeros_like(s_base))

        return t_list, np.array(s_scaled), T
