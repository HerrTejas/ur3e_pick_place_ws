#!/usr/bin/env python3
"""Trapezoidal velocity profile generator — pure math, no ROS.

Any node can import this directly:
    from ur3e_vision_pick_place.helper_functions.trajectory_profile import TrajectoryProfile

Author: Tejas
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt


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
            distance at each sample, and total profile duration.
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

        while t <= t_total + 1e-9:
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
