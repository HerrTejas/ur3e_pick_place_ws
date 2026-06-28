"""Joint-space trapezoidal trajectory generator.

Pure math, no ROS. One function that takes start and end joint
configurations and returns a synchronized 6-DOF trapezoidal profile
with positions AND velocities at every waypoint.

Synchronization: all joints share one time base, scaled to whichever
joint has the largest distance. This means every joint accelerates,
cruises, and decelerates together, arriving at the target simultaneously.
"""

import numpy as np

from ur3e_vision_pick_place.helper_functions.trajectory_utils import (
    shortest_angular_distance,
)


def trapezoidal_joint_trajectory(start_q, end_q,
                                 vmax: float = 0.3,
                                 amax: float = 0.6,
                                 dt: float = 0.05,
                                 min_duration: float = 4.0):
    """Synchronized trapezoidal velocity profile for 6-DOF joint motion.

    All joints follow a single trapezoidal time profile sized to the
    joint with the largest signed wrapped distance. Each joint moves
    along the shortest angular path (no long-way-around spinning).

    Parameters
    ----------
    start_q : sequence of 6 floats
        Current joint angles (radians).
    end_q : sequence of 6 floats
        Target joint angles (radians).
    vmax : float
        Maximum joint velocity (rad/s) for the slowest-cruising joint.
        Other joints scale down proportionally.
    amax : float
        Maximum joint acceleration (rad/s^2).
    dt : float
        Time step between waypoints (seconds).
    min_duration : float
        Minimum total trajectory duration (seconds). Short moves get
        stretched to this duration to keep motion gentle.

    Returns
    -------
    waypoints : list of (time, positions, velocities)
        - time: float, seconds since trajectory start
        - positions: np.ndarray, shape (6,) — joint angles
        - velocities: np.ndarray, shape (6,) — joint angular velocities
        First waypoint has t=0 and zero velocity.
        Last waypoint has zero velocity (rest-to-rest).
    """
    start = np.asarray(start_q, dtype=float)
    end = np.asarray(end_q, dtype=float)
    n_joints = len(start)

    # Shortest signed delta per joint (handles wraparound)
    deltas = np.array([
        shortest_angular_distance(start[j], end[j])
        for j in range(n_joints)
    ])
    distances = np.abs(deltas)
    directions = np.sign(deltas)

    max_distance = float(np.max(distances))

    # Degenerate: already there
    if max_distance < 1e-8:
        return [(0.0, start.copy(), np.zeros(n_joints))]

    # Size the trapezoid for the slowest joint, then derive duration
    duration = max(min_duration, max_distance / vmax)

    # Phase splits: 25% accel, 50% cruise, 25% decel
    t_accel = duration * 0.25
    t_cruise = duration * 0.50
    t_decel = duration * 0.25

    # Per-joint peak velocity, derived so that distance_j is covered
    # in exactly `duration` seconds.
    # Trapezoid area = v_max_j * (t_accel/2 + t_cruise + t_decel/2)
    #                = v_max_j * (duration - t_accel)
    # The 0.75 factor below is (1 - 0.25) = the fraction of duration
    # that's not pure accel.
    v_peak = distances / (0.75 * duration)
    v_peak = np.where(distances < 1e-8, 0.0, v_peak)
    a_peak = np.where(t_accel > 0, v_peak / t_accel, 0.0)

    # Sanity: clamp acceleration to amax (rare for normal moves)
    if np.max(a_peak) > amax:
        scale = amax / np.max(a_peak)
        v_peak = v_peak * scale
        a_peak = a_peak * scale
        # Recompute duration to match the slower velocities
        duration = duration / scale
        t_accel = duration * 0.25
        t_cruise = duration * 0.50

    waypoints = []
    times = np.arange(0.0, duration + dt, dt)

    for t in times:
        positions = np.zeros(n_joints)
        velocities = np.zeros(n_joints)

        for j in range(n_joints):
            if distances[j] < 1e-8:
                positions[j] = start[j]
                continue

            d = directions[j]
            v = v_peak[j]
            a = a_peak[j]

            if t <= t_accel:
                # Accelerating
                pos_offset = 0.5 * a * t * t
                vel = a * t
            elif t <= t_accel + t_cruise:
                # Cruising
                t_in_cruise = t - t_accel
                d_accel = 0.5 * a * t_accel * t_accel
                pos_offset = d_accel + v * t_in_cruise
                vel = v
            else:
                # Decelerating
                t_in_decel = t - t_accel - t_cruise
                d_accel = 0.5 * a * t_accel * t_accel
                d_cruise = v * t_cruise
                pos_offset = (d_accel + d_cruise +
                              v * t_in_decel - 0.5 * a * t_in_decel * t_in_decel)
                vel = max(0.0, v - a * t_in_decel)

            positions[j] = start[j] + d * pos_offset
            velocities[j] = d * vel

        waypoints.append((float(t), positions, velocities))

    # Ensure final waypoint exactly matches the target with zero velocity.
    # If the last waypoint from the loop is already at or past `duration`,
    # replace it instead of appending — otherwise we'd write a non-monotonic
    # timestamp and the trajectory controller will reject the whole message.
    final_positions = start + directions * distances
    if waypoints and waypoints[-1][0] >= duration:
        waypoints[-1] = (float(duration), final_positions, np.zeros(n_joints))
    else:
        waypoints.append((float(duration), final_positions, np.zeros(n_joints)))

    return waypoints
