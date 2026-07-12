#!/usr/bin/env python3
"""Small shared ROS-message helpers.

The pure math lives in ``helper_functions/`` (no ROS imports there);
this module is the one place where numpy profiles get packed into ROS
trajectory messages, shared by every motion node.

Author: Tejas
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt

from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def seconds_to_duration(t: float) -> Duration:
    """Convert float seconds to a builtin_interfaces Duration."""
    sec = int(t)
    return Duration(sec=sec, nanosec=int((t - sec) * 1e9))


def profile_to_trajectory_msg(
    joint_names: List[str],
    times: npt.NDArray[np.float64],
    positions: npt.NDArray[np.float64],
    velocities: Optional[npt.NDArray[np.float64]] = None,
) -> JointTrajectory:
    """Pack a sampled profile into a JointTrajectory message.

    Args:
        joint_names: Controller joint names, in column order.
        times: (N,) sample times, seconds from start.
        positions: (N, J) joint positions per sample.
        velocities: Optional (N, J) joint velocities per sample. Include
            them whenever possible — without velocities the controller
            interpolates at constant speed and the motion jerks at every
            waypoint.

    Returns:
        A populated JointTrajectory (header left to the caller).
    """
    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for i, t in enumerate(times):
        point = JointTrajectoryPoint()
        point.positions = np.asarray(positions[i], dtype=float).tolist()
        if velocities is not None:
            point.velocities = np.asarray(velocities[i], dtype=float).tolist()
        point.time_from_start = seconds_to_duration(float(t))
        msg.points.append(point)
    return msg
