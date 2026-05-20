import numpy as np


def _euler_to_quat(rx, ry, rz):
    """Convert ZYX Euler angles (rad) to unit quaternion [w, x, y, z]."""
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return np.array([w, x, y, z])


def _quat_to_euler(q):
    """Convert unit quaternion [w, x, y, z] to ZYX Euler angles (rad)."""
    w, x, y, z = q
    rx = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_ry = 2 * (w * y - z * x)
    ry = np.arcsin(np.clip(sin_ry, -1.0, 1.0))
    rz = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([rx, ry, rz])


def _slerp(q0, q1, t):
    """
    Spherical linear interpolation between two unit quaternions.

    Args:
        q0, q1: unit quaternions [w, x, y, z]
        t:      scalar 0 -> 1

    Returns:
        interpolated unit quaternion [w, x, y, z]
    """
    dot = np.dot(q0, q1)
    if dot < 0.0:       # ensure shortest arc
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:    # quaternions nearly identical — fall back to lerp
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


class PathInterpolator:
    """
    Generalised interpolator for both joint-space and Cartesian-space paths.

    Both modes share a single init / get_pose_at interface:

        path_lengths = pi.init(start, end)           # 6-vector delta
        config       = pi.get_pose_at(displacements) # 6-vector absolute config

    The displacements argument is the output of TrajectoryPlanner.get_trajectory_points()
    — a 6-vector of signed displacements in the same units as path_lengths.

    Joint mode  (mode='joint', default)
    -----------
        start / end : [q0, q1, q2, q3, q4, q5]  (rad)
        All 6 elements use linear interpolation:
            result[i] = start[i] + displacements[i]

    Cartesian mode  (mode='cartesian')
    ---------------
        start / end : [x, y, z, rx, ry, rz]  (m, rad ZYX Euler)
        path_lengths: [dx, dy, dz, drx, dry, drz]
        Position  [0:3] — linear:  result[:3] = start[:3] + displacements[:3]
        Orientation [3:6] — SLERP: progress derived from
                            ||displacements[3:]|| / ||path_lengths[3:]||
                            so the rotation always follows the shortest arc.
    """

    MODE_JOINT = 'joint'
    MODE_CARTESIAN = 'cartesian'

    def __init__(self):
        self._mode = None
        self._start = None          # np.ndarray shape (6,)
        self._path_lengths = None   # np.ndarray shape (6,)

        # Cartesian-only: stored quaternions for SLERP
        self._start_quat = None     # np.ndarray [w, x, y, z]
        self._end_quat = None       # np.ndarray [w, x, y, z]
        self._ori_length = None     # scalar — ||path_lengths[3:]||

    # ------------------------------------------------------------------
    # Unified init
    # ------------------------------------------------------------------

    def init(self, start, end, mode=MODE_JOINT):
        """
        Set up a path from start to end.

        Args:
            start: array-like shape (6,)
                   Joint mode:     joint angles in rad
                   Cartesian mode: [x, y, z, rx, ry, rz]
            end:   array-like shape (6,) — same convention as start
            mode:  PathInterpolator.MODE_JOINT  (default)
                   PathInterpolator.MODE_CARTESIAN

        Returns:
            np.ndarray shape (6,) — signed path_lengths (end - start),
            with orientation deltas wrapped to [-pi, pi] in Cartesian mode.
        """
        self._mode = mode
        self._start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)

        if mode == self.MODE_CARTESIAN:
            # Position delta — straight Euclidean
            pos_delta = end[:3] - self._start[:3]
            # Orientation delta — wrap to shortest arc for scalar length
            ori_delta = end[3:] - self._start[3:]
            ori_delta = (ori_delta + np.pi) % (2 * np.pi) - np.pi
            self._path_lengths = np.concatenate([pos_delta, ori_delta])
            # Pre-compute quaternions for SLERP
            self._start_quat = _euler_to_quat(*self._start[3:])
            self._end_quat   = _euler_to_quat(*end[3:])
            self._ori_length = float(np.linalg.norm(ori_delta))
        else:
            # Joint mode: plain delta, no wrapping needed
            self._path_lengths = end - self._start
            self._start_quat = None
            self._end_quat   = None
            self._ori_length = None

        return self._path_lengths

    # ------------------------------------------------------------------
    # Unified get
    # ------------------------------------------------------------------

    def get_pose_at(self, displacements):
        """
        Evaluate the path given a displacement vector from the TrajectoryPlanner.

        Args:
            displacements: np.ndarray shape (6,) — signed displacement
                           (same units as path_lengths; output of TrajectoryPlanner)

        Returns:
            np.ndarray shape (6,) — absolute configuration
                Joint mode:     absolute joint angles (rad)
                Cartesian mode: [x, y, z, rx, ry, rz]
        """
        if self._start is None:
            raise RuntimeError('PathInterpolator: call init() first.')

        if self._mode == self.MODE_CARTESIAN:
            # Position: linear
            pos = self._start[:3] + displacements[:3]

            # Orientation: SLERP — derive progress from how far along
            # the orientation component we are
            ori_disp_norm = float(np.linalg.norm(displacements[3:]))
            if self._ori_length > 1e-9:
                t = np.clip(ori_disp_norm / self._ori_length, 0.0, 1.0)
            else:
                t = 1.0   # no rotation needed — SLERP returns start quat
            euler = _quat_to_euler(_slerp(self._start_quat, self._end_quat, t))

            return np.concatenate([pos, euler])
        else:
            # Joint mode: direct addition
            return self._start + displacements

