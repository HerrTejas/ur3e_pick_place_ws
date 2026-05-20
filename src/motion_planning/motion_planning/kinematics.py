import numpy as np


class UR3eKinematics:
    """
    Forward and Inverse Kinematics for UR3e using modified DH parameters.
    Reference: Universal Robots UR3e datasheet (modified DH convention).

    Modified DH parameters per joint: (a, d, alpha, theta_offset)
      a     : link length (m)
      d     : link offset (m)
      alpha : link twist (rad)
    """

    DH_PARAMS = [
        # (a,         d,        alpha,       theta_offset)
        (0.0,       0.15185,  np.pi / 2,   0.0),   # Joint 1
        (-0.24365,  0.0,      0.0,          0.0),   # Joint 2
        (-0.21325,  0.0,      0.0,          0.0),   # Joint 3
        (0.0,       0.13105,  np.pi / 2,   0.0),   # Joint 4
        (0.0,       0.08535, -np.pi / 2,   0.0),   # Joint 5
        (0.0,       0.0921,   0.0,          0.0),   # Joint 6
    ]

    # Joint limits [min, max] in radians (UR3e: ±360 deg)
    JOINT_LIMITS = [(-2 * np.pi, 2 * np.pi)] * 6

    def _dh_transform(self, a, d, alpha, theta):
        """Compute single modified DH transformation matrix (4x4)."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,    -st,      0,     a     ],
            [st*ca,  ct*ca,  -sa,   -sa*d  ],
            [st*sa,  ct*sa,   ca,    ca*d  ],
            [0,      0,       0,     1     ],
        ])

    def forward_kinematics(self, joint_angles):
        """
        Compute end-effector pose from joint angles.

        Args:
            joint_angles: array-like of 6 angles in radians

        Returns:
            np.ndarray [x, y, z, rx, ry, rz] — position (m) and ZYX Euler angles (rad)
        """
        T = np.eye(4)
        for i, (a, d, alpha, theta_off) in enumerate(self.DH_PARAMS):
            T = T @ self._dh_transform(a, d, alpha, joint_angles[i] + theta_off)

        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        R = T[:3, :3]
        # ZYX Euler angles from rotation matrix
        ry = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        cos_ry = np.cos(ry)
        if abs(cos_ry) > 1e-6:
            rz = np.arctan2(R[1, 0] / cos_ry, R[0, 0] / cos_ry)
            rx = np.arctan2(R[2, 1] / cos_ry, R[2, 2] / cos_ry)
        else:
            # Gimbal lock: rz = 0 convention
            rz = 0.0
            rx = np.arctan2(R[0, 1], R[1, 1])
        return np.array([x, y, z, rx, ry, rz])

    def _numerical_jacobian(self, joint_angles, delta=1e-6):
        """Compute 6x6 Jacobian numerically via finite differences."""
        J = np.zeros((6, 6))
        pose0 = self.forward_kinematics(joint_angles)
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = delta
            pose_plus = self.forward_kinematics(joint_angles + dq)
            J[:, i] = (pose_plus - pose0) / delta
        return J

    def inverse_kinematics(self, target_pose, q_init=None, max_iter=150, tol=1e-4):
        """
        Numerical IK using damped least-squares (Levenberg-Marquardt style).

        Args:
            target_pose: [x, y, z, rx, ry, rz]
            q_init:      initial joint angles (defaults to zeros)
            max_iter:    maximum iterations
            tol:         convergence tolerance on pose error norm

        Returns:
            np.ndarray of 6 joint angles, or None if IK failed to converge
        """
        q = np.zeros(6) if q_init is None else np.array(q_init, dtype=float)
        target = np.array(target_pose, dtype=float)
        damping = 0.05

        for iteration in range(max_iter):
            current_pose = self.forward_kinematics(q)
            error = target - current_pose
            # Wrap orientation error to [-pi, pi]
            error[3:] = (error[3:] + np.pi) % (2 * np.pi) - np.pi

            if np.linalg.norm(error) < tol:
                return q

            J = self._numerical_jacobian(q)
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damping ** 2 * np.eye(6), error)
            q = q + dq

            # Enforce joint limits
            for i, (lo, hi) in enumerate(self.JOINT_LIMITS):
                q[i] = np.clip(q[i], lo, hi)

        # Accept result if close enough even without full convergence
        final_error = np.linalg.norm(self.forward_kinematics(q) - target)
        if final_error < 0.01:
            return q
        return None
