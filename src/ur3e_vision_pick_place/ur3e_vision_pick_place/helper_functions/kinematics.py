"""Forward and inverse kinematics for the UR3e arm.

Pure math, no ROS. Wraps Pinocchio with a clean interface.

Functions
---------
load_pinocchio   : load URDF, return (model, data, ee_frame_id)
compute_fk       : joint angles -> end-effector pose
compute_ik       : end-effector pose -> joint angles (damped least-squares)
"""

import numpy as np
import pinocchio as pin


def load_pinocchio(urdf_path: str = "/tmp/ur3e.urdf",
                   ee_frame: str = "tool0"):
    """Load Pinocchio model from URDF.

    Parameters
    ----------
    urdf_path : str
        Path to URDF file.
    ee_frame : str
        End-effector frame name in URDF.

    Returns
    -------
    model : pinocchio.Model
    data : pinocchio.Data
    ee_frame_id : int
        Frame ID for end-effector lookups.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId(ee_frame)
    return model, data, ee_frame_id


def compute_fk(model, data, ee_frame_id: int, q: np.ndarray):
    """Compute forward kinematics.

    Parameters
    ----------
    model, data : Pinocchio model and data.
    ee_frame_id : int
        From load_pinocchio().
    q : np.ndarray, shape (nq,)
        Joint angles. Only the first 6 entries are used for the arm.

    Returns
    -------
    position : np.ndarray, shape (3,)
        End-effector position in base frame.
    quaternion : np.ndarray, shape (4,)
        End-effector orientation as [x, y, z, w] quaternion.
    """
    q_full = np.zeros(model.nq)
    q_full[:6] = q[:6]

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    ee_pose = data.oMf[ee_frame_id]
    position = np.array(ee_pose.translation)
    quat = pin.Quaternion(ee_pose.rotation)
    quaternion = np.array([quat.x, quat.y, quat.z, quat.w])

    return position, quaternion


def compute_ik(model, data, ee_frame_id: int,
               target_pos: np.ndarray, target_rot: np.ndarray,
               q_seed: np.ndarray,
               max_iter: int = 200, tol: float = 1e-4,
               damping: float = 1e-6):
    """Damped least-squares inverse kinematics.

    Parameters
    ----------
    model, data : Pinocchio model and data.
    ee_frame_id : int
        From load_pinocchio().
    target_pos : np.ndarray, shape (3,)
        Desired end-effector position.
    target_rot : np.ndarray, shape (3, 3)
        Desired end-effector rotation matrix.
    q_seed : np.ndarray, shape (6,) or (nq,)
        Initial guess for joint angles. The branch this seed
        sits in determines which IK solution is returned.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on the SE(3) error norm.
    damping : float
        Damping factor for the pseudo-inverse (Levenberg-Marquardt).

    Returns
    -------
    q : np.ndarray, shape (6,) or None
        Joint angles (wrapped to [-pi, pi]) on success, None on failure.
    """
    target_se3 = pin.SE3(target_rot, target_pos)

    q = np.zeros(model.nq)
    q[:6] = q_seed[:6]

    error_norm = np.inf

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current_se3 = data.oMf[ee_frame_id]
        error = pin.log6(current_se3.inverse() * target_se3).vector
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q[:6].copy()

        J = pin.computeFrameJacobian(
            model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL
        )
        J_arm = J[:, :6]

        JtJ = J_arm.T @ J_arm + damping * np.eye(6)
        delta_q = np.linalg.solve(JtJ, J_arm.T @ error)
        q[:6] += delta_q
        # Wrap each joint to [-pi, pi] to keep solutions canonical
        q[:6] = (q[:6] + np.pi) % (2 * np.pi) - np.pi

    # Best-effort return if we got close
    if error_norm < 0.01:
        return q[:6].copy()
    return None
