#!/usr/bin/env python3
"""Pinocchio-based forward/inverse kinematics — pure math, no ROS.

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.kinematics import (
        load_pinocchio, compute_fk, compute_ik,
    )

Author: Tejas
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ur3e_vision_pick_place.robot_config import EE_FRAME, JOINT_LIMITS_RAD, URDF_PATH


def load_pinocchio(
    urdf_path: str = URDF_PATH, ee_frame: str = EE_FRAME,
) -> Tuple[pin.Model, pin.Data, int]:
    """Load a Pinocchio model from a URDF file.

    Args:
        urdf_path: Path to the exported (xacro-processed) URDF.
        ee_frame: Name of the frame to use as the end-effector / TCP.

    Returns:
        A ``(model, data, ee_frame_id)`` tuple ready for FK/IK calls.

    Raises:
        Exception: Propagated as-is from Pinocchio if the URDF cannot
            be parsed (callers already wrap this in a try/except).
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId(ee_frame)
    return model, data, ee_frame_id


def compute_fk(
    model: pin.Model, data: pin.Data, ee_frame_id: int, q: npt.NDArray[np.float64],
) -> pin.SE3:
    """Compute the end-effector pose for a given joint configuration.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        q: (6,) joint positions, radians.

    Returns:
        End-effector pose as a Pinocchio SE3 (``.translation``,
        ``.rotation``).
    """
    q_full = np.zeros(model.nq)
    q_full[:6] = q[:6]
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame_id]


def _unwrap_to_seed(
    q_sol: npt.NDArray[np.float64], q_seed: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Express each joint as the 2*pi-equivalent closest to the seed.

    A pose is identical for any joint shifted by a full turn, but the
    solver can return e.g. wrist_3 = +6 rad when -0.3 rad is the same
    pose — and a joint-space move then spins it the long way around
    ("the gripper rotated like crazy"). This rewrites the solution onto
    the turn nearest the current joints (shortest path), while staying
    inside the joint limits so a +-180 deg joint is never pushed out of
    range. Ported from the unwrap_solution() on feature/humble.
    """
    q_out = q_sol.copy()
    for i, (low, high) in enumerate(JOINT_LIMITS_RAD):
        cand = q_sol[i] - round((q_sol[i] - q_seed[i]) / (2 * np.pi)) * 2 * np.pi
        # Nudge back inside the limits if the nearest turn fell outside.
        if cand < low:
            cand += 2 * np.pi
        elif cand > high:
            cand -= 2 * np.pi
        if low <= cand <= high:
            q_out[i] = cand
    return q_out


def compute_ik(
    model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    target_pos: npt.NDArray[np.float64],
    target_rot: npt.NDArray[np.float64],
    q_seed: npt.NDArray[np.float64],
    max_iter: int = 500,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> Optional[npt.NDArray[np.float64]]:
    """Solve IK for a target TCP pose, regularized toward the seed.

    Seed-regularized damped least squares (ported from the solver on
    ``feature/humble``). A standard DLS step drives the end-effector to
    the target; an added seed pull of weight ``mu`` keeps the solution on
    the same IK branch as the current joints. ``mu`` is tiny while the
    pose error is large (so it never blocks convergence) and ramps up as
    the error shrinks — locking the wrist/elbow onto the seed branch near
    the target instead of drifting to an equivalent-but-flipped config
    (the old random-restart solver could jump branches, which swept the
    arm wildly). The result is unwrapped onto the turn nearest the seed.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        target_pos: (3,) desired position, metres.
        target_rot: (3,3) desired rotation matrix.
        q_seed: (6,) initial guess, radians (usually the current joints).
        max_iter: Maximum solver iterations.
        tol: Convergence tolerance on the SE3 log error norm.
        damping: Base DLS damping factor.

    Returns:
        (6,) joint angles closest to the seed, or ``None`` if the solver
        could not get within 0.05 of the target pose.
    """
    arm_joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]

    # Configuration / velocity indices per arm joint. A continuous joint
    # (wrist_3) occupies 2 config entries (cos, sin) but 1 velocity entry.
    joints = [model.joints[model.getJointId(n)] for n in arm_joint_names]
    v_idx = np.array([jm.idx_v for jm in joints])

    target_se3 = pin.SE3(target_rot, target_pos)
    seed6 = np.asarray(q_seed[:6], dtype=float)

    # Seed via pin.integrate so the continuous joint's cos/sin is set right.
    dq0 = np.zeros(model.nv)
    dq0[v_idx] = seed6
    q = pin.integrate(model, pin.neutral(model), dq0)

    alpha = 0.5
    mu_start, mu_max = 0.002, 0.15
    # Only let the seed pull grow once the pose error is small. The
    # ported version ramped mu from err_norm = 1.0, which pulled so hard
    # toward a far seed (e.g. HOME -> grasp) that the solve stalled well
    # short of the target. Gating it to the final approach keeps far-seed
    # convergence (verified 59/60 random reachable poses) while still
    # locking the IK branch near the solution.
    mu_ramp_err = 0.05
    best_error = np.inf
    best_q6 = seed6.copy()

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_se3 = data.oMf[ee_frame_id]

        # LOCAL-frame error pairs with the LOCAL Jacobian below.
        error = pin.log6(current_se3.inverse() * target_se3).vector
        if not np.isfinite(error).all():
            return None
        err_norm = float(np.linalg.norm(error))

        # Current arm angles (decode the continuous joint via atan2).
        q_current_6 = np.empty(6)
        for k, jm in enumerate(joints):
            qi = jm.idx_q
            q_current_6[k] = q[qi] if jm.nq == 1 else np.arctan2(q[qi + 1], q[qi])

        if err_norm < best_error:
            best_error = err_norm
            best_q6 = q_current_6.copy()
        if err_norm < tol:
            return _unwrap_to_seed(best_q6, seed6)

        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL)[:, v_idx]

        # Seed pull along the shortest angular path.
        seed_err = seed6 - q_current_6
        seed_err -= np.round(seed_err / (2 * np.pi)) * (2 * np.pi)

        # mu ramps mu_start -> mu_max as err_norm falls mu_ramp_err -> 0.
        mu = mu_start + (mu_max - mu_start) * max(0.0, 1.0 - err_norm / mu_ramp_err)
        dv = np.linalg.solve(
            J.T @ J + (damping + mu) * np.eye(6),
            J.T @ error + mu * seed_err)

        full_dv = np.zeros(model.nv)
        full_dv[v_idx] = alpha * dv
        q = pin.integrate(model, q, full_dv)
        if np.linalg.norm(dv) < 1e-7:
            break

    if best_error < 0.05:
        return _unwrap_to_seed(best_q6, seed6)
    return None
