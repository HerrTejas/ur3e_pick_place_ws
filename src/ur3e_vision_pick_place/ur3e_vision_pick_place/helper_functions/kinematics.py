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


def _bound_joint(q: float, low: float, high: float) -> float:
    """Bring a single joint value into its physical limits.

    Joints with more than a full turn of range (e.g. ``wrist_3``, which
    spans 720 deg) are wrapped to the equivalent angle inside
    ``[low, high)`` instead of being clamped, so the solver can keep
    spinning that joint freely. Joints with a single bounded range are
    clamped directly. This replaces a blanket wrap to ``[-pi, pi]``,
    which silently cut off part of ``shoulder_pan``'s real +-200 deg
    range and forced ``wrist_3`` into the wrong period — a likely
    source of the intermittent IK failures.

    Args:
        q: Candidate joint value, radians.
        low: Lower joint limit, radians.
        high: Upper joint limit, radians.

    Returns:
        The bounded joint value, radians.
    """
    span = high - low
    if span >= 2 * np.pi:
        q = (q - low) % span + low
    return float(np.clip(q, low, high))


def _bound_joints(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply :func:`_bound_joint` to all 6 arm joints."""
    bounded = q.copy()
    for i, (low, high) in enumerate(JOINT_LIMITS_RAD):
        bounded[i] = _bound_joint(q[i], low, high)
    return bounded


def _solve_from_seed(
    model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    target_se3: pin.SE3,
    q_seed: npt.NDArray[np.float64],
    max_iter: int,
    tol: float,
    damping: float,
) -> Tuple[Optional[npt.NDArray[np.float64]], float]:
    """Damped least-squares (Levenberg-Marquardt) IK from one seed.

    The damping factor adapts each iteration: it shrinks after a step
    that reduces the pose error (Gauss-Newton-like, fast convergence)
    and grows after a step that doesn't (more robust near singularities,
    e.g. the UR wrist-2 = 0 singularity), instead of using one fixed
    damping value for the whole solve.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        target_se3: Desired end-effector pose.
        q_seed: (6,) initial joint guess.
        max_iter: Maximum solver iterations.
        tol: Convergence tolerance on the SE3 log error norm.
        damping: Initial Levenberg-Marquardt damping factor.

    Returns:
        A ``(q, error_norm)`` tuple. ``q`` is ``None`` if the solve
        never reached within 0.01 of ``tol``.
    """
    q = np.zeros(model.nq)
    q[:6] = q_seed[:6]

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    error_norm = np.linalg.norm(
        pin.log6(data.oMf[ee_frame_id].inverse() * target_se3).vector)

    lam = damping
    for _ in range(max_iter):
        if error_norm < tol:
            return q[:6].copy(), error_norm

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_se3 = data.oMf[ee_frame_id]
        error = pin.log6(current_se3.inverse() * target_se3).vector

        J = pin.computeFrameJacobian(
            model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL)
        J_arm = J[:, :6]

        JtJ = J_arm.T @ J_arm + lam * np.eye(6)
        delta_q = np.linalg.solve(JtJ, J_arm.T @ error)

        q_trial = q.copy()
        q_trial[:6] = _bound_joints(q[:6] + delta_q)

        pin.forwardKinematics(model, data, q_trial)
        pin.updateFramePlacements(model, data)
        trial_error_norm = np.linalg.norm(
            pin.log6(data.oMf[ee_frame_id].inverse() * target_se3).vector)

        if trial_error_norm < error_norm:
            q = q_trial
            error_norm = trial_error_norm
            lam = max(lam * 0.7, 1e-8)
        else:
            lam = min(lam * 2.0, 1e6)

    if error_norm < 0.01:
        return q[:6].copy(), error_norm
    return None, error_norm


def compute_ik(
    model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    target_pos: npt.NDArray[np.float64],
    target_rot: npt.NDArray[np.float64],
    q_seed: npt.NDArray[np.float64],
    max_iter: int = 200,
    tol: float = 1e-4,
    damping: float = 1e-6,
    num_restarts: int = 3,
    rng_seed: Optional[int] = None,
) -> Optional[npt.NDArray[np.float64]]:
    """Solve IK for a target TCP pose, seeded from the current joints.

    First tries from ``q_seed`` (the current/last joint state) so paths
    stay smooth. If that doesn't converge — typically near a singularity
    or a joint-limit boundary — it retries from a few random seeds drawn
    within the joint limits before giving up, which is what removes most
    of the sporadic "IK failed" cases the single-seed solver hit.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        target_pos: (3,) desired position, metres.
        target_rot: (3,3) desired rotation matrix.
        q_seed: (6,) initial guess, radians (usually the current joints).
        max_iter: Max iterations per solve attempt.
        tol: Convergence tolerance on the SE3 log error norm.
        damping: Initial Levenberg-Marquardt damping factor.
        num_restarts: Extra random-seed attempts if the seeded solve fails.
        rng_seed: Optional seed for the random restarts, for repeatable tests.

    Returns:
        (6,) joint angles within their physical limits, or ``None`` if
        every attempt failed to converge.
    """
    target_se3 = pin.SE3(target_rot, target_pos)

    q, _ = _solve_from_seed(
        model, data, ee_frame_id, target_se3, q_seed, max_iter, tol, damping)
    if q is not None:
        return q

    rng = np.random.default_rng(rng_seed)
    best_q, best_error = None, np.inf
    for _ in range(num_restarts):
        random_seed = np.array([
            rng.uniform(low, high) for low, high in JOINT_LIMITS_RAD
        ])
        q, error_norm = _solve_from_seed(
            model, data, ee_frame_id, target_se3, random_seed,
            max_iter, tol, damping)
        if q is not None:
            return q
        if error_norm < best_error:
            best_q, best_error = q, error_norm

    return best_q
