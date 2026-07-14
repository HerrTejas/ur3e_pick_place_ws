#!/usr/bin/env python3
"""Pinocchio-based forward/inverse kinematics — pure math, no ROS.

Arm-agnostic: joint list, joint limits and DOF count are all read from
the loaded Pinocchio model (same approach as Synapse's CLIK solvers),
so any serial arm works by pointing ``load_pinocchio`` at its URDF and
naming its TCP frame. Nothing UR-specific lives here — the UR3e values
in robot_config.py are only the *defaults* for this project.

Any node can import these directly:
    from ur3e_vision_pick_place.helper_functions.kinematics import (
        load_pinocchio, compute_fk, compute_ik,
    )

Author: Tejas
"""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ur3e_vision_pick_place.robot_config import EE_FRAME, URDF_PATH


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
        ValueError: If ``ee_frame`` does not exist in the model.
            (Without this check ``getFrameId`` silently returns an
            out-of-range id and FK/IK fail much later, cryptically.)
        Exception: Propagated as-is from Pinocchio if the URDF cannot
            be parsed (callers already wrap this in a try/except).
    """
    # Auto-export the URDF on first use in a fresh session — this used
    # to be a manual `xacro ... > /tmp/ur3e.urdf` step every time.
    # Lazy import so this module stays importable without ament/xacro
    # (e.g. in Isaac Sim, where callers pass their own urdf_path).
    import os
    if not os.path.exists(urdf_path):
        from ur3e_vision_pick_place.urdf_export import ensure_urdf
        ensure_urdf(urdf_path)

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    if not model.existFrame(ee_frame):
        frames = [f.name for f in model.frames]
        raise ValueError(
            f"Frame '{ee_frame}' not found in URDF '{urdf_path}'. "
            f"Available frames: {frames}")
    ee_frame_id = model.getFrameId(ee_frame)
    return model, data, ee_frame_id


def _actuated_joints(model: pin.Model, joint_names: Optional[List[str]] = None) -> list:
    """The model's actuated 1-DOF joints, in kinematic-chain order.

    Args:
        model: Loaded Pinocchio model.
        joint_names: Optional explicit selection (e.g. just the arm when
            the URDF also contains gripper joints). Defaults to every
            single-DOF joint in the model.

    Returns:
        List of Pinocchio joint models.
    """
    if joint_names is not None:
        return [model.joints[model.getJointId(n)] for n in joint_names]
    # Joint 0 is Pinocchio's "universe"; fixed joints never appear in
    # model.joints, so every remaining nv==1 entry is an actuated joint.
    return [model.joints[jid] for jid in range(1, model.njoints)
            if model.joints[jid].nv == 1]


def _joint_limits(model: pin.Model, joints: list) -> List[Tuple[float, float]]:
    """Per-joint (min, max) angle limits, read from the model.

    Continuous joints (nq == 2, cos/sin parameterization) have no angle
    limits and get ``(-inf, inf)`` so unwrapping is unrestricted.
    """
    limits = []
    for jm in joints:
        if jm.nq == 1:
            low = float(model.lowerPositionLimit[jm.idx_q])
            high = float(model.upperPositionLimit[jm.idx_q])
            if not (np.isfinite(low) and np.isfinite(high)) or low >= high:
                low, high = -np.inf, np.inf
        else:
            low, high = -np.inf, np.inf
        limits.append((low, high))
    return limits


def _decode_angles(q: npt.NDArray[np.float64], joints: list) -> npt.NDArray[np.float64]:
    """Read each joint's angle out of a full configuration vector.

    A continuous joint stores (cos, sin) in the configuration, so its
    angle is recovered via atan2.
    """
    angles = np.empty(len(joints))
    for k, jm in enumerate(joints):
        qi = jm.idx_q
        angles[k] = q[qi] if jm.nq == 1 else np.arctan2(q[qi + 1], q[qi])
    return angles


def _angles_to_configuration(
    model: pin.Model, joints: list, angles: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build a full configuration vector from per-joint angles.

    Uses ``pin.integrate`` from the neutral configuration so continuous
    joints get their (cos, sin) entries set consistently.
    """
    dq = np.zeros(model.nv)
    for k, jm in enumerate(joints):
        dq[jm.idx_v] = angles[k]
    return pin.integrate(model, pin.neutral(model), dq)


def compute_fk(
    model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
    q: npt.NDArray[np.float64],
    joint_names: Optional[List[str]] = None,
) -> pin.SE3:
    """Compute the end-effector pose for a given joint configuration.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        q: (n,) joint angles in radians, one per actuated joint.
        joint_names: Optional explicit joint selection (see
            :func:`_actuated_joints`).

    Returns:
        End-effector pose as a Pinocchio SE3 (``.translation``,
        ``.rotation``).
    """
    joints = _actuated_joints(model, joint_names)
    q_full = _angles_to_configuration(
        model, joints, np.asarray(q, dtype=float)[:len(joints)])
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    return data.oMf[ee_frame_id]


def _unwrap_to_seed(
    q_sol: npt.NDArray[np.float64],
    q_seed: npt.NDArray[np.float64],
    limits: List[Tuple[float, float]],
) -> npt.NDArray[np.float64]:
    """Express each joint as the 2*pi-equivalent closest to the seed.

    A pose is identical for any joint shifted by a full turn, but the
    solver can return e.g. wrist_3 = +6 rad when -0.3 rad is the same
    pose — and a joint-space move then spins it the long way around
    ("the gripper rotated like crazy"). This rewrites the solution onto
    the turn nearest the current joints (shortest path), while staying
    inside the joint limits so a +-180 deg joint is never pushed out of
    range.
    """
    q_out = q_sol.copy()
    for i, (low, high) in enumerate(limits):
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
    joint_names: Optional[List[str]] = None,
) -> Optional[npt.NDArray[np.float64]]:
    """Solve IK for a target TCP pose, regularized toward the seed.

    Seed-regularized damped least squares. A standard DLS step drives
    the end-effector to the target; an added seed pull of weight ``mu``
    keeps the solution on the same IK branch as the current joints.
    ``mu`` is tiny while the pose error is large (so it never blocks
    convergence) and ramps up as the error shrinks — locking the
    wrist/elbow onto the seed branch near the target instead of drifting
    to an equivalent-but-flipped config (a random-restart solver can
    jump branches, which sweeps the arm wildly). The result is unwrapped
    onto the turn nearest the seed.

    The Jacobian is mapped into error space with ``pin.Jlog6`` (the
    same correction Synapse's CLIK solver applies), which makes each
    step an exact Newton step on the SE3 log error instead of a
    small-error approximation — better convergence when the target is
    far or the orientation error is large.

    Joint list and limits come from the model itself, so this works for
    any serial arm's URDF, not just the UR3e.

    Args:
        model, data, ee_frame_id: Pinocchio model, from :func:`load_pinocchio`.
        target_pos: (3,) desired position, metres.
        target_rot: (3,3) desired rotation matrix.
        q_seed: (n,) initial guess, radians (usually the current joints).
        max_iter: Maximum solver iterations.
        tol: Convergence tolerance on the SE3 log error norm.
        damping: Base DLS damping factor.
        joint_names: Optional explicit joint selection (see
            :func:`_actuated_joints`).

    Returns:
        (n,) joint angles closest to the seed, or ``None`` if the solver
        could not get within 0.05 of the target pose.
    """
    joints = _actuated_joints(model, joint_names)
    n = len(joints)
    v_idx = np.array([jm.idx_v for jm in joints])
    limits = _joint_limits(model, joints)

    target_se3 = pin.SE3(target_rot, target_pos)
    seed = np.asarray(q_seed[:n], dtype=float)

    q = _angles_to_configuration(model, joints, seed)

    alpha = 0.5
    mu_start, mu_max = 0.002, 0.15
    # Only let the seed pull grow once the pose error is small: ramping
    # from a large error pulls so hard toward a far seed (e.g. HOME ->
    # grasp) that the solve stalls short of the target. Gating it to the
    # final approach keeps far-seed convergence while still locking the
    # IK branch near the solution.
    mu_ramp_err = 0.05
    best_error = np.inf
    best_q = seed.copy()

    for _ in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_se3 = data.oMf[ee_frame_id]

        # LOCAL-frame error pairs with the LOCAL Jacobian below.
        i_T_d = current_se3.inverse() * target_se3
        error = pin.log6(i_T_d).vector
        if not np.isfinite(error).all():
            return None
        err_norm = float(np.linalg.norm(error))

        q_current = _decode_angles(q, joints)

        if err_norm < best_error:
            best_error = err_norm
            best_q = q_current.copy()
        if err_norm < tol:
            return _unwrap_to_seed(best_q, seed, limits)

        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL)[:, v_idx]
        # Map the Jacobian into error space (exact derivative of the
        # log6 error). Jlog6 -> identity as the error -> 0, so this only
        # changes behaviour far from the target — where it helps.
        J = pin.Jlog6(i_T_d.inverse()) @ J

        # Seed pull along the shortest angular path.
        seed_err = seed - q_current
        seed_err -= np.round(seed_err / (2 * np.pi)) * (2 * np.pi)

        # mu ramps mu_start -> mu_max as err_norm falls mu_ramp_err -> 0.
        mu = mu_start + (mu_max - mu_start) * max(0.0, 1.0 - err_norm / mu_ramp_err)
        dv = np.linalg.solve(
            J.T @ J + (damping + mu) * np.eye(n),
            J.T @ error + mu * seed_err)

        full_dv = np.zeros(model.nv)
        full_dv[v_idx] = alpha * dv
        q = pin.integrate(model, q, full_dv)
        if np.linalg.norm(dv) < 1e-7:
            break

    if best_error < 0.05:
        return _unwrap_to_seed(best_q, seed, limits)
    return None
