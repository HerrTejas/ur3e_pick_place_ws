#!/usr/bin/env python3
"""Unit tests for helper_functions/kinematics.py.

Needs Pinocchio and the exported URDF, so these run on the robot/sim
machine and are skipped automatically anywhere else:

    xacro ... > /tmp/ur3e.urdf   # see README
    cd src/ur3e_vision_pick_place && python3 -m pytest test/test_kinematics.py -v
"""

import os

import numpy as np
import pytest

pin = pytest.importorskip('pinocchio')

from ur3e_vision_pick_place.helper_functions import kinematics  # noqa: E402
from ur3e_vision_pick_place.robot_config import JOINT_NAMES, URDF_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.exists(URDF_PATH),
    reason=f'URDF not exported to {URDF_PATH} (see README)')


@pytest.fixture(scope='module')
def robot():
    return kinematics.load_pinocchio()


def test_load_rejects_unknown_frame():
    with pytest.raises(ValueError, match='not found'):
        kinematics.load_pinocchio(ee_frame='no_such_frame_xyz')


def test_actuated_joints_found(robot):
    model, _, _ = robot
    joints = kinematics._actuated_joints(model)
    assert len(joints) >= 6  # the 6 arm joints (URDF may add gripper DOFs)


def test_limits_read_from_model(robot):
    model, _, _ = robot
    joints = kinematics._actuated_joints(model)
    limits = kinematics._joint_limits(model, joints)
    assert len(limits) == len(joints)
    for low, high in limits:
        assert low < high


def test_fk_ik_roundtrip_random_poses(robot):
    """FK a random reachable config, solve IK back, FK must match.

    Uses joint_names=JOINT_NAMES exactly like the nodes do — the URDF
    also contains the gripper's revolute joints, which must not take
    part in arm IK.
    """
    model, data, ee_frame_id = robot
    joints = kinematics._actuated_joints(model, JOINT_NAMES)
    n = len(joints)
    limits = kinematics._joint_limits(model, joints)
    rng = np.random.default_rng(42)

    failures = 0
    trials = 30
    for _ in range(trials):
        # Random config comfortably inside the limits.
        q_true = np.array([
            rng.uniform(max(lo, -np.pi) * 0.8, min(hi, np.pi) * 0.8)
            for lo, hi in limits])
        target = kinematics.compute_fk(
            model, data, ee_frame_id, q_true, joint_names=JOINT_NAMES)

        # Seed near (but not at) the answer, like a real motion step.
        seed = q_true + rng.normal(0.0, 0.2, n)
        q_sol = kinematics.compute_ik(
            model, data, ee_frame_id,
            np.asarray(target.translation), np.asarray(target.rotation), seed,
            joint_names=JOINT_NAMES)

        if q_sol is None:
            failures += 1
            continue

        reached = kinematics.compute_fk(
            model, data, ee_frame_id, q_sol, joint_names=JOINT_NAMES)
        pos_err = np.linalg.norm(reached.translation - target.translation)
        rot_err = np.linalg.norm(
            pin.log3(np.asarray(reached.rotation).T @ np.asarray(target.rotation)))
        assert pos_err < 1e-3, f'position error {pos_err:.4f} m'
        assert rot_err < 1e-2, f'rotation error {rot_err:.4f} rad'

    # Allow the occasional genuinely hard pose, but not systematic failure.
    assert failures <= 2, f'{failures}/{trials} IK solves failed'


def test_ik_solution_stays_near_seed(robot):
    """The unwrap must return the turn closest to the seed."""
    model, data, ee_frame_id = robot
    n = len(JOINT_NAMES)

    q_home = np.zeros(n)
    q_home[1] = -1.57
    q_home[3] = -1.57
    target = kinematics.compute_fk(
        model, data, ee_frame_id, q_home, joint_names=JOINT_NAMES)

    q_sol = kinematics.compute_ik(
        model, data, ee_frame_id,
        np.asarray(target.translation), np.asarray(target.rotation), q_home,
        joint_names=JOINT_NAMES)
    assert q_sol is not None
    # No joint should be a full turn away from the seed.
    assert np.max(np.abs(q_sol - q_home)) < np.pi
