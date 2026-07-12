#!/usr/bin/env python3
"""Unit tests for the pure-math helper_functions modules.

These run without ROS or a simulator:
    cd src/ur3e_vision_pick_place && python -m pytest test/test_helper_functions.py

(kinematics.py needs Pinocchio + the exported URDF, so it is covered by
running the IK node against the sim rather than here.)
"""

import numpy as np
import pytest

from ur3e_vision_pick_place.helper_functions.trajectory_profile import (
    TrajectoryProfile, joint_trapezoid, shortest_angular_distance, wrap_angle,
)
from ur3e_vision_pick_place.helper_functions.path_interpolation import (
    interpolate_cartesian_path, quat_rotation_distance, quat_slerp,
    quat_to_rotation_matrix,
)
from ur3e_vision_pick_place.helper_functions.dh_kinematics import (
    compute_fk_dh, rotation_to_quaternion,
)


# ── angles ────────────────────────────────────────────────────────────

def test_wrap_angle():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert wrap_angle(2 * np.pi) == pytest.approx(0.0)


def test_shortest_angular_distance_takes_short_way():
    # 350 deg -> 10 deg should be +20 deg, not -340.
    a, b = np.deg2rad(-10), np.deg2rad(10)
    assert shortest_angular_distance(a, b) == pytest.approx(np.deg2rad(20))
    assert shortest_angular_distance(b, a) == pytest.approx(np.deg2rad(-20))


# ── joint_trapezoid ───────────────────────────────────────────────────

def test_joint_trapezoid_endpoints_exact():
    start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    goal = np.array([1.2, -0.9, 1.4, -1.8, -1.6, -0.3])
    times, pos, vel = joint_trapezoid(start, goal, max_vel=0.5, dt=0.05)

    np.testing.assert_allclose(pos[0], start, atol=1e-12)
    np.testing.assert_allclose(pos[-1], goal, atol=1e-12)
    np.testing.assert_allclose(vel[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(vel[-1], 0.0, atol=1e-12)


def test_joint_trapezoid_respects_max_vel():
    start = np.zeros(6)
    goal = np.array([2.0, 0.5, -1.0, 0.1, 0.0, -2.0])
    max_vel = 0.5
    times, pos, vel = joint_trapezoid(start, goal, max_vel, dt=0.01)
    assert np.max(np.abs(vel)) <= max_vel + 1e-9


def test_joint_trapezoid_velocity_consistent_with_position():
    # Finite-difference of positions should match reported velocities.
    start = np.zeros(2)
    goal = np.array([1.0, -0.4])
    times, pos, vel = joint_trapezoid(start, goal, max_vel=0.5, dt=0.01)
    fd = np.gradient(pos, times, axis=0)
    np.testing.assert_allclose(fd, vel, atol=0.02)


def test_joint_trapezoid_zero_move():
    q = np.array([0.3, -1.0, 0.5, 0.0, 0.1, 0.0])
    times, pos, vel = joint_trapezoid(q, q.copy(), max_vel=0.5, dt=0.05)
    assert len(times) == 1
    np.testing.assert_allclose(pos[0], q)
    np.testing.assert_allclose(vel[0], 0.0)


def test_joint_trapezoid_min_duration():
    start = np.zeros(6)
    goal = np.full(6, 0.01)  # tiny move
    times, _, _ = joint_trapezoid(start, goal, max_vel=0.5, dt=0.05,
                                  min_duration=2.0)
    assert times[-1] == pytest.approx(2.0)


# ── TrajectoryProfile ─────────────────────────────────────────────────

def test_trapezoid_time_scaled_reaches_target_exactly():
    profile = TrajectoryProfile()
    # t_total deliberately not a multiple of dt.
    t, s, T = profile.trapezoid_time_scaled(L=0.37, vmax=0.3, amax=0.3, dt=0.05)
    assert t[-1] == pytest.approx(T)
    assert s[-1] == pytest.approx(0.37)
    assert np.all(np.diff(s) >= -1e-12)  # monotonic


def test_trapezoid_multi_synchronized():
    profile = TrajectoryProfile()
    t, s, T = profile.trapezoid_multi([0.4, 0.1], vmax=0.3, amax=0.3, dt=0.05)
    assert s.shape[0] == 2
    assert s[0][-1] == pytest.approx(0.4)
    assert s[1][-1] == pytest.approx(0.1)
    # Same fraction of travel at every sample (synchronized).
    np.testing.assert_allclose(s[1] / 0.1, s[0] / 0.4, atol=1e-9)


# ── quaternions / cartesian interpolation ─────────────────────────────

def test_quat_slerp_endpoints_and_norm():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])  # 90 deg about Y
    np.testing.assert_allclose(quat_slerp(q0, q1, 0.0), q0, atol=1e-12)
    np.testing.assert_allclose(quat_slerp(q0, q1, 1.0), q1, atol=1e-12)
    qm = quat_slerp(q0, q1, 0.5)
    assert np.linalg.norm(qm) == pytest.approx(1.0)
    # Halfway = 45 deg about Y.
    assert quat_rotation_distance(q0, qm) == pytest.approx(np.pi / 4, abs=1e-9)


def test_quat_slerp_takes_short_arc():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = -np.array([0.0, np.sin(0.1), 0.0, np.cos(0.1)])  # same rotation, negated
    qm = quat_slerp(q0, q1, 0.5)
    assert quat_rotation_distance(q0, qm) == pytest.approx(0.1, abs=1e-6)


def test_quat_to_rotation_matrix_orthonormal():
    q = np.array([0.3, -0.2, 0.5, 0.7])
    R = quat_to_rotation_matrix(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_interpolate_cartesian_path_endpoints():
    start_pos = np.array([0.3, 0.0, 0.4])
    end_pos = np.array([0.2, 0.25, 0.15])
    q_down = np.array([1.0, 0.0, 0.0, 0.0])  # gripper facing down
    q_tilt = quat_slerp(q_down, np.array([0.9239, 0.0, 0.3827, 0.0]), 1.0)

    times, pos, quats = interpolate_cartesian_path(
        start_pos, q_down, end_pos, q_tilt, vmax=0.3, amax=0.3, dt=0.05)

    np.testing.assert_allclose(pos[0], start_pos, atol=1e-9)
    np.testing.assert_allclose(pos[-1], end_pos, atol=1e-9)
    assert quat_rotation_distance(quats[0], q_down) == pytest.approx(0.0, abs=1e-6)
    assert quat_rotation_distance(quats[-1], q_tilt) == pytest.approx(0.0, abs=1e-6)
    # Straight line: every waypoint on the start-end segment.
    d = end_pos - start_pos
    for p in pos:
        cross = np.cross(p - start_pos, d)
        assert np.linalg.norm(cross) < 1e-9


def test_interpolate_cartesian_path_zero_move():
    p = np.array([0.3, 0.0, 0.4])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    times, pos, quats = interpolate_cartesian_path(p, q, p, q, 0.3, 0.3, 0.05)
    assert len(times) == 1


# ── DH forward kinematics ─────────────────────────────────────────────

def test_dh_fk_rotation_orthonormal():
    q = [0.3, -1.2, 0.8, -1.5, 0.4, 0.2]
    pos, rot = compute_fk_dh(q)
    assert np.all(np.isfinite(pos))
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)


def test_dh_fk_zero_config_position():
    # At q = 0 the UR3e lies stretched along its link lengths; the TCP
    # distance from base must equal the reach along the DH chain.
    pos, _ = compute_fk_dh([0.0] * 6)
    # Known UR3e DH values (d1..d6, a2, a3).
    a2, a3 = -0.24355, -0.2132
    d1, d4, d5, d6 = 0.15185, 0.13105, 0.08535, 0.0921
    # At zero config: x = -(a2+a3) rotated by base correction, z = d1 - d5... use invariants:
    # horizontal reach & height derived from the standard UR zero pose.
    expected_planar = abs(a2 + a3)
    assert np.hypot(pos[0], pos[1]) == pytest.approx(
        np.hypot(expected_planar, d4 + d6), rel=1e-6)
    assert pos[2] == pytest.approx(d1 - d5, abs=1e-9)


def test_rotation_to_quaternion_unit_norm():
    _, rot = compute_fk_dh([0.5, -0.7, 1.1, -0.3, 0.9, -1.4])
    x, y, z, w = rotation_to_quaternion(rot)
    assert x * x + y * y + z * z + w * w == pytest.approx(1.0)
