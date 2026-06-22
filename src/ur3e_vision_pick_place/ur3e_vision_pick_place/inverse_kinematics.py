#!/usr/bin/env python3
"""
Inverse Kinematics Node using Pinocchio

The IK math is in standalone functions at the top.
The Node class below is just ROS wiring that calls them.

Other nodes (like path_interpolation) can import the math directly:
    from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik

Author: Tejas
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

from ur3e_vision_pick_place.robot_config import (
    EE_FRAME, JOINT_LIMITS_RAD, JOINT_NAMES, URDF_PATH,
)


# ══════════════════════════════════════════════════════════════════
#  Pure math — no ROS, importable by any node
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
#  ROS Node — just wiring, calls the functions above
# ══════════════════════════════════════════════════════════════════

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')

        self.joint_names = JOINT_NAMES

        # Load Pinocchio
        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
            self.get_logger().info(f'Model nq: {self.model.nq}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            self.get_logger().error('Run: xacro ... > /tmp/ur3e.urdf')
            return

        # State
        self.current_q = np.zeros(self.model.nq)
        self.joints_received = False
        self.target_pose = None

        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(PoseStamped, '/target_ee_pose', self.pose_cb, 10)

        # Publisher
        self.joint_pub = self.create_publisher(Float64MultiArray, '/ik_solution', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Inverse Kinematics Node Ready!')
        self.get_logger().info('Subscribing to /target_ee_pose')

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            for i, name in enumerate(self.joint_names):
                self.current_q[i] = positions[name]
            self.joints_received = True

    def pose_cb(self, msg):
        self.target_pose = msg.pose

    def timer_cb(self):
        if self.target_pose is None or not self.joints_received:
            return

        # Extract target
        pos = np.array([
            self.target_pose.position.x,
            self.target_pose.position.y,
            self.target_pose.position.z
        ])
        quat = [
            self.target_pose.orientation.x,
            self.target_pose.orientation.y,
            self.target_pose.orientation.z,
            self.target_pose.orientation.w
        ]
        rot = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()

        # Call the standalone function
        q = compute_ik(self.model, self.data, self.ee_frame_id,
                       pos, rot, self.current_q[:6])

        if q is not None:
            msg = Float64MultiArray()
            msg.data = q.tolist()
            self.joint_pub.publish(msg)
            self.current_q[:6] = q
        else:
            self.get_logger().warn('IK failed to converge for target pose')


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
