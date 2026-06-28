#!/usr/bin/env python3
"""Inverse Kinematics Node.

Provides IK as a topic-based service:
  Subscribes: /target_ee_pose  (geometry_msgs/PoseStamped)
  Publishes:  /ik_solution     (std_msgs/Float64MultiArray)
  Subscribes: /joint_states    (sensor_msgs/JointState) — for current pose

Multi-seed fallback strategy:
  1. Current joints — works for short follow-up moves
  2. Working-area config — known-good elbow-up branch for the workspace
  3. Pan sweep — covers both sides of the workspace [-pi/2, 0, pi/2]
  4. HOME — last resort

The first seed that converges below the cost threshold wins. Caller
doesn't have to think about IK seeding.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import pinocchio as pin

from ur3e_vision_pick_place.helper_functions.kinematics import (
    load_pinocchio,
    compute_ik,
)
from ur3e_vision_pick_place.helper_functions.trajectory_utils import (
    joint_cost,
)


class InverseKinematicsNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    HOME = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    WORKING_AREA = np.array([1.255, -0.98, 1.4, -1.8, -1.61, -0.3])

    # Solutions with cost above this against current joints are rejected.
    # 6 rad ~ all six joints moving 1 rad each, which is generous for
    # pick-and-place but catches the back-of-arm folds (cost ~ 7+).
    MAX_COST = 6.0

    def __init__(self):
        super().__init__('inverse_kinematics')

        try:
            self.model, self.data, self.ee_frame_id = load_pinocchio()
            self.get_logger().info(f'Loaded URDF: {self.model.name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load URDF: {e}')
            raise

        self.current_q = None

        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(
            PoseStamped, '/target_ee_pose', self.target_cb, 10)
        self.solution_pub = self.create_publisher(
            Float64MultiArray, '/ik_solution', 10)

        self.get_logger().info('Inverse Kinematics Ready')
        self.get_logger().info('  Send target to /target_ee_pose')
        self.get_logger().info('  Receive solution on /ik_solution')

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_q = np.array(
                [positions[n] for n in self.JOINT_NAMES])

    def candidate_seeds(self):
        """Return seeds in priority order for the multi-seed fallback."""
        seeds = []
        if self.current_q is not None:
            seeds.append(('current', self.current_q.copy()))
        seeds.append(('working_area', self.WORKING_AREA.copy()))

        # Pan sweep — covers both sides of the workspace
        for pan in [-1.5, 0.0, 1.5]:
            s = self.WORKING_AREA.copy()
            s[0] = pan
            seeds.append((f'pan_{pan:+.1f}', s))

        seeds.append(('home', self.HOME.copy()))
        return seeds

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray):
        """Run multi-seed IK, return first solution under cost threshold.

        Returns
        -------
        q : np.ndarray, shape (6,) or None
        seed_name : str — which seed produced the solution (for logging)
        cost : float — joint cost vs current
        """
        if self.current_q is None:
            self.get_logger().warn('No joint states yet, cannot solve IK')
            return None, None, None

        for name, seed in self.candidate_seeds():
            q = compute_ik(self.model, self.data, self.ee_frame_id,
                           target_pos, target_rot, seed)
            if q is None:
                continue

            cost = joint_cost(self.current_q, q)
            self.get_logger().info(
                f'  Seed "{name}" converged, cost {cost:.2f} rad')
            if cost <= self.MAX_COST:
                return q, name, cost

            self.get_logger().warn(
                f'  Seed "{name}" rejected: cost {cost:.2f} > {self.MAX_COST}')

        self.get_logger().error(
            'All seeds exhausted — no acceptable IK solution')
        return None, None, None

    def target_cb(self, msg):
        if self.current_q is None:
            self.get_logger().warn('No joint states yet')
            return

        pos = np.array([msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z])
        quat = [msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w]
        rot = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()

        q, seed_name, cost = self.solve(pos, rot)

        if q is None:
            self.get_logger().error(
                f'IK failed for ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})')
            return

        self.get_logger().info(
            f'IK ok: seed="{seed_name}", cost={cost:.2f} rad')

        out = Float64MultiArray()
        out.data = q.tolist()
        self.solution_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
