#!/usr/bin/env python3
"""
Inverse Kinematics Node using Pinocchio

The IK math is in standalone functions at the top.
The Node class below is just ROS wiring that calls them.

Other nodes (like path_interpolation) can import the math directly:
    from ur3e_vision_pick_place.inverse_kinematics import load_pinocchio, compute_ik

Author: Tejas
"""

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray


# ══════════════════════════════════════════════════════════════════
#  Pure math — no ROS, importable by any node
# ══════════════════════════════════════════════════════════════════

def load_pinocchio(urdf_path="/tmp/ur3e.urdf", ee_frame="tool0"):
    """Load Pinocchio model. Returns (model, data, ee_frame_id)."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId(ee_frame)
    return model, data, ee_frame_id


def compute_ik(model, data, ee_frame_id, target_pos, target_rot,
               q_seed, max_iter=200, tol=1e-4, damping=1e-6):
    """
    Damped least-squares IK.

    Args:
        model, data, ee_frame_id: from load_pinocchio()
        target_pos:  (3,) desired position
        target_rot:  (3,3) desired rotation matrix
        q_seed:      (6,) initial guess
        max_iter:    max iterations
        tol:         convergence tolerance
        damping:     damping factor

    Returns:
        (6,) joint angles or None if failed
    """
    target_se3 = pin.SE3(target_rot, target_pos)

    q = np.zeros(model.nq)
    q[:6] = q_seed[:6]

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current_se3 = data.oMf[ee_frame_id]
        error = pin.log6(current_se3.inverse() * target_se3).vector
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q[:6].copy()

        J = pin.computeFrameJacobian(
            model, data, q, ee_frame_id,
            pin.ReferenceFrame.LOCAL
        )
        J_arm = J[:, :6]

        JtJ = J_arm.T @ J_arm + damping * np.eye(6)
        delta_q = np.linalg.solve(JtJ, J_arm.T @ error)
        q[:6] += delta_q
        q[:6] = (q[:6] + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    # Return best effort if close enough
    if error_norm < 0.01:
        return q[:6].copy()
    return None


# ══════════════════════════════════════════════════════════════════
#  ROS Node — just wiring, calls the functions above
# ══════════════════════════════════════════════════════════════════

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

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


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()