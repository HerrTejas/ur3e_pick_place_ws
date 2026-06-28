#!/usr/bin/env python3
"""Joint-Space Planner Node.

Subscribes:  /cmd_joint_positions  (std_msgs/Float64MultiArray, 6 floats)
Subscribes:  /joint_states         (sensor_msgs/JointState)
Publishes:   /scaled_joint_trajectory_controller/joint_trajectory
                                   (trajectory_msgs/JointTrajectory)

When a 6-DOF target arrives on /cmd_joint_positions, generates a
synchronized trapezoidal profile from current joints to target and
publishes the trajectory to the controller.

All math is in helper_functions.trapezoidal_planner.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from ur3e_vision_pick_place.helper_functions.trapezoidal_planner import (
    trapezoidal_joint_trajectory,
)


class JointPlannerNode(Node):

    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    # Trapezoidal profile parameters
    VMAX = 0.3        # rad/s, max joint velocity for the slowest joint
    AMAX = 0.6        # rad/s^2
    DT = 0.05         # 20 Hz waypoints
    MIN_DURATION = 4.0  # seconds, floor for short moves

    def __init__(self):
        super().__init__('joint_planner')

        self.current_q = None

        self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/cmd_joint_positions', self.cmd_cb, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        self.get_logger().info('Joint Planner Ready')
        self.get_logger().info(
            '  Send 6-DOF target to /cmd_joint_positions')

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_q = np.array(
                [positions[n] for n in self.JOINT_NAMES])

    def cmd_cb(self, msg):
        self.get_logger().info(
            f'cmd_cb FIRED with {len(msg.data)} values')
        
        if self.current_q is None:
            self.get_logger().error('No joint states yet, cannot plan')
            return
        if len(msg.data) != 6:
            self.get_logger().error(
                f'Expected 6 joint values, got {len(msg.data)}')
            return
    
        target_q = np.array(msg.data)
        self.get_logger().info(
            f'Planning to: {[f"{x:.2f}" for x in target_q]}')
    
        try:
            waypoints = trapezoidal_joint_trajectory(
                self.current_q, target_q,
                vmax=self.VMAX, amax=self.AMAX,
                dt=self.DT, min_duration=self.MIN_DURATION)
            self.get_logger().info(
                f'Generated {len(waypoints)} waypoints')
        except Exception as e:
            self.get_logger().error(
                f'trapezoidal_joint_trajectory crashed: {e}')
            import traceback
            traceback.print_exc()
            return

        waypoints = trapezoidal_joint_trajectory(
            self.current_q, target_q,
            vmax=self.VMAX, amax=self.AMAX,
            dt=self.DT, min_duration=self.MIN_DURATION)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.JOINT_NAMES

        for t, positions, velocities in waypoints:
            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            point.velocities = velocities.tolist()
            point.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t - int(t)) * 1e9))
            traj_msg.points.append(point)

        self.traj_pub.publish(traj_msg)
        self.get_logger().info(
            f'Published {len(traj_msg.points)} waypoints, '
            f'duration {waypoints[-1][0]:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = JointPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
