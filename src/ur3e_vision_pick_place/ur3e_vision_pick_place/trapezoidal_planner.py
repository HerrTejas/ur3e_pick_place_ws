#!/usr/bin/env python3
"""
Trapezoidal Planner - Coordinated Joint-Space Motion

The profile math lives in helper_functions/trajectory_profile.py
(joint_trapezoid + shortest_angular_distance). This file is just ROS
wiring: it listens for a 6-joint target on /cmd_joint_positions and
streams a synchronized trapezoidal JointTrajectory to the controller.

Author: Tejas
"""

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from ur3e_vision_pick_place.helper_functions.trajectory_profile import (
    joint_trapezoid, shortest_angular_distance, wrap_angle,
)
from ur3e_vision_pick_place.robot_config import HOME, JOINT_NAMES
from ur3e_vision_pick_place.ros_utils import profile_to_trajectory_msg

#: Cruise velocity of the largest-moving joint (rad/s) and floor on the
#: move duration (s) — short moves are stretched so they stay gentle.
MAX_JOINT_VEL = 0.5
MIN_MOVE_SEC = 3.0
TRAJ_DT = 0.1


class TrapezoidalPlanner(Node):
    def __init__(self):
        super().__init__('trapezoidal_planner')

        self.joint_names = JOINT_NAMES

        self.HOME = HOME
        self.RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]

        self.current_positions = None

        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(Float64MultiArray, '/cmd_joint_positions', self.cmd_cb, 10)

        self.traj_pub = self.create_publisher(
            JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

    def joint_state_cb(self, msg):
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]

    def cmd_cb(self, msg):
        if len(msg.data) != 6:
            self.get_logger().error('Need exactly 6 joint values!')
            return
        self.move_to(list(msg.data))

    def move_to(self, target):
        """Stream a trapezoidal trajectory from the current joints to ``target``.

        The goal is rebuilt from the *unwrapped* current position plus the
        shortest angular delta per joint: wrapping the start would snap
        the robot if a joint sits outside [-pi, pi] (UR wrists/pan often
        do), and taking the long way around is what used to spin the arm.
        """
        if self.current_positions is None:
            self.get_logger().error('No joint states yet!')
            return

        start = np.asarray(self.current_positions, dtype=float)
        deltas = np.array([
            shortest_angular_distance(wrap_angle(start[j]), wrap_angle(target[j]))
            for j in range(6)
        ])
        goal = start + deltas

        times, positions, velocities = joint_trapezoid(
            start, goal, MAX_JOINT_VEL, TRAJ_DT, MIN_MOVE_SEC)

        self.get_logger().info(f'Moving to: {[f"{t:.2f}" for t in target]}')
        self.get_logger().info(
            f'Duration: {times[-1]:.2f}s, max joint move: {np.max(np.abs(deltas)):.2f} rad')

        traj_msg = profile_to_trajectory_msg(
            self.joint_names, times, positions, velocities)
        self.traj_pub.publish(traj_msg)
        self.get_logger().info(f'Published {len(traj_msg.points)} points')


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalPlanner()

    node.get_logger().info('Waiting for joint states...')
    while node.current_positions is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    node.get_logger().info(f'Current: {[f"{p:.2f}" for p in node.current_positions]}')
    node.get_logger().info('')
    node.get_logger().info('Listening for manual commands on /cmd_joint_positions')
    node.get_logger().info(f'HOME: {node.HOME}')
    node.get_logger().info(f'RED_BOX_GRASP: {node.RED_BOX_GRASP}')

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
