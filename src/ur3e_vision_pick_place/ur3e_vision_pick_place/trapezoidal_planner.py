#!/usr/bin/env python3
"""
Trapezoidal Planner - Coordinated Joint-Space Motion

The TrajectoryProfile math used elsewhere (e.g. path_interpolation.py)
lives in helper_functions/trajectory_profile.py. The Node below has its
own single-target trapezoidal move_to(), since it builds the
JointTrajectory message inline as it goes.

Author: Tejas
"""

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from ur3e_vision_pick_place.robot_config import HOME, JOINT_NAMES


class TrapezoidalPlanner(Node):
    def __init__(self):
        super().__init__('trapezoidal_planner')

        self.joint_names = JOINT_NAMES

        self.HOME = HOME
        self.RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]

        self.dt = 0.1
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

    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def shortest_angular_distance(self, start, end):
        """
        Compute the shortest distance between two angles.
        Returns a signed value: positive = counter-clockwise, negative = clockwise.
        This prevents the robot from spinning the long way around.
        """
        diff = self.wrap_angle(end - start)
        return diff

    def move_to(self, target):
        """
        Compute and execute trapezoidal trajectory to target position.

        Fixes:
        - Uses shortest angular path for each joint (no long-way spins)
        - Keeps the actual (unwrapped) start so the first point matches
          the controller's real state — wrapping it would snap the robot
          if a joint sits outside [-pi, pi] (UR wrists/pan often do)
        - Scales duration based on largest joint movement
        """
        if self.current_positions is None:
            self.get_logger().error('No joint states yet!')
            return

        # Keep the real start; only wrap to compute the shortest delta.
        start = list(self.current_positions)
        start_wrapped = [self.wrap_angle(p) for p in start]
        target_wrapped = [self.wrap_angle(t) for t in target]

        # Compute shortest angular distance for each joint
        deltas = [self.shortest_angular_distance(start_wrapped[j], target_wrapped[j])
                  for j in range(6)]

        # Apply the delta to the unwrapped start so the trajectory stays
        # continuous with the controller's current position.
        end = [start[j] + deltas[j] for j in range(6)]

        # Scale duration based on the largest joint movement
        # Max speed ~0.8 rad/s, minimum 2 seconds
        max_distance = max(abs(d) for d in deltas)
        duration = max(3.0, max_distance / 0.5)

        self.get_logger().info(f'Moving to: {[f"{t:.2f}" for t in target]}')
        self.get_logger().info(f'Duration: {duration:.2f}s, max joint move: {max_distance:.2f} rad')

        # Time parameters. Sample with linspace clamped to exactly
        # `duration`: np.arange could emit a point past duration, where
        # the decel parabola reverses (overshoot then back up) -> jerk.
        t_accel = duration * 0.25
        t_cruise = duration * 0.50
        n = int(np.ceil(duration / self.dt))
        times = np.linspace(0.0, duration, n + 1)

        # Build trajectory
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        for t in times:
            point = JointTrajectoryPoint()

            for j in range(6):
                s = start[j]
                e = end[j]
                distance = abs(deltas[j])
                direction = 1 if deltas[j] > 0 else -1

                if distance < 1e-6:
                    point.positions.append(s)
                    point.velocities.append(0.0)
                    continue

                v_max = distance / (0.75 * duration)
                accel = v_max / t_accel

                if t <= t_accel:
                    vel = accel * t
                    pos = s + direction * 0.5 * accel * t ** 2
                elif t <= t_accel + t_cruise:
                    t_c = t - t_accel
                    d_accel = 0.5 * accel * t_accel ** 2
                    vel = v_max
                    pos = s + direction * (d_accel + v_max * t_c)
                else:
                    t_d = t - t_accel - t_cruise
                    d_accel = 0.5 * v_max * t_accel
                    d_cruise = v_max * t_cruise
                    d_decel = v_max * t_d - 0.5 * accel * t_d ** 2
                    vel = v_max - accel * t_d
                    pos = s + direction * (d_accel + d_cruise + d_decel)

                point.positions.append(pos)
                point.velocities.append(direction * max(0, vel))

            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nsecs)
            traj_msg.points.append(point)

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
