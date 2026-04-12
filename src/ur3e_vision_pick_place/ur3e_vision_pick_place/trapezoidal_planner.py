#!/usr/bin/env python3
"""
Trapezoidal Planner - Coordinated Joint-Space Motion

The TrajectoryProfile class at the top is pure math.
The Node class below is just ROS wiring.

Other nodes (like path_interpolation) can import the math directly:
    from ur3e_vision_pick_place.trapezoidal_planner import TrajectoryProfile

Author: Tejas
"""

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ══════════════════════════════════════════════════════════════════
#  Pure math — no ROS, importable by any node
# ══════════════════════════════════════════════════════════════════

class TrajectoryProfile:
    """Trapezoidal velocity profile generator."""

    def trapezoid_time_scaled(self, L, vmax, amax, dt):
        """
        Generate trapezoidal profile for distance L.

        Returns:
            t_array, s_array, t_total
        """
        if L < 1e-8:
            return np.array([0.0]), np.array([0.0]), 0.0

        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc ** 2

        if 2 * d_acc > L:
            t_acc = np.sqrt(L / amax)
            t_flat = 0.0
            t_total = 2 * t_acc
        else:
            d_flat = L - 2 * d_acc
            t_flat = d_flat / vmax
            t_total = 2 * t_acc + t_flat

        t_list = []
        s_list = []
        t = 0.0

        while t <= t_total + 1e-9:
            if t < t_acc:
                s = 0.5 * amax * t ** 2
            elif t < t_acc + t_flat:
                s = d_acc + vmax * (t - t_acc)
            else:
                t_dec = t - (t_acc + t_flat)
                s = d_acc + vmax * t_flat + vmax * t_dec - 0.5 * amax * t_dec ** 2

            t_list.append(t)
            s_list.append(min(s, L))
            t += dt

        return np.array(t_list), np.array(s_list), t_total

    def trapezoid_multi(self, L_array, vmax, amax, dt):
        """
        Synchronized trapezoid for multiple dimensions.
        All share the same time base, scaled by their distance.

        Returns:
            t_array, s_scaled (one row per dimension), t_total
        """
        L_array = np.array(L_array)
        L_max = np.max(L_array)

        t_list, s_base, T = self.trapezoid_time_scaled(L_max, vmax, amax, dt)

        s_scaled = []
        for L in L_array:
            if L > 1e-8:
                s_scaled.append(s_base * (L / L_max))
            else:
                s_scaled.append(np.zeros_like(s_base))

        return t_list, np.array(s_scaled), T


# ══════════════════════════════════════════════════════════════════
#  ROS Node — joint-space planner, calls TrajectoryProfile above
# ══════════════════════════════════════════════════════════════════

class TrapezoidalPlanner(Node):
    def __init__(self):
        super().__init__('trapezoidal_planner')

        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]

        self.duration = 3.0
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

    def move_to(self, target):
        """
        Compute and execute trapezoidal trajectory to target position.

        Trapezoidal Profile (25% accel, 50% cruise, 25% decel):

            Velocity
                ^
           Vmax |      ___________
                |     /           \\
                |    /             \\
                |___/               \\___
                0  T/4    3T/4      T
        """
        if self.current_positions is None:
            self.get_logger().error('No joint states yet!')
            return

        self.get_logger().info(f'Moving to: {[f"{x:.2f}" for x in target]}')

        t_accel = self.duration * 0.25
        t_cruise = self.duration * 0.50
        times = np.arange(0, self.duration + self.dt, self.dt)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        for t in times:
            point = JointTrajectoryPoint()

            for j in range(6):
                start = self.current_positions[j]
                end = target[j]
                distance = abs(end - start)
                direction = 1 if end > start else -1

                if distance < 1e-6:
                    point.positions.append(start)
                    point.velocities.append(0.0)
                    continue

                v_max = distance / (0.75 * self.duration)
                accel = v_max / t_accel

                if t <= t_accel:
                    vel = accel * t
                    pos = start + direction * 0.5 * accel * t ** 2
                elif t <= t_accel + t_cruise:
                    t_c = t - t_accel
                    d_accel = 0.5 * accel * t_accel ** 2
                    vel = v_max
                    pos = start + direction * (d_accel + v_max * t_c)
                else:
                    t_d = t - t_accel - t_cruise
                    d_accel = 0.5 * v_max * t_accel
                    d_cruise = v_max * t_cruise
                    d_decel = v_max * t_d - 0.5 * accel * t_d ** 2
                    vel = v_max - accel * t_d
                    pos = start + direction * (d_accel + d_cruise + d_decel)

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