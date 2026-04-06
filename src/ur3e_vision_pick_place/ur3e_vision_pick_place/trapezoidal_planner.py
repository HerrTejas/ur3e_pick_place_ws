#!/usr/bin/env python3
"""
Trapezoidal Planner - Coordinated Motion
Author: Tejas
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np


class TrapezoidalPlanner(Node):
    def __init__(self):
        super().__init__('trapezoidal_planner')
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Reference positions
        self.HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]
        
        # Parameters
        self.duration = 3.0
        self.dt = 0.1
        self.current_positions = None
        
        # Subscriber: Joint states only
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        
        # Subscriber: Manual position commands
        self.create_subscription(Float64MultiArray, '/cmd_joint_positions', self.cmd_cb, 10)
        
        # Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10
        )
    
    def joint_state_cb(self, msg):
        """Only store current joint positions."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        if len(positions) == 6:
            self.current_positions = [positions[name] for name in self.joint_names]
    
    def cmd_cb(self, msg):
        """Receive manual position command."""
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
                |     /           \
                |    /             \
                |___/               \\___
                0  T/4    3T/4      T
        """
        if self.current_positions is None:
            self.get_logger().error('No joint states yet!')
            return
        
        self.get_logger().info(f'Moving to: {[f"{x:.2f}" for x in target]}')
        
        # Time parameters
        t_accel = self.duration * 0.25
        t_cruise = self.duration * 0.50
        times = np.arange(0, self.duration + self.dt, self.dt)
        
        # Build trajectory
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        for t in times:
            point = JointTrajectoryPoint()
            
            for j in range(6):
                start = self.current_positions[j]
                end = target[j]
                distance = abs(end - start)
                direction = 1 if end > start else -1
                
                # No movement case
                if distance < 1e-6:
                    point.positions.append(start)
                    point.velocities.append(0.0)
                    continue
                
                # Calculate velocity to cover distance in given duration
                v_max = distance / (0.75 * self.duration)
                accel = v_max / t_accel
                
                if t <= t_accel:
                    # Phase 1: Accelerating
                    vel = accel * t
                    pos = start + direction * 0.5 * accel * t**2
                    
                elif t <= t_accel + t_cruise:
                    # Phase 2: Cruising
                    t_c = t - t_accel
                    d_accel = 0.5 * accel * t_accel**2
                    vel = v_max
                    pos = start + direction * (d_accel + v_max * t_c)
                    
                else:
                    # Phase 3: Decelerating
                    t_d = t - t_accel - t_cruise
                    d_accel = 0.5 * v_max * t_accel
                    d_cruise = v_max * t_cruise
                    d_decel = v_max * t_d - 0.5 * accel * t_d**2
                    vel = v_max - accel * t_d
                    pos = start + direction * (d_accel + d_cruise + d_decel)
                
                point.positions.append(pos)
                point.velocities.append(direction * max(0, vel))
            
            # Set timestamp
            secs = int(t)
            nsecs = int((t - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nsecs)
            traj_msg.points.append(point)
        
        # Publish
        self.traj_pub.publish(traj_msg)
        self.get_logger().info(f'Published {len(traj_msg.points)} points')


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalPlanner()
    
    # Wait for joint states
    node.get_logger().info('Waiting for joint states...')
    while node.current_positions is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
    
    node.get_logger().info(f'Current: {[f"{p:.2f}" for p in node.current_positions]}')
    
    # Default: Move to RED_BOX_GRASP
    node.get_logger().info('Default: Moving to RED_BOX_GRASP')
    node.move_to(node.RED_BOX_GRASP)
    
    # Keep running for manual commands
    node.get_logger().info('')
    node.get_logger().info('Listening for manual commands on /cmd_joint_positions')
    node.get_logger().info(f'HOME: {node.HOME}')
    node.get_logger().info(f'RED_BOX_GRASP: {node.RED_BOX_GRASP}')
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
