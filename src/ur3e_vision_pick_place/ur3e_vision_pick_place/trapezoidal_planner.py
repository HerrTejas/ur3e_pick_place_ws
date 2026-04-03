#!/usr/bin/env python3
"""
Trapezoidal Velocity Profile Trajectory Planner

This node demonstrates joint space motion planning using trapezoidal
velocity profiles with hard-coded velocity and acceleration limits.

Trapezoidal Profile:
    
    Velocity
        ^
        |      ___________
   Vmax |     /           \
        |    /             \
        |   /               \
        |  /                 \
        |_/___________________\__> Time
        0  t1      t2       t3
        
        Phase 1: Acceleration (0 to t1)
        Phase 2: Constant Velocity (t1 to t2)  
        Phase 3: Deceleration (t2 to t3)

Author: Tejas (Learning Exercise)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import math


class TrapezoidalPlanner(Node):
    def __init__(self):
        super().__init__('trapezoidal_planner')
        
        # ==========================================
        # PARAMETERS (Hard-coded for learning)
        # ==========================================
        
        # Maximum velocity for each joint (rad/s)
        self.max_velocity = 0.5
        
        # Maximum acceleration for each joint (rad/s^2)
        self.max_acceleration = 0.5
        
        # Time step for trajectory points (seconds)
        self.dt = 0.1
        
        # Joint names (UR3e arm)
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # ==========================================
        # KNOWN POSITIONS
        # ==========================================
        self.HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]
        
        # ==========================================
        # STATE VARIABLES
        # ==========================================
        
        # Current joint positions (from /joint_states)
        self.current_positions = None
        
        # ==========================================
        # ROS2 SETUP
        # ==========================================
        
        # Subscriber: Get current joint positions
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Publisher: Send trajectory to controller
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        self.get_logger().info('='*50)
        self.get_logger().info('TRAPEZOIDAL PLANNER')
        self.get_logger().info('='*50)
        self.get_logger().info(f'Max Velocity: {self.max_velocity} rad/s')
        self.get_logger().info(f'Max Acceleration: {self.max_acceleration} rad/s^2')
        self.get_logger().info(f'Time Step: {self.dt} s')
        
    def joint_state_callback(self, msg):
        """
        Callback: Store current joint positions
        """
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        
        if len(positions) == len(self.joint_names):
            self.current_positions = [positions[name] for name in self.joint_names]
    
    def compute_trapezoidal_profile(self, start_pos, end_pos, max_vel, max_acc):
        """
        Compute trapezoidal velocity profile for a single joint.
        
        Returns: List of (time, position, velocity) tuples
        """
        
        # Calculate distance to travel
        distance = abs(end_pos - start_pos)
        direction = 1 if end_pos > start_pos else -1
        
        # Handle zero distance case
        if distance < 1e-6:
            return [(0.0, start_pos, 0.0)]
        
        # Time to accelerate to max velocity
        t_accel = max_vel / max_acc
        
        # Distance covered during acceleration (and deceleration)
        d_accel = 0.5 * max_acc * t_accel ** 2
        
        # Check if we can reach max velocity (trapezoidal) or not (triangular)
        if 2 * d_accel <= distance:
            # CASE 1: Trapezoidal profile
            t1 = t_accel
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / max_vel
            t2 = t1 + t_cruise
            t3 = t2 + t_accel
            v_peak = max_vel
            profile_type = "TRAPEZOIDAL"
        else:
            # CASE 2: Triangular profile
            t1 = math.sqrt(distance / max_acc)
            t2 = t1
            t3 = 2 * t1
            v_peak = max_acc * t1
            profile_type = "TRIANGULAR"
        
        # Generate trajectory points
        trajectory = []
        t = 0.0
        
        while t <= t3:
            if t <= t1:
                # Phase 1: Acceleration
                vel = max_acc * t
                pos = start_pos + direction * (0.5 * max_acc * t ** 2)
            elif t <= t2:
                # Phase 2: Constant velocity
                d1 = 0.5 * max_acc * t1 ** 2
                pos = start_pos + direction * (d1 + v_peak * (t - t1))
                vel = v_peak
            else:
                # Phase 3: Deceleration
                t_decel = t - t2
                d1 = 0.5 * max_acc * t1 ** 2
                d2 = v_peak * (t2 - t1) if t2 > t1 else 0
                d3 = v_peak * t_decel - 0.5 * max_acc * t_decel ** 2
                pos = start_pos + direction * (d1 + d2 + d3)
                vel = v_peak - max_acc * t_decel
            
            vel = max(0, vel) * direction
            trajectory.append((t, pos, vel))
            t += self.dt
        
        # Ensure we end exactly at the target
        trajectory.append((t3, end_pos, 0.0))
        
        return trajectory, profile_type, t3, v_peak
    
    def plan_trajectory(self, target_positions):
        """
        Plan trajectory for all joints to reach target positions.
        """
        
        if self.current_positions is None:
            self.get_logger().error('No current joint positions!')
            return None
        
        self.get_logger().info('\n' + '='*50)
        self.get_logger().info('PLANNING TRAJECTORY')
        self.get_logger().info('='*50)
        
        # Print start and target
        self.get_logger().info('\nStart positions:')
        for i, name in enumerate(self.joint_names):
            self.get_logger().info(f'  {name}: {self.current_positions[i]:.3f} rad')
        
        self.get_logger().info('\nTarget positions:')
        for i, name in enumerate(self.joint_names):
            self.get_logger().info(f'  {name}: {target_positions[i]:.3f} rad')
        
        # Compute profile for each joint
        joint_profiles = []
        max_time = 0.0
        
        self.get_logger().info('\n' + '-'*50)
        self.get_logger().info('JOINT PROFILES:')
        self.get_logger().info('-'*50)
        
        for i, (start, end) in enumerate(zip(self.current_positions, target_positions)):
            delta = end - start
            profile, profile_type, total_time, peak_vel = self.compute_trapezoidal_profile(
                start, end, 
                self.max_velocity, 
                self.max_acceleration
            )
            joint_profiles.append(profile)
            
            self.get_logger().info(f'\n{self.joint_names[i]}:')
            self.get_logger().info(f'  Delta: {delta:.3f} rad')
            self.get_logger().info(f'  Profile: {profile_type}')
            self.get_logger().info(f'  Peak Velocity: {peak_vel:.3f} rad/s')
            self.get_logger().info(f'  Duration: {total_time:.2f} s')
            
            if total_time > max_time:
                max_time = total_time
        
        self.get_logger().info(f'\nTotal trajectory time: {max_time:.2f} s')
        self.get_logger().info(f'Number of points: {int(max_time / self.dt) + 1}')
        
        # Create unified time stamps
        time_stamps = np.arange(0, max_time + self.dt, self.dt)
        
        # Interpolate each joint's profile to unified time stamps
        trajectory_points = []
        
        for t in time_stamps:
            positions = []
            velocities = []
            
            for profile in joint_profiles:
                pos, vel = self.interpolate_profile(profile, t)
                positions.append(pos)
                velocities.append(vel)
            
            trajectory_points.append({
                'time': t,
                'positions': positions,
                'velocities': velocities
            })
        
        return trajectory_points
    
    def interpolate_profile(self, profile, t):
        """
        Interpolate position and velocity at time t from a profile.
        """
        if t >= profile[-1][0]:
            return profile[-1][1], profile[-1][2]
        
        for i in range(len(profile) - 1):
            t0, p0, v0 = profile[i]
            t1, p1, v1 = profile[i + 1]
            
            if t0 <= t <= t1:
                ratio = (t - t0) / (t1 - t0) if t1 > t0 else 0
                pos = p0 + ratio * (p1 - p0)
                vel = v0 + ratio * (v1 - v0)
                return pos, vel
        
        return profile[0][1], profile[0][2]
    
    def execute_trajectory(self, trajectory_points):
        """
        Send planned trajectory to robot controller.
        """
        if trajectory_points is None:
            return False
        
        self.get_logger().info('\n' + '='*50)
        self.get_logger().info('EXECUTING TRAJECTORY')
        self.get_logger().info('='*50)
        
        # Create JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        # Add all trajectory points
        for point_data in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']
            point.velocities = point_data['velocities']
            
            secs = int(point_data['time'])
            nsecs = int((point_data['time'] - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nsecs)
            
            traj_msg.points.append(point)
        
        # Publish trajectory
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(f'Trajectory published! ({len(trajectory_points)} points)')
        
        return True
    
    def move_to(self, target_positions):
        """
        Main function: Plan and execute trajectory to target positions.
        """
        trajectory = self.plan_trajectory(target_positions)
        return self.execute_trajectory(trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalPlanner()
    
    # Wait for joint states
    node.get_logger().info('\nWaiting for joint states...')
    while node.current_positions is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)
    
    node.get_logger().info('Joint states received!')
    
    try:
        import time
        
        # Move from current position to RED BOX GRASP
        input("\nPress Enter to move to RED_BOX_GRASP position...")
        node.move_to(node.RED_BOX_GRASP)
        
        # Wait for motion to complete
        time.sleep(6)
        
        node.get_logger().info('\n' + '='*50)
        node.get_logger().info('MOTION COMPLETE!')
        node.get_logger().info('='*50)
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
