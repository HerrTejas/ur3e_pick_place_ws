#!/usr/bin/env python3
"""
Forward Kinematics - Pure Python Implementation

This implements FK using Denavit-Hartenberg (DH) convention.
No external robotics libraries - just numpy for matrix operations.

=============================================================================
DENAVIT-HARTENBERG (DH) PARAMETERS
=============================================================================

Each joint is described by 4 parameters:
    θ (theta) - Joint angle (rotation about Z-axis) - THIS IS WHAT CHANGES
    d         - Link offset (translation along Z-axis)
    a         - Link length (translation along X-axis)
    α (alpha) - Link twist (rotation about X-axis)

For UR3e robot:
    Joint    θ        d (m)      a (m)       α (rad)
    ─────────────────────────────────────────────────
    1        θ1       0.15185    0           π/2
    2        θ2       0          -0.24355    0
    3        θ3       0          -0.2132     0
    4        θ4       0.13105    0           π/2
    5        θ5       0.08535    0           -π/2
    6        θ6       0.0921     0           0

Author: Tejas (Learning Exercise)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np


class ForwardKinematicsPure(Node):
    def __init__(self):
        super().__init__('forward_kinematics_pure')
        
        # ==========================================
        # UR3e DH PARAMETERS
        # ==========================================
        # Format: [d, a, alpha] for each joint
        # theta is the joint variable (input)
        
        self.dh_params = [
            # [d (m),     a (m),      alpha (rad)]
            [0.15185,    0,          np.pi/2 ],   # Joint 1
            [0,          -0.24355,   0       ],   # Joint 2
            [0,          -0.2132,    0       ],   # Joint 3
            [0.13105,    0,          np.pi/2 ],   # Joint 4
            [0.08535,    0,          -np.pi/2],   # Joint 5
            [0.0921,     0,          0       ],   # Joint 6
        ]
        
        # Joint names
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # ==========================================
        # BASE FRAME CORRECTION
        # ==========================================
        # URDF base frame differs from DH convention
        # We need to rotate 180° around Z-axis
        
        self.T_base_correction = np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])
        
        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.current_positions = None
        
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('FORWARD KINEMATICS - PURE PYTHON')
        self.get_logger().info('='*60)
        self.get_logger().info('Using DH parameters for UR3e')
        self.get_logger().info('With base frame correction for URDF compatibility')
        self.get_logger().info('')
    
    def dh_matrix(self, theta, d, a, alpha):
        """
        Compute the DH transformation matrix for one joint.
        
        The DH matrix is:
        T = Rz(θ) × Tz(d) × Tx(a) × Rx(α)
        
        Which expands to:
        ┌                                                    ┐
        │ cos(θ)   -sin(θ)cos(α)    sin(θ)sin(α)    a·cos(θ) │
        │ sin(θ)    cos(θ)cos(α)   -cos(θ)sin(α)    a·sin(θ) │
        │ 0         sin(α)          cos(α)          d        │
        │ 0         0               0               1        │
        └                                                    ┘
        """
        ct = np.cos(theta)  # cos(theta)
        st = np.sin(theta)  # sin(theta)
        ca = np.cos(alpha)  # cos(alpha)
        sa = np.sin(alpha)  # sin(alpha)
        
        T = np.array([
            [ct,    -st*ca,     st*sa,      a*ct],
            [st,     ct*ca,    -ct*sa,      a*st],
            [0,      sa,        ca,         d   ],
            [0,      0,         0,          1   ]
        ])
        
        return T
    
    def compute_fk(self, joint_angles, apply_correction=True):
        """
        Compute Forward Kinematics.
        
        Args:
            joint_angles: List of 6 joint angles [θ1, θ2, θ3, θ4, θ5, θ6] in radians
            apply_correction: Apply base frame correction to match URDF
            
        Returns:
            4x4 transformation matrix of end-effector pose
        """
        # Start with identity matrix (no transformation)
        T_total = np.eye(4)
        
        # Multiply transformation matrices for each joint
        for i in range(6):
            theta = joint_angles[i]
            d = self.dh_params[i][0]
            a = self.dh_params[i][1]
            alpha = self.dh_params[i][2]
            
            # Compute this joint's transformation
            T_i = self.dh_matrix(theta, d, a, alpha)
            
            # Multiply to get cumulative transformation
            T_total = T_total @ T_i
        
        # Apply base frame correction to match URDF/Pinocchio
        if apply_correction:
            T_total = self.T_base_correction @ T_total
        
        return T_total
    
    def rotation_matrix_to_rpy(self, R):
        """
        Convert rotation matrix to Roll-Pitch-Yaw angles.
        """
        if abs(R[2, 0]) >= 1:
            yaw = 0
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        
        return roll, pitch, yaw
    
    def rotation_matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion [x, y, z, w].
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return x, y, z, w
    
    def print_fk(self, joint_angles):
        """
        Compute and print FK result.
        """
        # Compute FK with correction
        T = self.compute_fk(joint_angles, apply_correction=True)
        
        # Extract position
        position = T[:3, 3]
        
        # Extract rotation matrix
        rotation = T[:3, :3]
        
        # Convert to RPY and quaternion
        roll, pitch, yaw = self.rotation_matrix_to_rpy(rotation)
        qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rotation)
        
        # Print results
        print('\n' + '='*60)
        print('FORWARD KINEMATICS RESULT (Pure Python + Correction)')
        print('='*60)
        
        print('\nINPUT - Joint Angles:')
        for i, name in enumerate(self.joint_names):
            print(f'  {name}: {joint_angles[i]:.4f} rad ({np.degrees(joint_angles[i]):.2f}°)')
        
        print('\n' + '-'*60)
        print('TRANSFORMATION MATRIX:')
        print('-'*60)
        print(f'\n  ┌{T[0,0]:8.4f}  {T[0,1]:8.4f}  {T[0,2]:8.4f}  {T[0,3]:8.4f} ┐')
        print(f'  │{T[1,0]:8.4f}  {T[1,1]:8.4f}  {T[1,2]:8.4f}  {T[1,3]:8.4f} │')
        print(f'  │{T[2,0]:8.4f}  {T[2,1]:8.4f}  {T[2,2]:8.4f}  {T[2,3]:8.4f} │')
        print(f'  └{T[3,0]:8.4f}  {T[3,1]:8.4f}  {T[3,2]:8.4f}  {T[3,3]:8.4f} ┘')
        print(f'\n       ↑ Rotation (3x3)          ↑ Position')
        
        print('\n' + '-'*60)
        print('OUTPUT - End Effector Pose:')
        print('-'*60)
        
        print(f'\n  Position (meters):')
        print(f'    X: {position[0]:8.4f} m')
        print(f'    Y: {position[1]:8.4f} m')
        print(f'    Z: {position[2]:8.4f} m')
        
        print(f'\n  Orientation (RPY in degrees):')
        print(f'    Roll:  {np.degrees(roll):8.2f}°')
        print(f'    Pitch: {np.degrees(pitch):8.2f}°')
        print(f'    Yaw:   {np.degrees(yaw):8.2f}°')
        
        print(f'\n  Orientation (Quaternion):')
        print(f'    x: {qx:8.4f}')
        print(f'    y: {qy:8.4f}')
        print(f'    z: {qz:8.4f}')
        print(f'    w: {qw:8.4f}')
        
        print('\n' + '='*60)
        
        return T
    
    def print_step_by_step(self, joint_angles):
        """
        Show FK calculation step by step.
        """
        print('\n' + '#'*60)
        print('STEP-BY-STEP FK CALCULATION')
        print('#'*60)
        
        print('\nDH Parameters for UR3e:')
        print('─'*50)
        print(f'{"Joint":<8} {"θ (rad)":<10} {"d (m)":<10} {"a (m)":<10} {"α (rad)":<10}')
        print('─'*50)
        for i in range(6):
            theta = joint_angles[i]
            d = self.dh_params[i][0]
            a = self.dh_params[i][1]
            alpha = self.dh_params[i][2]
            print(f'{i+1:<8} {theta:<10.4f} {d:<10.4f} {a:<10.4f} {alpha:<10.4f}')
        
        print('\n\nComputing transformation matrices...\n')
        
        T_total = np.eye(4)
        
        for i in range(6):
            theta = joint_angles[i]
            d = self.dh_params[i][0]
            a = self.dh_params[i][1]
            alpha = self.dh_params[i][2]
            
            T_i = self.dh_matrix(theta, d, a, alpha)
            T_total = T_total @ T_i
            
            pos = T_total[:3, 3]
            print(f'After Joint {i+1}: Position = ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}) [DH frame]')
        
        # Apply correction
        T_corrected = self.T_base_correction @ T_total
        pos_corrected = T_corrected[:3, 3]
        
        print(f'\nAfter base correction: ({pos_corrected[0]:7.4f}, {pos_corrected[1]:7.4f}, {pos_corrected[2]:7.4f}) [URDF frame]')
        
        print('\n' + '-'*60)
        print('WHY BASE CORRECTION?')
        print('-'*60)
        print('''
The URDF and standard DH convention have different base orientations.
URDF base frame is rotated 180° around Z compared to DH convention.

Base correction matrix (180° rotation around Z):
┌ -1   0   0   0 ┐
│  0  -1   0   0 │
│  0   0   1   0 │
└  0   0   0   1 ┘

This flips X and Y to match Pinocchio/URDF output.
''')
        
        return T_corrected
    
    def compare_with_pinocchio(self, joint_angles):
        """
        Compare pure Python FK with Pinocchio.
        """
        # Pure Python (with correction)
        T_pure = self.compute_fk(joint_angles, apply_correction=True)
        pos_pure = T_pure[:3, 3]
        
        print('\n' + '='*60)
        print('COMPARISON: Pure Python vs Pinocchio')
        print('='*60)
        print(f'\nJoint angles: {[f"{a:.2f}" for a in joint_angles]}')
        print('\n' + '-'*40)
        print(f'{"Method":<20} {"X":>10} {"Y":>10} {"Z":>10}')
        print('-'*40)
        print(f'{"Pure Python":<20} {pos_pure[0]:10.4f} {pos_pure[1]:10.4f} {pos_pure[2]:10.4f}')
        print(f'{"(Run Pinocchio FK to compare)":<20}')
        print('-'*40)
        
        return T_pure
    
    def joint_state_callback(self, msg):
        """Store current joint positions."""
        positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                positions[name] = msg.position[i]
        
        if len(positions) == len(self.joint_names):
            self.current_positions = [positions[name] for name in self.joint_names]


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematicsPure()
    
    # Test positions
    HOME = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
    RED_BOX_GRASP = [1.255, -0.98, 1.4, -1.8, -1.61, -0.3]
    
    print('\n' + '#'*60)
    print('FORWARD KINEMATICS - PURE PYTHON IMPLEMENTATION')
    print('#'*60)
    print('\nNo external robotics libraries - just numpy!')
    print('Using Denavit-Hartenberg convention + base frame correction.')
    
    # Show step-by-step for HOME
    print('\n>>> Step-by-step FK for HOME position...')
    node.print_step_by_step(HOME)
    
    # Show full result for RED_BOX_GRASP
    print('\n>>> Full FK result for RED_BOX_GRASP position...')
    node.print_fk(RED_BOX_GRASP)
    
    # Interactive mode
    print('\n' + '#'*60)
    print('INTERACTIVE MODE')
    print('#'*60)
    print('\nEnter joint angles (space separated, in radians)')
    print('Example: 0 -1.57 0 -1.57 0 0')
    print('Commands:')
    print('  "s <angles>" - Step-by-step calculation')
    print('  "c <angles>" - Compare with Pinocchio')
    print('  "q"          - Quit')
    print('')
    
    try:
        while rclpy.ok():
            user_input = input('Enter joint angles: ').strip()
            
            if user_input.lower() == 'q':
                break
            
            step_by_step = False
            compare = False
            
            if user_input.lower().startswith('s '):
                step_by_step = True
                user_input = user_input[2:]
            elif user_input.lower().startswith('c '):
                compare = True
                user_input = user_input[2:]
            
            try:
                angles = [float(x) for x in user_input.split()]
                if len(angles) != 6:
                    print('Error: Enter exactly 6 values!')
                    continue
                
                if step_by_step:
                    node.print_step_by_step(angles)
                elif compare:
                    node.compare_with_pinocchio(angles)
                else:
                    node.print_fk(angles)
                
            except ValueError:
                print('Error: Invalid input!')
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
