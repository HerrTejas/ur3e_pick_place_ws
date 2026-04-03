#!/usr/bin/env python3
"""
Pick and Place with 3-DOF IK
- Color-specific wrist orientations
- Color-specific offsets (calculated from your working configs)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from builtin_interfaces.msg import Duration
import numpy as np
import pinocchio as pin
import json


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('ik_pick_place')
        
        self.get_logger().info('Initializing 3-DOF IK Pick and Place...')
        
        self.current_joints = None
        self.detected_objects = {}
        
        self.setup_pinocchio()
        
        # Action clients
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory'
        )
        
        self.arm_joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.gripper_joints = ['rh_r1_joint']
        
        self.GRIPPER_OPEN = [0.0]
        self.GRIPPER_CLOSE = [0.7]
        self.HOME = [0, -1.57, 0, -1.57, 0, 0]
        
        # COLOR-SPECIFIC configurations
        # Each color has its own wrist orientation and offset
        # Offset = tool0_position - object_world_position (from your working configs)
        self.color_configs = {
            'red': {
                'wrist': [-1.8, -1.61, -0.3],
                'offset': [0.002, 0.084, -0.269],  # From analysis
                'lift_wrist': [-1.9, -1.57, -0.3],  # From your lift config
            },
            'green': {
                'wrist': [-2.068, -1.5, 0.0],
                'offset': [-0.086, 0.064, -0.243],  # From analysis
                'lift_wrist': [-2.068, -1.5, 0.0],
            },
            'blue': {
                'wrist': [-1.9, -1.65, -0.55],
                'offset': [0.073, 0.070, -0.274],  # From analysis
                'lift_wrist': [-2.1, -1.65, -0.7],  # From your lift config
            },
        }
        
        # Place configurations (hardcoded - they work)
        self.place_configs = {
            'red': {
                'place_pre':  [0.3, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [0.3, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
            'green': {
                'place_pre':  [0.5, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [0.5, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
            'blue': {
                'place_pre':  [1.2, -0.94, 0.8, -1.9, -1.57, -0.3],
                'place_down': [1.2, -0.94, 1.35, -1.9, -1.57, -0.3],
            },
        }
        
        # Heights
        self.pre_grasp_height = 0.06
        self.lift_height = 0.12
        
        # Joint limits for first 3 joints
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, np.pi])
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.detection_sub = self.create_subscription(
            String, '/detected_objects_3d', self.detection_callback, 10)
        
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Pick and Place ready!')
        
    def setup_pinocchio(self):
        urdf_path = "/tmp/ur3e.urdf"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("tool0")
        
    def joint_state_callback(self, msg):
        joints = {}
        for i, name in enumerate(msg.name):
            joints[name] = msg.position[i]
        self.current_joints = joints
        
    def detection_callback(self, msg):
        try:
            self.detected_objects = json.loads(msg.data)
        except:
            pass
    
    def get_ee_position(self, q6):
        q_full = np.zeros(self.model.nq)
        q_full[:6] = q6
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id].translation.copy()
    
    def compute_ik_3dof(self, target_tool0, initial_3joints, fixed_wrist):
        """3-DOF IK: only adjust first 3 joints, wrist stays fixed"""
        q_full = np.zeros(self.model.nq)
        q_full[:3] = initial_3joints
        q_full[3:6] = fixed_wrist
        
        alpha = 0.5
        damping = 0.01
        best_q3 = q_full[:3].copy()
        best_error = float('inf')
        
        for i in range(1000):
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pos = self.data.oMf[self.ee_frame_id].translation
            error = target_tool0 - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < best_error:
                best_error = error_norm
                best_q3 = q_full[:3].copy()
            
            if error_norm < 0.003:
                self.get_logger().info(f'IK converged: {i} iters, err={error_norm*1000:.1f}mm')
                return list(q_full[:3]) + list(fixed_wrist)
            
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q_full,
                self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            J = J_full[:3, :3]
            
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            q_full[:3] = np.clip(q_full[:3] + alpha * dq, 
                                  self.joint_limits_lower, 
                                  self.joint_limits_upper)
        
        if best_error < 0.01:
            self.get_logger().warn(f'IK partial: err={best_error*1000:.1f}mm')
            return list(best_q3) + list(fixed_wrist)
        
        self.get_logger().error(f'IK FAILED: err={best_error*1000:.1f}mm')
        return None
        
    def move_arm(self, positions, duration=2.0):
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints
        
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        trajectory.points = [point]
        goal.trajectory = trajectory
        
        self.get_logger().info(f'Moving: {[f"{p:.2f}" for p in positions]}')
        
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True
        
    def move_gripper(self, positions, duration=0.5):
        goal = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        trajectory.points = [point]
        goal.trajectory = trajectory
        
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True
    
    def pick_object(self, color_name):
        if color_name not in self.detected_objects:
            self.get_logger().error(f'{color_name} not detected!')
            return False
        
        config = self.color_configs[color_name]
        det = self.detected_objects[color_name]
        world_pos = np.array([det['x'], det['y'], det['z']])
        
        self.get_logger().info(f'=== PICKING {color_name.upper()} ===')
        self.get_logger().info(f'World: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]')
        
        # Get color-specific offset and wrist
        offset = np.array(config['offset'])
        wrist = config['wrist']
        lift_wrist = config['lift_wrist']
        
        self.get_logger().info(f'Offset: {offset}')
        self.get_logger().info(f'Wrist: {wrist}')
        
        # Compute tool0 targets using color-specific offset
        grasp_tool0 = world_pos + offset
        pre_grasp_tool0 = grasp_tool0.copy()
        pre_grasp_tool0[2] += self.pre_grasp_height
        lift_tool0 = grasp_tool0.copy()
        lift_tool0[2] += self.lift_height
        
        self.get_logger().info(f'Grasp tool0: [{grasp_tool0[0]:.4f}, {grasp_tool0[1]:.4f}, {grasp_tool0[2]:.4f}]')
        
        # Initial guess - estimate shoulder_pan from X
        shoulder_pan = 1.255 - grasp_tool0[0] * 2.0
        initial_3 = [shoulder_pan, -1.0, 1.4]
        
        # Compute IK with color-specific wrist
        pre_grasp_joints = self.compute_ik_3dof(pre_grasp_tool0, initial_3, wrist)
        if pre_grasp_joints is None:
            return False
        
        grasp_joints = self.compute_ik_3dof(grasp_tool0, pre_grasp_joints[:3], wrist)
        if grasp_joints is None:
            return False
        
        lift_joints = self.compute_ik_3dof(lift_tool0, grasp_joints[:3], lift_wrist)
        if lift_joints is None:
            return False
        
        # Execute
        self.move_gripper(self.GRIPPER_OPEN)
        self.move_arm(pre_grasp_joints)
        self.move_arm(grasp_joints)
        self.move_gripper(self.GRIPPER_CLOSE)
        self.move_arm(lift_joints)
        
        return True
    
    def place_object(self, color_name):
        config = self.place_configs[color_name]
        
        self.get_logger().info(f'=== PLACING {color_name.upper()} ===')
        self.move_arm(config['place_pre'])
        self.move_arm(config['place_down'])
        self.move_gripper(self.GRIPPER_OPEN)
        self.move_arm(config['place_pre'])
        
        return True
    
    def execute_pick_place(self, color_name):
        self.get_logger().info(f'========== {color_name.upper()} ==========')
        
        self.move_arm(self.HOME)
        self.move_gripper(self.GRIPPER_OPEN)
        
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if color_name not in self.detected_objects:
            self.get_logger().error(f'{color_name} not detected!')
            return False
        
        if not self.pick_object(color_name):
            return False
        
        if not self.place_object(color_name):
            return False
        
        self.move_arm(self.HOME)
        self.get_logger().info(f'========== {color_name.upper()} DONE ==========')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()
    
    try:
        node.get_logger().info('Waiting for detections...')
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
        
        for color in ['red', 'green', 'blue']:
            node.execute_pick_place(color)
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
