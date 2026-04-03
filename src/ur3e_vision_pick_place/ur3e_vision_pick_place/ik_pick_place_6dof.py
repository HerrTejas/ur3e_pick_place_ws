#!/usr/bin/env python3
"""
6-DOF IK - Position only, with safe transit and stable placing
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


class PickAndPlace6DOF(Node):
    def __init__(self):
        super().__init__('ik_pick_place_6dof')
        
        self.get_logger().info('Initializing 6-DOF IK...')
        
        self.current_joints = None
        self.detected_objects = {}
        
        self.setup_pinocchio()
        self.setup_references()
        
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
        
        # Heights
        self.PRE_GRASP_HEIGHT = 0.03
        self.LIFT_HEIGHT = 0.20
        self.TRANSIT_HEIGHT = 0.20
        self.PLACE_RELEASE_HEIGHT = 0.02  # Release slightly above surface
        
        # Place positions
        self.PLACE_POSITIONS = {
            'red':   [-0.22, 0.22, 0.43],
            'green': [0.0, 0.22, 0.43],
            'blue':  [0.22, 0.22, 0.43],
        }
        
        # Joint limits
        self.joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
        self.joint_limits_upper = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.detection_sub = self.create_subscription(
            String, '/detected_objects_3d', self.detection_callback, 10)
        
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('6-DOF IK ready!')
        
    def setup_pinocchio(self):
        urdf_path = "/tmp/ur3e.urdf"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("tool0")
        
    def setup_references(self):
        """Setup reference configs for each color"""
        self.refs = {
            'red': {
                'joints': [1.255, -0.98, 1.4, -1.8, -1.61, -0.3],
                'object': np.array([0.0001, 0.333, 0.43])
            },
            'green': {
                'joints': [1.6, -1.06, 1.5, -2.068, -1.5, 0],
                'object': np.array([-0.0629, 0.3344, 0.4301])
            },
            'blue': {
                'joints': [0.95, -0.95, 1.4, -1.9, -1.65, -0.55],
                'object': np.array([0.0631, 0.333, 0.4299])
            },
        }
        
        for color, ref in self.refs.items():
            q = np.zeros(self.model.nq)
            q[:6] = ref['joints']
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            tool0_pos = self.data.oMf[self.ee_frame_id].translation.copy()
            ref['tool0'] = tool0_pos
            ref['offset'] = tool0_pos - ref['object']
            self.get_logger().info(f'{color} offset: [{ref["offset"][0]:.4f}, {ref["offset"][1]:.4f}, {ref["offset"][2]:.4f}]')
        
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
    
    def compute_ik_position(self, target_pos, q_init, max_iter=2000):
        """Position-only IK"""
        q = np.zeros(self.model.nq)
        q[:6] = q_init[:6]
        
        alpha = 0.5
        damping = 0.01
        
        best_q = q[:6].copy()
        best_error = float('inf')
        
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pos = self.data.oMf[self.ee_frame_id].translation
            
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < best_error:
                best_error = error_norm
                best_q = q[:6].copy()
            
            if error_norm < 0.003:
                self.get_logger().info(f'IK OK: {i} iters, err={error_norm*1000:.1f}mm')
                return list(q[:6])
            
            J = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :6]
            
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            q[:6] = np.clip(q[:6] + alpha * dq, 
                           self.joint_limits_lower, 
                           self.joint_limits_upper)
        
        if best_error < 0.01:
            self.get_logger().warn(f'IK partial: err={best_error*1000:.1f}mm')
            return list(best_q)
        
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
        """Pick using position IK"""
        if color_name not in self.detected_objects:
            self.get_logger().error(f'{color_name} not detected!')
            return False
        
        det = self.detected_objects[color_name]
        object_pos = np.array([det['x'], det['y'], det['z']])
        
        self.get_logger().info(f'=== PICKING {color_name.upper()} ===')
        self.get_logger().info(f'Object: [{object_pos[0]:.4f}, {object_pos[1]:.4f}, {object_pos[2]:.4f}]')
        
        ref = self.refs[color_name]
        offset = ref['offset']
        q_init = ref['joints']
        
        grasp_tool0 = object_pos + offset
        pre_grasp_tool0 = grasp_tool0.copy()
        pre_grasp_tool0[2] += self.PRE_GRASP_HEIGHT
        lift_tool0 = grasp_tool0.copy()
        lift_tool0[2] += self.LIFT_HEIGHT
        
        pre_grasp_joints = self.compute_ik_position(pre_grasp_tool0, q_init)
        if pre_grasp_joints is None:
            return False
        
        grasp_joints = self.compute_ik_position(grasp_tool0, pre_grasp_joints)
        if grasp_joints is None:
            return False
        
        lift_joints = self.compute_ik_position(lift_tool0, grasp_joints)
        if lift_joints is None:
            return False
        
        self.move_gripper(self.GRIPPER_OPEN)
        self.move_arm(pre_grasp_joints)
        self.move_arm(grasp_joints)
        self.move_gripper(self.GRIPPER_CLOSE)
        self.move_arm(lift_joints)
        
        self.last_lift_joints = lift_joints
        self.current_color = color_name
        
        return True
    
    def place_object(self, color_name):
        """Place using same color's reference for stable orientation"""
        place_pos = np.array(self.PLACE_POSITIONS[color_name])
        
        self.get_logger().info(f'=== PLACING {color_name.upper()} ===')
        
        # Use SAME color's offset for placing (maintains stable orientation)
        ref = self.refs[color_name]
        offset = ref['offset']
        
        # Place targets
        place_tool0 = place_pos + offset
        
        # Release slightly above to let it drop straight
        release_tool0 = place_tool0.copy()
        release_tool0[2] += self.PLACE_RELEASE_HEIGHT
        
        transit_tool0 = place_tool0.copy()
        transit_tool0[2] += self.TRANSIT_HEIGHT
        
        # Compute IK
        transit_joints = self.compute_ik_position(transit_tool0, self.last_lift_joints)
        if transit_joints is None:
            return False
        
        release_joints = self.compute_ik_position(release_tool0, transit_joints)
        if release_joints is None:
            return False
        
        # Execute:
        # 1. Transit high
        self.get_logger().info('Transit high...')
        self.move_arm(transit_joints)
        
        # 2. Lower to release height (not all the way down)
        self.get_logger().info('Lowering to release...')
        self.move_arm(release_joints)
        
        # 3. Open gripper - let object drop straight
        self.get_logger().info('Releasing...')
        self.move_gripper(self.GRIPPER_OPEN)
        
        # 4. Retreat up
        self.get_logger().info('Retreating...')
        self.move_arm(transit_joints)
        
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
    node = PickAndPlace6DOF()
    
    try:
        node.get_logger().info('Waiting for detections...')
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
        
        for color in ['red', 'green', 'blue']:
            if color in node.detected_objects:
                node.execute_pick_place(color)
            else:
                node.get_logger().warn(f'{color} not detected, skipping')
        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
