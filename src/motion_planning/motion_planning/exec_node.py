import numpy as np
import rclpy
from rclpy.node import Node
from custom_interfaces.msg import RobotCommand, RobotStatus
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from .kinematics import UR3eKinematics
from .path_interpolator import PathInterpolator
from .trajectory_planner import TrajectoryPlanner

# UR3e joint names as reported by the joint_states topic
JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

PLANNING_SPACE_JOINT = 0
PLANNING_SPACE_CARTESIAN = 1
PLANNING_METHOD_VEL_ACC = 0   # trapezoidal: needs v_max and a_max
PLANNING_METHOD_TIME = 1       # cubic polynomial: needs time_of_motion


class ExecutableNode(Node):
    """
    ROS 2 node that receives RobotCommand messages, plans a joint-space
    trajectory (either in joint space or Cartesian space), and publishes
    the resulting JointTrajectory to the robot controller.

    Topics:
      Subscribed:
        /joint_states         (sensor_msgs/JointState)
        /robot_cmd            (custom_interfaces/RobotCommand)
      Published:
        /joint_trajectory_controller/joint_trajectory  (trajectory_msgs/JointTrajectory)
        /robot_status         (custom_interfaces/RobotStatus)
    """

    def __init__(self):
        super().__init__('executable_node')

        # State
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)
        self.is_in_motion = False

        # Planning objects
        self.kinematics = UR3eKinematics()
        self.path_interpolator = PathInterpolator()
        self.trajectory_planner = TrajectoryPlanner(dt=0.01)

        # Subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.target_cmd_sub = self.create_subscription(
            RobotCommand, '/robot_cmd', self.cmd_callback, 10)

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)
        self.status_pub = self.create_publisher(RobotStatus, '/robot_status', 10)

        # Status broadcast at 10 Hz
        self.status_timer = self.create_timer(0.1, self.status_timer_callback)

        self.get_logger().info('[exec_node] ExecutableNode started.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def joint_state_callback(self, msg: JointState):
        """Cache the latest joint positions/velocities from the robot."""
        if len(msg.position) >= 6:
            self.current_joint_positions = np.array(msg.position[:6])
        if len(msg.velocity) >= 6:
            self.current_joint_velocities = np.array(msg.velocity[:6])

    def status_timer_callback(self):
        """Publish a RobotStatus message at 10 Hz."""
        status = RobotStatus()
        status.joint_position = self.current_joint_positions.tolist()
        status.joint_velocity = self.current_joint_velocities.tolist()
        status.joint_acceleration = [0.0] * 6
        status.joint_torque = [0.0] * 6

        ee_pose = self.kinematics.forward_kinematics(self.current_joint_positions)
        status.end_effector_position = ee_pose[:3].tolist()
        status.end_effector_velocity = [0.0] * 3
        status.end_effector_force = [0.0] * 3
        status.is_robot_in_motion = self.is_in_motion

        self.status_pub.publish(status)

    def cmd_callback(self, cmd_msg: RobotCommand):
        """Dispatch incoming command to the correct planning routine."""
        target_position = list(cmd_msg.target_position)
        planning_space = int(cmd_msg.planning_space)
        planning_method = int(cmd_msg.planning_method)

        self.get_logger().info(
            f'[exec_node] Command received — space={planning_space}, '
            f'method={planning_method}, target={target_position}')

        if planning_space == PLANNING_SPACE_JOINT:
            self._plan_and_execute_joint_space(target_position, planning_method, cmd_msg)
        elif planning_space == PLANNING_SPACE_CARTESIAN:
            self._plan_and_execute_cartesian_space(target_position, planning_method, cmd_msg)
        else:
            self.get_logger().warn(
                f'[exec_node] Unknown planning_space={planning_space}, ignoring.')

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------

    def _plan_and_execute_joint_space(self, target_joint_angles, planning_method, cmd_msg):
        if len(target_joint_angles) != 6:
            self.get_logger().error(
                f'[exec_node] Joint-space target must have 6 values, '
                f'got {len(target_joint_angles)}.')
            return

        start_q = self.current_joint_positions.copy()
        path_lengths = self.path_interpolator.init(
            start_q, target_joint_angles, mode=PathInterpolator.MODE_JOINT)

        self.get_logger().info(
            f'[exec_node] Joint path lengths (rad): {path_lengths.round(4).tolist()}')

        if not self._init_time_parameterisation(path_lengths, planning_method, cmd_msg):
            return

        path_array = self._make_path_array_for_execution()
        self._publish_trajectory(path_array)

    def _plan_and_execute_cartesian_space(self, target_cartesian_pose, planning_method, cmd_msg):
        if len(target_cartesian_pose) != 6:
            self.get_logger().error(
                '[exec_node] Cartesian target must have 6 values [x,y,z,rx,ry,rz].')
            return

        # Compute current end-effector pose via FK
        start_ee_pose = self.kinematics.forward_kinematics(
            self.current_joint_positions).tolist()

        self.get_logger().info(
            f'[exec_node] Cartesian path: start={[round(v,4) for v in start_ee_pose]}, '
            f'end={[round(v,4) for v in target_cartesian_pose]}')

        path_lengths = self.path_interpolator.init(
            start_ee_pose, target_cartesian_pose, mode=PathInterpolator.MODE_CARTESIAN)

        self.get_logger().info(
            f'[exec_node] Cartesian path_lengths: {path_lengths.round(4).tolist()}')

        if not self._init_time_parameterisation(path_lengths, planning_method, cmd_msg):
            return

        path_array = self._make_path_array_for_execution()
        self._publish_trajectory(path_array)

    def _init_time_parameterisation(self, path_lengths, planning_method, cmd_msg):
        """
        Initialise TrajectoryPlanner based on the requested method.
        Returns True on success, False on error.
        """
        start_velocities = np.zeros(len(path_lengths))

        if planning_method == PLANNING_METHOD_VEL_ACC:
            v_max = float(cmd_msg.v_max)
            a_max = float(cmd_msg.a_max)
            if v_max <= 0 or a_max <= 0:
                self.get_logger().error(
                    f'[exec_node] v_max={v_max} and a_max={a_max} must be positive.')
                return False
            total_time = self.trajectory_planner.init_trajectory_planner(
                start_velocities, path_lengths, v_max, a_max)
            self.get_logger().info(
                f'[exec_node] Trapezoidal profile: v_max={v_max} rad/s, '
                f'a_max={a_max} rad/s², total_time={total_time:.3f} s')

        elif planning_method == PLANNING_METHOD_TIME:
            total_time = float(cmd_msg.time_of_motion)
            if total_time <= 0:
                self.get_logger().error(
                    f'[exec_node] time_of_motion={total_time} must be positive.')
                return False
            self.trajectory_planner.init_trajectory_planner_time(path_lengths, total_time)
            self.get_logger().info(
                f'[exec_node] Cubic profile: total_time={total_time:.3f} s')

        else:
            self.get_logger().warn(
                f'[exec_node] Unknown planning_method={planning_method}, ignoring.')
            return False

        return True

    def _make_path_array_for_execution(self):
        """
        Query TrajectoryPlanner for sampled waypoints and convert to absolute
        joint positions via PathInterpolator.get_pose_at().

        Works for both joint and Cartesian mode — the interpolator handles
        the difference internally (linear for joint, linear+SLERP for Cartesian).
        In Cartesian mode the resulting pose is then resolved to joint angles via IK.

        Returns:
            list of dicts: {'time': float, 'positions': list[float],
                            'velocities': list[float]}
            Empty list if Cartesian IK fails at any waypoint.
        """
        traj_points = self.trajectory_planner.get_trajectory_points()
        path_array = []
        q_seed = self.current_joint_positions.copy()
        is_cartesian = (self.path_interpolator._mode == PathInterpolator.MODE_CARTESIAN)

        for pt in traj_points:
            config = self.path_interpolator.get_pose_at(pt['positions'])

            if is_cartesian:
                # config is [x,y,z,rx,ry,rz] — resolve to joint angles via IK
                q_sol = self.kinematics.inverse_kinematics(config, q_seed)
                if q_sol is None:
                    self.get_logger().error(
                        f'[exec_node] Cartesian IK failed at t={pt["time"]:.3f} s. '
                        f'Aborting trajectory.')
                    return []
                q_seed = q_sol
                joint_positions = q_sol.tolist()
            else:
                joint_positions = config.tolist()

            path_array.append({
                'time': pt['time'],
                'positions': joint_positions,
                'velocities': pt['velocities'].tolist(),
            })

        label = 'Cartesian IK' if is_cartesian else 'Joint'
        self.get_logger().info(
            f'[exec_node] {label} path: {len(path_array)} waypoints '
            f'over {traj_points[-1]["time"]:.3f} s.')
        return path_array

    def _publish_trajectory(self, path_array):
        """
        Build and publish a JointTrajectory message to the robot controller.
        """
        if not path_array:
            self.get_logger().error('[exec_node] Empty path array — trajectory not published.')
            return

        traj_msg = JointTrajectory()
        traj_msg.joint_names = JOINT_NAMES

        for pt in path_array:
            traj_pt = JointTrajectoryPoint()
            traj_pt.positions = pt['positions']
            traj_pt.velocities = pt['velocities']
            t_sec = pt['time']
            traj_pt.time_from_start = Duration(
                sec=int(t_sec),
                nanosec=int((t_sec % 1.0) * 1_000_000_000)
            )
            traj_msg.points.append(traj_pt)

        self.is_in_motion = True
        self.trajectory_pub.publish(traj_msg)
        self.get_logger().info(
            f'[exec_node] Published JointTrajectory ({len(path_array)} points).')


def main(args=None):
    rclpy.init(args=args)
    node = ExecutableNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

