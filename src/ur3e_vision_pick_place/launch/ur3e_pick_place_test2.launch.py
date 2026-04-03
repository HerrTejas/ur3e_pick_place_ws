import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import xacro

def generate_launch_description():
    pkg_dir = get_package_share_directory('ur3e_vision_pick_place')
    ur_sim_gz_pkg = get_package_share_directory('ur_simulation_gz')
    ur_description_pkg = get_package_share_directory('ur_description')
    
    world_file = os.path.join(pkg_dir, 'worlds', 'pick_place_test2.sdf')
    
    # Start Gazebo with our world
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_file],
        output='screen'
    )
    
    # Generate robot description with xacro
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([ur_sim_gz_pkg, 'urdf', 'ur_gz.urdf.xacro']),
        ' ur_type:=ur3e',
        ' name:=ur',
        ' safety_limits:=true',
        ' simulation_controllers:=',
        PathJoinSubstitution([ur_sim_gz_pkg, 'config', 'ur_controllers.yaml']),
    ])
    
    robot_description = {'robot_description': robot_description_content}
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # Spawn robot with ROTATION: -Y 1.5708 rotates 90 degrees (pi/2)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_description_content,
            '-name', 'ur',
            '-allow_renaming', 'true',
            '-x', '0.0',
            '-y', '0.0', 
            '-z', '0.4',
            '-Y', '1.5708',  # Rotate 90 degrees around Z
        ],
        output='screen'
    )
    
    # Controllers
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
    )
    
    joint_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['scaled_joint_trajectory_controller', 'gripper_controller', '-c', '/controller_manager'],
    )
    
    # Bridge
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen'
    )
    
    # Delay controllers to let robot spawn first
    delayed_controllers = TimerAction(
        period=3.0,
        actions=[joint_state_broadcaster_spawner, joint_controller_spawner]
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[spawn_robot]),
        gz_bridge,
        delayed_controllers,
    ])
