#!/usr/bin/env python3
"""
UR3e Vision Pick and Place Launch File
Launches UR3e simulation with custom world and camera bridge
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package paths
    pkg_ur3e_vision = FindPackageShare('ur3e_vision_pick_place')
    pkg_ur_simulation = FindPackageShare('ur_simulation_gz')

    # Launch arguments
    world_file_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([pkg_ur3e_vision, 'worlds', 'pick_place_world.sdf']),
        description='Path to world file'
    )

    ur_type_arg = DeclareLaunchArgument(
        'ur_type',
        default_value='ur3e',
        description='UR robot type'
    )

    # Include UR simulation launch
    ur_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ur_simulation, 'launch', 'ur_sim_control.launch.py'])
        ]),
        launch_arguments={
            'ur_type': LaunchConfiguration('ur_type'),
            'world_file': LaunchConfiguration('world'),
            'launch_rviz': 'true',
        }.items()
    )

    # Gazebo bridge for camera topics
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge_camera',
        parameters=[{
            'config_file': PathJoinSubstitution([pkg_ur3e_vision, 'config', 'gz_bridge.yaml']),
            'use_sim_time': True,
        }],
        output='screen'
    )

    return LaunchDescription([
        world_file_arg,
        ur_type_arg,
        ur_simulation,
        gz_bridge,
    ])
