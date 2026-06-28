#!/usr/bin/env python3
"""
UR3e Pick and Place Launch — v4

Toggle between two controller modes:
  use_cartesian:=false  → mixed mode (joint-space for long moves,
                          cartesian for verticals). Default.
  use_cartesian:=true   → all-cartesian mode (every move uses the
                          cartesian planner). Mirrors v1 behavior.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_ur3e_vision = FindPackageShare('ur3e_vision_pick_place')
    pkg_ur_simulation = FindPackageShare('ur_simulation_gz')

    world_file_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([pkg_ur3e_vision, 'worlds', 'pick_place_world.sdf']),
        description='Path to world file'
    )
    ur_type_arg = DeclareLaunchArgument(
        'ur_type', default_value='ur3e', description='UR robot type'
    )
    use_cartesian_arg = DeclareLaunchArgument(
        'use_cartesian',
        default_value='false',
        description='If true, use all-cartesian controller. If false, mixed mode.'
    )

    use_cartesian = LaunchConfiguration('use_cartesian')

    ur_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ur_simulation, 'launch', 'ur_sim_control.launch.py'])
        ]),
        launch_arguments={
            'ur_type': LaunchConfiguration('ur_type'),
            'world_file': LaunchConfiguration('world'),
            'launch_rviz': 'false',
        }.items()
    )

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

    # Always-on infrastructure
    forward_kinematics_v2 = Node(
        package='ur3e_vision_pick_place',
        executable='forward_kinematics_v2',
        name='forward_kinematics',
        output='screen'
    )

    cartesian_planner = Node(
        package='ur3e_vision_pick_place',
        executable='cartesian_planner',
        name='cartesian_planner',
        output='screen'
    )

    object_detector_node = Node(
        package='ur3e_vision_pick_place',
        executable='object_detector',
        name='object_detector',
        output='screen'
    )

    gui_node = Node(
        package='ur3e_vision_pick_place',
        executable='gui_node',
        name='gui_node',
        output='screen'
    )

    # Mixed-mode-only nodes (only when use_cartesian=false)
    inverse_kinematics_v2 = Node(
        package='ur3e_vision_pick_place',
        executable='inverse_kinematics_v2',
        name='inverse_kinematics',
        output='screen',
        condition=UnlessCondition(use_cartesian),
    )

    joint_planner = Node(
        package='ur3e_vision_pick_place',
        executable='joint_planner',
        name='joint_planner',
        output='screen',
    )

    pick_and_place_mixed = Node(
        package='ur3e_vision_pick_place',
        executable='pick_and_place_v4',
        name='pick_and_place_v4',
        output='screen',
        condition=UnlessCondition(use_cartesian),
    )

    # Cartesian-only controller (only when use_cartesian=true)
    pick_and_place_cartesian = Node(
        package='ur3e_vision_pick_place',
        executable='pick_and_place_v4_cartesian',
        name='pick_and_place_v4_cartesian',
        output='screen',
        condition=IfCondition(use_cartesian),
    )

    return LaunchDescription([
        world_file_arg,
        ur_type_arg,
        use_cartesian_arg,
        ur_simulation,
        gz_bridge,
        forward_kinematics_v2,
        cartesian_planner,
        object_detector_node,
        gui_node,
        # Mixed-mode trio
        inverse_kinematics_v2,
        joint_planner,
        pick_and_place_mixed,
        # Cartesian-only controller
        pick_and_place_cartesian,
    ])