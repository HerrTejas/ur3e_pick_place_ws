#!/usr/bin/env python3
"""Object detector + live image view.

Starts object_detector.py and automatically opens rqt_image_view already
pointed at the annotated detection image, so you no longer have to launch
the viewer by hand in a second terminal.

Run:
    ros2 launch ur3e_vision_pick_place detector.launch.py

Optional: pick which image topic the viewer opens on
    ros2 launch ur3e_vision_pick_place detector.launch.py \
        view_topic:=/overhead_camera/image
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Which topic rqt_image_view opens on. Default is the debug image the
    # detector annotates with boxes/labels — the one that actually shows
    # the detections. Override to /overhead_camera/image for the raw feed
    # or /overhead_camera/depth_image for depth.
    view_topic_arg = DeclareLaunchArgument(
        'view_topic',
        default_value='/detected_objects_debug',
        description='Image topic rqt_image_view opens on at startup',
    )

    object_detector = Node(
        package='ur3e_vision_pick_place',
        executable='object_detector',
        name='object_detector',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # rqt_image_view takes the topic as a positional argument, so it opens
    # already showing that topic instead of a blank dropdown.
    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='detection_image_view',
        arguments=[LaunchConfiguration('view_topic')],
        output='screen',
    )

    return LaunchDescription([
        view_topic_arg,
        object_detector,
        image_view,
    ])
