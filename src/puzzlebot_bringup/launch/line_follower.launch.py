from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Activar debug visual del vision_node'
    )

    vision_params = os.path.join(
        get_package_share_directory('puzzlebot_bringup'),
        'config', 'vision_params.yaml'
    )

    controller_params = os.path.join(
        get_package_share_directory('puzzlebot_bringup'),
        'config', 'controller_params.yaml'
    )

    vision_node = Node(
        package='puzzlebot_perception',
        executable='vision_node',
        name='vision_node',
        parameters=[
            vision_params,
            {'debug': LaunchConfiguration('debug')}
        ],
        output='screen'
    )

    controller_node = Node(
        package='puzzlebot_control',
        executable='line_controller_node',
        name='line_controller_node',
        parameters=[controller_params],
        output='screen'
    )

    return LaunchDescription([
        debug_arg,
        vision_node,
        controller_node,
    ])
