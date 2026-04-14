from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir     = get_package_share_directory('vision_yolo_ros2')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    camera_node = Node(
        package='vision_yolo_ros2',
        executable='camera_publisher_node',
        name='camera_publisher_node',
        parameters=[params_file],
        output='screen',
    )

    recognition_node = Node(
        package='vision_yolo_ros2',
        executable='object_recognition_node',
        name='object_recognition_node',
        parameters=[params_file],
        output='screen',
    )

    return LaunchDescription([
        camera_node,
        recognition_node,
    ])
