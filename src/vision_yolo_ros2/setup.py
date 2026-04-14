from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vision_yolo_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='Nodos ROS2 para reconocimiento de objetos con YOLO',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_publisher_node = vision_yolo_ros2.camera_publisher_node:main',
            'object_recognition_node = vision_yolo_ros2.object_recognition_node:main',
        ],
    },
)
