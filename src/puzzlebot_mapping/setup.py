from setuptools import setup
import os
from glob import glob

package_name = 'puzzlebot_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='urielCrespo',
    maintainer_email='a01412388@tec.mx',
    description='Occupancy grid mapping from scratch para PuzzleBot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odometry_node = puzzlebot_mapping.odometry_node:main',
            'occupancy_grid_node = puzzlebot_mapping.occupancy_grid_node:main',
            'map_saver_node = puzzlebot_mapping.map_saver_node:main',
        ],
    },
)