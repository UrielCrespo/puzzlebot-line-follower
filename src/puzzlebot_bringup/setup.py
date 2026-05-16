from setuptools import setup
import os
from glob import glob

package_name = 'puzzlebot_bringup'

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
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuye',
    maintainer_email='a01412388@tec.mx',
    description='Launch files y configs del PuzzleBot',
    license='MIT',
    entry_points={
        'console_scripts': [],
    },
)
