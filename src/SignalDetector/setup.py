from setuptools import find_packages, setup

package_name = 'SignalDetector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skqchs',
    maintainer_email='leyva.gael@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_node = SignalDetector.detector_node:main',
            'detector_ei_node = SignalDetector.detector_edgeImpulse_node:main',
        ],
    },
)
