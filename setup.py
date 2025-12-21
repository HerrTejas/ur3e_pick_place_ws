from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur3e_vision_pick_place'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tejas',
    maintainer_email='herrtejasmurkute@gmail.com',
    description='Vision-based pick and place with UR3e robot arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detector = ur3e_vision_pick_place.color_detector:main',
            'color_detector_v2 = ur3e_vision_pick_place.color_detector_v2:main',
            'color_tuner = ur3e_vision_pick_place.color_tuner:main',
            'object_detector_3d = ur3e_vision_pick_place.object_detector_3d:main',
            'single_color_detector = ur3e_vision_pick_place.single_color_detector:main',
        ],
    },
)
