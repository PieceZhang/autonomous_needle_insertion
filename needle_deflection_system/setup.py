from setuptools import setup
import os
from glob import glob

package_name = 'needle_deflection_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (
            os.path.join('share', package_name, 'params'),
            glob(os.path.join(package_name, 'params', '*.yaml')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='Needle deflection system for robotic surgery',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ati_force_sensor_driver = needle_deflection_system.ati_force_sensor_driver:main',
            # Keep a backward-compatible alias for existing scripts/service names.
            'ati_ft_nano17_driver = needle_deflection_system.ati_force_sensor_driver:main',
            'needle_deflection_calculator = needle_deflection_system.needle_deflection_calculator:main',
        ],
    },
)
