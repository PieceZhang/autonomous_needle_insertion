from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('needle_deflection_system')

    ati_sensor_config = os.path.join(package_dir, 'params', 'ati_sensor.yaml')
    needle_deflection_config = os.path.join(package_dir, 'params', 'needle_deflection.yaml')

    ati_sensor_node = Node(
        package='needle_deflection_system',
        executable='ati_force_sensor_driver',
        name='ati_force_sensor_driver',
        parameters=[ati_sensor_config],
        output='screen'
    )

    needle_deflection_node = Node(
        package='needle_deflection_system',
        executable='needle_deflection_calculator',
        name='needle_deflection_calculator',
        parameters=[needle_deflection_config],
        output='screen'
    )

    return LaunchDescription([
        ati_sensor_node,
        needle_deflection_node
    ])