import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Launch arguments
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur5e")
    ur_type = LaunchConfiguration("ur_type")

    # Package paths
    ur_moveit_pkg = Path(get_package_share_directory("ur_moveit_config"))
    ur_desc_pkg   = Path(get_package_share_directory("ur_description"))
    my_pkg_share  = Path(get_package_share_directory("auto_needle_insertion"))

    # SRDF configuration
    srdf_rel = "srdf/ur.srdf.xacro"
    # Prefer xacro; fall back to plain SRDF if needed
    if not (ur_moveit_pkg / srdf_rel).exists():
        alt = "srdf/ur.srdf"
        if (ur_moveit_pkg / alt).exists():
            srdf_rel = alt

    # URDF configuration
    urdf_xacro = str(ur_desc_pkg / "urdf/ur.urdf.xacro")

    # Required xacro mappings for UR robots
    kin_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "default_kinematics.yaml"])  # per model
    jl_yaml   = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "joint_limits.yaml"])       # per model
    phys_yaml = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "physical_parameters.yaml"]) # per model
    vis_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "visual_parameters.yaml"])   # per model

    # MoveIt configuration
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(
            file_path=urdf_xacro,
            mappings={
                "name": ur_type,
                "kinematics_params": kin_yaml,
                "joint_limit_params": jl_yaml,
                "physical_params": phys_yaml,
                "visual_params": vis_yaml,
            },
        )
        .robot_description_semantic(
            file_path=srdf_rel,
            mappings={"ur_type": ur_type, "name": ur_type}
        )
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True
        )
        .planning_pipelines(
            default_planning_pipeline="ompl",
            pipelines=["ompl"]
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # Augment robot_description_planning with jerk limits (needed by Ruckig)
    moveit_dict = moveit_config.to_dict()
    rdp = moveit_dict.get("robot_description_planning", {})
    jl = rdp.get("joint_limits", {})
    for j_name, lim in jl.items():
        if not lim.get("has_jerk_limits", False):
            lim["has_jerk_limits"] = True
            # Safe default; tune if needed
            lim["max_jerk"] = 1000.0

    # Servo node
    servo_params = os.path.join(str(my_pkg_share), "config", "servo_params.yaml")
    servo = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"robot_description_planning": rdp},
            servo_params,
            {
                "moveit_servo.move_group_name": "ur_manipulator",
                "moveit_servo.planning_frame": "base_link",
            },
        ],
    )

    # Static TF: base_link -> polaris_vega_base (adjust once calibrated)
    static_polaris_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="polaris_to_base_link",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "base_link", "--child-frame-id", "polaris_vega_base",
        ],
        output="screen",
    )

    # Keyboard servo node
    keyboard_servo = Node(
        package="auto_needle_insertion",
        executable="keyboard_servo",
        name="keyboard_servo",
        output="screen",
    )

    return LaunchDescription([
        ur_type_arg,
        servo,
        static_polaris_to_base,
        keyboard_servo,
    ])
