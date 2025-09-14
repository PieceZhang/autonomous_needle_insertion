from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from moveit_configs_utils import MoveItConfigsBuilder
import os


def generate_launch_description():
    # --- Args ---
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur5e")
    ur_type = LaunchConfiguration("ur_type")

    # --- Package shares ---
    ur_moveit_pkg = Path(get_package_share_directory("ur_moveit_config"))
    ur_desc_pkg = Path(get_package_share_directory("ur_description"))
    my_pkg_share = Path(get_package_share_directory("ani_moveit_py"))

    # --- SRDF path in ur_moveit_config ---
    # Prefer xacro; fall back to plain SRDF if needed
    srdf_rel = "srdf/ur.srdf.xacro"
    if not (ur_moveit_pkg / srdf_rel).exists():
        alt = "srdf/ur.srdf"
        if (ur_moveit_pkg / alt).exists():
            srdf_rel = alt

    # --- URDF xacro in ur_description ---
    urdf_xacro = str(ur_desc_pkg / "urdf/ur.urdf.xacro")

    # Required xacro mappings for UR robots
    kin_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "default_kinematics.yaml"])  # per model
    jl_yaml   = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "joint_limits.yaml"])       # per model
    phys_yaml = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "physical_parameters.yaml"]) # per model
    vis_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "visual_parameters.yaml"])   # per model

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        # Provide URDF explicitly (driver/mock can still publish /robot_description)
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
        # SRDF (needs ur_type)
        .robot_description_semantic(file_path=srdf_rel, mappings={"ur_type": ur_type, "name": ur_type})
        # Kinematics plugins (gives Servo an IK solver for ur_manipulator)
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        # Ensure descriptions are published for nodes that read from topics
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        # Minimal pipeline so MoveItPy/Servo have a planning setup
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        # Controller + execution params from ur_moveit_config
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

    # --- Servo node ---
    servo_params = os.path.join(str(my_pkg_share), "config", "servo_params.yaml")
    servo = Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),                 # robot_description, SRDF, kinematics, controllers, joint limits, etc.
            {"robot_description_planning": rdp},   # jerk-augmented limits for Ruckig (overrides the previous)
            servo_params,                            # your Servo YAML
            {
                "moveit_servo.move_group_name": "ur_manipulator",
                "moveit_servo.planning_frame": "base_link",
            },
        ],
    )

    # --- Static TF: base_link -> polaris_vega_base (adjust once calibrated) ---
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

    # --- Your follower node (Python) ---
    tool_follower = Node(
        package="ani_moveit_py",
        executable="tool_follower",
        name="tool_follower",
        output="screen",
        # It doesn’t need robot description; it uses TF and Servo topics
    )

    return LaunchDescription([
        ur_type_arg,
        servo,
        static_polaris_to_base,
        tool_follower,
    ])