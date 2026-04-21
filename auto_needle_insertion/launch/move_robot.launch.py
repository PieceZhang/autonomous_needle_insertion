from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


MOVE_ROBOT_MODES = [
    "ee_moveit_square",
    "ee_moveit_keyboard",
    "ee_pose_logger",
    "hand_eye_calib",
    "tool_reporter",
    "tool_follower",
    "keyboard_control",
]


def generate_launch_description():
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur5e")
    ur_type     = LaunchConfiguration("ur_type")
    ur_calibration_file_arg = DeclareLaunchArgument(
        "ur_calibration_file",
        default_value="/ani_ws/calibration/ur5e_calibration.yaml",
        description="Path to UR calibration kinematics YAML",
    )
    ur_calibration_file = LaunchConfiguration("ur_calibration_file")

    mode_arg  = DeclareLaunchArgument(
        "mode",
        default_value="ee_moveit_square",
        choices=MOVE_ROBOT_MODES,
    )
    mode_name = LaunchConfiguration("mode")

    target_arg = DeclareLaunchArgument("target", default_value="us_probe")
    target_name = LaunchConfiguration("target")

    # Control ROS 2 log verbosity for this node (affects all loggers in-process)
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="error",
        description="Global log level for the node: DEBUG|INFO|WARN|ERROR|FATAL",
    )
    log_level = LaunchConfiguration("log_level")

    # Use URDF from /robot_description of the robot driver, and select SRDF explicitly.
    ur_moveit_pkg = Path(get_package_share_directory("ur_moveit_config"))
    ur_desc_pkg   = Path(get_package_share_directory("ur_description"))
    my_pkg_share  = Path(get_package_share_directory("auto_needle_insertion"))

    # SRDF configuration
    srdf_rel = "srdf/ur.srdf.xacro"
    # Fallback to plain SRDF if the xacro is absent
    if not (ur_moveit_pkg / srdf_rel).exists():
        alt = "srdf/ur.srdf"
        if (ur_moveit_pkg / alt).exists():
            srdf_rel = alt

    # Absolute path to the canonical UR xacro
    urdf_xacro = str(ur_desc_pkg / "urdf/ur.urdf.xacro")

    # UR xacro requires these four files; build them using substitutions so they accept ur_type at launch
    kin_yaml  = ur_calibration_file
    jl_yaml   = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "joint_limits.yaml"])
    phys_yaml = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "physical_parameters.yaml"])
    vis_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "visual_parameters.yaml"])

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
        .moveit_cpp(file_path=str(my_pkg_share / "config" / "moveit_cpp.yaml"))
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    node = Node(
        package="auto_needle_insertion",
        executable=mode_name,
        name="move_robot",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_robot_description_semantic": True},
            {"calibration_target": target_name},
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription([ur_type_arg, ur_calibration_file_arg, mode_arg, target_arg, log_level_arg, node])
