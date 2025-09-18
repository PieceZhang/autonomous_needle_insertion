from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur5e")
    ur_type     = LaunchConfiguration("ur_type")

    # Use URDF from /robot_description of the robot driver, and select SRDF explicitly.
    ur_moveit_pkg = Path(get_package_share_directory("ur_moveit_config"))
    ur_desc_pkg   = Path(get_package_share_directory("ur_description"))
    my_pkg_share  = Path(get_package_share_directory("ani_moveit_py"))

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
    kin_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "default_kinematics.yaml"])
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

    # Node configuration
    node = Node(
        package="ani_moveit_py",
        executable="ee_moveit_square",
        name="moveit_py",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_robot_description_semantic": True}
        ],
    )

    return LaunchDescription([ur_type_arg, node])