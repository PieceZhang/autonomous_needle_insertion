from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from launch.substitutions import TextSubstitution

from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    ur_type_arg = DeclareLaunchArgument("ur_type", default_value="ur5e")
    ur_type = LaunchConfiguration("ur_type")

    # Point builder at the UR MoveIt package, but DO NOT hardcode URDF/SRDF filenames.
    # Let URDF come from /robot_description (driver), and select SRDF explicitly.
    ur_moveit_pkg = Path(get_package_share_directory("ur_moveit_config"))
    ur_desc_pkg = Path(get_package_share_directory("ur_description"))
    my_pkg_share = Path(get_package_share_directory("ani_moveit_py"))

    # On Jazzy, ur_moveit_config installs SRDF under srdf/ (srdf/ur.srdf.xacro).
    srdf_rel = "srdf/ur.srdf.xacro"
    # Fallback to plain SRDF if the xacro is absent.
    if not (ur_moveit_pkg / srdf_rel).exists():
        alt = "srdf/ur.srdf"
        if (ur_moveit_pkg / alt).exists():
            srdf_rel = alt

    # Absolute path to the canonical UR xacro (kinematics-only; control lives in driver/mock)
    urdf_xacro = str(ur_desc_pkg / "urdf/ur.urdf.xacro")

    # UR xacro requires these four files; build them using substitutions so they accept ur_type at launch
    kin_yaml = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "default_kinematics.yaml"]) 
    jl_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "joint_limits.yaml"]) 
    phys_yaml = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "physical_parameters.yaml"]) 
    vis_yaml  = PathJoinSubstitution([str(ur_desc_pkg), "config", ur_type, "visual_parameters.yaml"]) 

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        # Provide URDF explicitly to avoid the fallback to non-existent config/ur.urdf
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
        # SRDF is provided by ur_moveit_config; it is a xacro and needs ur_type
        .robot_description_semantic(file_path=srdf_rel, mappings={"ur_type": ur_type, "name": ur_type})
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])  # ensure MoveItPy/MoveItCpp sees a pipeline
        .moveit_cpp(file_path=str(my_pkg_share / "config" / "moveit_cpp.yaml"))
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    node = Node(
        package="ani_moveit_py",
        executable="ee_moveit_square",
        name="moveit_py",
        output="screen",
        parameters=[moveit_config.to_dict(), {"publish_robot_description_semantic": True}],
    )

    return LaunchDescription([ur_type_arg, node])