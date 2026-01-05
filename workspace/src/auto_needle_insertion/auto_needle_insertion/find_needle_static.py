"""End-effector square trajectory execution module.

This module drives a robot end-effector to trace a square in its local XY plane
while maintaining constant orientation. Uses MoveItPy for iterative pose goals.

Key behaviors:
    - Discovers planning group and EE link dynamically
    - Samples current pose, builds waypoints in EE local frame
    - Plans and executes sequentially with conservative speed scaling
    - Falls back across multiple controller names for sim vs hardware

Notes:
    - Edge length currently set to 0.2 m (200 mm)
    - Orientation is locked to initial orientation
    - No Cartesian path interpolation; each corner is a separate pose plan

Safety:
    Always verify clearance before executing trajectories on real hardware.
"""

import logging
import time
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_link_constraint
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.node import Node

from auto_needle_insertion.utils.needle import Needle
from auto_needle_insertion.utils.optical_tracking import read_instrument_pose
from auto_needle_insertion.utils.find_needle import align_image_to_needle_axis, center_needle_in_image
from auto_needle_insertion.utils.pose_representations import (
    homogeneous_to_pose_stamped,
    quat_to_T,
)
from auto_needle_insertion.utils.us_probe import USProbe

# Module constants
NODE_NAME = "auto_needle_insertion"
SQUARE_EDGE_LENGTH = 0.2  # meters
MAX_VELOCITY_SCALING = 0.2
MAX_ACCELERATION_SCALING = 0.2
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds

# Controller fallback order
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",  # UR hardware typical
    "",  # Default controller
    "joint_trajectory_controller"  # Simulation/common
]

# Preferred tip link names in order of preference
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def get_planning_group_name(robot: MoveItPy) -> str:
    """Select appropriate planning group using heuristics.

    Args:
        robot: MoveItPy instance

    Returns:
        Planning group name

    Raises:
        RuntimeError: If no planning groups are available
    """
    group_names = robot.get_robot_model().joint_model_group_names
    if not group_names:
        raise RuntimeError("No planning groups available")

    logger.info(f"Available planning groups: {group_names}")

    # Prefer manipulator-like groups
    for group_name in group_names:
        if "manipulator" in group_name or "ur" in group_name:
            return group_name

    return group_names[0]


def get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    """Resolve appropriate tip link for the planning group.

    Args:
        robot: MoveItPy instance
        group_name: Planning group name

    Returns:
        Tip link name

    Raises:
        RuntimeError: If no links are available in the group
    """
    group = robot.get_robot_model().get_joint_model_group(group_name)
    if not group:
        raise RuntimeError(f"Planning group '{group_name}' not found")

    link_names = list(group.link_model_names)
    if not link_names:
        raise RuntimeError(f"No links found in planning group '{group_name}'")

    # Try preferred links first
    for preferred_link in PREFERRED_TIP_LINKS:
        if preferred_link in link_names:
            return preferred_link

    return link_names[-1]  # Fallback to last link


def execute_trajectory_with_fallback(
    robot: MoveItPy,
    trajectory,
    controllers: List[str] = CONTROLLER_NAMES
) -> bool:
    """Execute trajectory with controller fallback strategy.

    Args:
        robot: MoveItPy instance
        trajectory: Planned trajectory to execute
        controllers: List of controller names to try in order

    Returns:
        True if execution succeeded, False otherwise
    """
    for controller in controllers:
        try:
            if controller:
                robot.execute(trajectory, controllers=[controller])
            else:
                robot.execute(trajectory)
            return True
        except Exception as e:
            logger.warning(f"Controller '{controller}' failed: {e}")
            continue

    logger.error("All controllers failed")
    return False


def main() -> None:
    """Main execution function for square trajectory."""
    rclpy.init()

    try:
        robot = MoveItPy(node_name=NODE_NAME)

        # Allow time for joint states to populate the planning scene
        time.sleep(PLANNING_SCENE_SYNC_DELAY)

        psm = robot.get_planning_scene_monitor()

        # Force sync to ensure up-to-date robot state
        with psm.read_write() as scene:
            scene.current_state.update()

        # Get planning frame
        with psm.read_only() as scene_ro:
            planning_frame = scene_ro.planning_frame
        logger.info(f"Planning frame: {planning_frame}")

        # Setup planning components
        arm_group_name = get_planning_group_name(robot)
        tip_link = get_tip_link_name(robot, arm_group_name)
        arm = robot.get_planning_component(arm_group_name)

        logger.info(f"Using planning group: {arm_group_name}")
        logger.info(f"Using tip link: {tip_link}")

        # Set initial state
        arm.set_start_state_to_current_state()

        # Get current end-effector pose and transform
        with robot.get_planning_scene_monitor().read_only() as scene:
            scene.current_state.update()
            current_ee_transform = scene.current_state.get_global_link_transform(tip_link)

        us_probe = USProbe()
        us_probe.load_calibrations(
            "./calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20251230_SRIL.xml",
            "./calibration/hand_eye_20251231_075559.json",
        )
        to_in_probe = us_probe.to_in_probe
        to_in_ee = us_probe.to_in_ee
        probe_in_ee = us_probe.probe_in_ee
        if to_in_probe is None or to_in_ee is None or probe_in_ee is None:
            raise RuntimeError("US probe calibration failed to compute TO transforms.")
        probe_pose = us_probe.report_pose(timeout_sec=2.0)

        needle = Needle()
        needle.load_tip_offset("./calibration/needle_1_tip_offset.json")
        needle_pose = needle.report_pose(timeout_sec=2.0)
        needle_tip_position = needle.tip_position_in_tracker(needle_pose)

        # Calculate tracker frame and robot base frame
        tracker_in_base = current_ee_transform @ probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))

        to_in_tracker = quat_to_T(probe_pose) @ to_in_probe
        image_in_tracker_after_alignment = align_image_to_needle_axis(to_in_tracker, needle_pose[0:3], needle_tip_position)
        image_in_tracker_after_centering = center_needle_in_image(
            image_in_tracker_after_alignment, needle_pose[0:3], needle_tip_position,
            x_center_in_plane=0.0, y_target_in_plane=0.080)

        ee_target_pose_in_base = tracker_in_base @ image_in_tracker_after_centering @ np.linalg.inv(to_in_ee)
        logger.info(f"Target pose of EE in base: {ee_target_pose_in_base}")

        pose_goal = homogeneous_to_pose_stamped(ee_target_pose_in_base, planning_frame)
        arm.set_start_state_to_current_state()
        pos = pose_goal.pose.position
        ori = pose_goal.pose.orientation
        goal_c = construct_link_constraint(
            link_name=tip_link,
            source_frame=planning_frame,
            cartesian_position=[pos.x, pos.y, pos.z],
            cartesian_position_tolerance=1e-4,  # meters (start here)
            orientation=[ori.x, ori.y, ori.z, ori.w],
            orientation_tolerance=1e-4,  # radians (~0.057°) (start here)
        )
        arm.set_goal_state(motion_plan_constraints=[goal_c])

        # Setup conservative planning parameters
        plan_params = PlanRequestParameters(robot, "")
        plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

        # Plan trajectory
        plan_result = arm.plan(single_plan_parameters=plan_params)
        if not plan_result:
            logger.error(f"Planning to waypoint failed; aborting")

        # Execute with fallback
        if not execute_trajectory_with_fallback(robot, plan_result.trajectory):
            logger.error(f"Execution to waypoint failed; aborting")

        logger.info(f"Reached target pose")


    except Exception as e:
        logger.error(f"Trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
