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
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters

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


def generate_square_waypoints(edge_length: float) -> List[Tuple[float, float]]:
    """Generate waypoints for a square path in local coordinates.

    Args:
        edge_length: Length of square edge in meters

    Returns:
        List of (dx, dy) waypoints in local frame
    """
    half_edge = edge_length / 2.0

    # Square path: bottom-left -> CCW -> close -> return to center
    return [
        (-half_edge, -half_edge),  # Bottom-left
        (half_edge, -half_edge),   # Bottom-right
        (half_edge, half_edge),    # Top-right
        (-half_edge, half_edge),   # Top-left
        (-half_edge, -half_edge),  # Close the square
        (0.0, 0.0),               # Return to center
    ]


def create_pose_stamped(
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    dx: float,
    dy: float,
    orientation,
    planning_frame: str
) -> PoseStamped:
    """Create PoseStamped message for waypoint in local frame.

    Args:
        origin: Origin position in global frame
        x_axis: Normalized X-axis of local frame
        y_axis: Normalized Y-axis of local frame
        dx: Displacement along local X-axis
        dy: Displacement along local Y-axis
        orientation: Original orientation to maintain
        planning_frame: Planning frame ID

    Returns:
        PoseStamped message for the waypoint
    """
    position = origin + dx * x_axis + dy * y_axis

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = planning_frame
    pose_stamped.pose.position.x = float(position[0])
    pose_stamped.pose.position.y = float(position[1])
    pose_stamped.pose.position.z = float(position[2])
    pose_stamped.pose.orientation = orientation

    return pose_stamped


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
            transform_matrix = scene.current_state.get_global_link_transform(tip_link)
            current_pose = scene.current_state.get_pose(tip_link)

        # Extract and normalize local coordinate frame
        rotation_matrix = transform_matrix[:3, :3]
        x_axis = rotation_matrix[:, 0] / np.linalg.norm(rotation_matrix[:, 0])
        y_axis = rotation_matrix[:, 1] / np.linalg.norm(rotation_matrix[:, 1])
        origin = transform_matrix[:3, 3]

        # Generate square waypoints
        local_waypoints = generate_square_waypoints(SQUARE_EDGE_LENGTH)

        # Create pose messages for each waypoint
        waypoint_poses = [
            create_pose_stamped(
                origin, x_axis, y_axis, dx, dy,
                current_pose.orientation, planning_frame
            )
            for dx, dy in local_waypoints
        ]

        # Execute square trajectory
        for i, waypoint_pose in enumerate(waypoint_poses):
            arm.set_start_state_to_current_state()
            arm.set_goal_state(pose_stamped_msg=waypoint_pose, pose_link=tip_link)

            # Setup conservative planning parameters
            plan_params = PlanRequestParameters(robot, "")
            plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
            plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

            # Plan trajectory
            plan_result = arm.plan(single_plan_parameters=plan_params)
            if not plan_result:
                logger.error(f"Planning to waypoint {i} failed; aborting")
                break

            # Execute with fallback
            if not execute_trajectory_with_fallback(robot, plan_result.trajectory):
                logger.error(f"Execution to waypoint {i} failed; aborting")
                break

            logger.info(f"Reached waypoint {i + 1}/{len(waypoint_poses)}")

        logger.info("Square trajectory completed successfully")

    except Exception as e:
        logger.error(f"Square trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

