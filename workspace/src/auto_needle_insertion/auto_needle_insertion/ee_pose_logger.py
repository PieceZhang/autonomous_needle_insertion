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
import csv
import os
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

# Logging
LOG_CSV_PATH = "handeye_logs/ee_poses.csv"  # relative to current working directory
POST_EXECUTION_SETTLE_SEC = 0.75  # small wait to ensure state update after motion
BASE_LINK_CANDIDATES = ["base_link", "base", "world"]

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


# --------------------
# Helper utilities
# --------------------

def ensure_log_path(path: str) -> None:
    """Create parent directory for the CSV log if it doesn't exist."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def rmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (x, y, z, w) quaternion.
    Numerically stable branch from the standard algorithm.
    """
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        # Find the major diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


def get_base_link_name(robot: MoveItPy) -> str:
    """Return the model frame (typically the root link name).

    MoveItPy's RobotModel does not expose global link names directly; instead,
    use `model_frame`, which is either the SRDF-defined model frame or the
    root link name. This is the correct base frame for transforms.
    """
    return robot.get_robot_model().model_frame


def relative_transform(parent_T: np.ndarray, child_T: np.ndarray) -> np.ndarray:
    """Compute homogeneous transform of child in parent frame.

    Args:
        parent_T: 4x4 transform of parent in world/model frame
        child_T: 4x4 transform of child in world/model frame
    Returns:
        4x4 transform of child in parent frame
    """
    R_p = parent_T[:3, :3]
    t_p = parent_T[:3, 3]
    R_c = child_T[:3, :3]
    t_c = child_T[:3, 3]

    R_rel = R_p.T @ R_c
    t_rel = R_p.T @ (t_c - t_p)

    T_rel = np.eye(4)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = t_rel
    return T_rel


def write_csv_header_if_needed(path: str) -> None:
    if not os.path.exists(path):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "x", "y", "z", "qx", "qy", "qz", "qw"])


def append_pose_row(path: str, index: int, T: np.ndarray) -> None:
    pos = T[:3, 3]
    quat = rmat_to_quat(T[:3, :3])
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index, float(pos[0]), float(pos[1]), float(pos[2]),
                         float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])


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
        (0.0, -half_edge),  # Bottom-edge center
        (half_edge, -half_edge),  # Bottom-right
        (half_edge, 0.0),  # Right-edge center
        (half_edge, half_edge),  # Top-right
        (0.0, half_edge),  # Top-edge center
        (-half_edge, half_edge),  # Top-left
        (-half_edge, 0.0),  # Left-edge center
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

        # Determine base link and prepare CSV logging
        base_frame = get_base_link_name(robot)
        logger.info(f"Using base frame: {base_frame}")
        ensure_log_path(LOG_CSV_PATH)
        write_csv_header_if_needed(LOG_CSV_PATH)

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

            # --- Pose logging in base frame ---
            # Allow a short settle time so the planning scene reflects the executed state
            time.sleep(POST_EXECUTION_SETTLE_SEC)
            with robot.get_planning_scene_monitor().read_only() as scene_after:
                scene_after.current_state.update()
                # Base is the model frame; this returns identity if frame == model frame
                T_base = scene_after.current_state.get_frame_transform(base_frame)
                T_tip = scene_after.current_state.get_global_link_transform(tip_link)
            T_tip_in_base = relative_transform(T_base, T_tip)
            append_pose_row(LOG_CSV_PATH, i, T_tip_in_base)
            logger.info(
                f"Logged EE pose (base frame) for waypoint {i + 1}: "
                f"pos=({T_tip_in_base[0,3]:.4f}, {T_tip_in_base[1,3]:.4f}, {T_tip_in_base[2,3]:.4f})"
            )

            logger.info(f"Reached waypoint {i + 1}/{len(waypoint_poses)}")

        logger.info("Square trajectory completed successfully")

    except Exception as e:
        logger.error(f"Square trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
