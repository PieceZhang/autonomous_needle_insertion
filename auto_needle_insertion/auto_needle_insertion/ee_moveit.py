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
    - Approach distance currently set to 0.3 m
    - Orientation is locked to initial orientation
    - No Cartesian path interpolation; each corner is a separate pose plan

Safety:
    Always verify clearance before executing trajectories on real hardware.
"""

import logging
import threading
import time
from typing import List, Tuple

import numpy as np
import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, WrenchStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.executors import SingleThreadedExecutor

# Module constants
NODE_NAME = "auto_needle_insertion"
DEFAULT_TRAJECTORY = "square"
SQUARE_EDGE_LENGTH = 0.2  # meters
APPROACH_DISTANCE = 0.3  # meters
APPROACH_FORCE_Z_TOPIC = "/ati_ft_broadcaster/wrench"
APPROACH_FORCE_Z_LIMIT = -10.0  # N
FORCE_Z_OFFSET_COLLECTION_SEC = 1.0
LOG_FORCE_Z_INFO = False
MAX_VELOCITY_SCALING = 0.02
MAX_ACCELERATION_SCALING = 0.1
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds

# Controller fallback order
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",  # UR hardware typical
    "",  # Default controller
    "joint_trajectory_controller"  # Simulation/common
]

# Preferred tip link names in order of preference
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

TRAJECTORY_CANCEL_SERVICES = [
    "/execute_trajectory/_action/cancel_goal",
    "/move_group/execute_trajectory/_action/cancel_goal",
    "/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/cancel_goal",
    "/joint_trajectory_controller/follow_joint_trajectory/_action/cancel_goal",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForceZMonitor:
    """Monitor wrench force Z and cancel active trajectory goals below a limit."""

    def __init__(
        self,
        topic_name: str = APPROACH_FORCE_Z_TOPIC,
        force_limit: float = APPROACH_FORCE_Z_LIMIT,
        cancel_services: List[str] = TRAJECTORY_CANCEL_SERVICES,
    ) -> None:
        self.topic_name = topic_name
        self.force_limit = force_limit
        self.force_offset = 0.0
        self._collecting_offset = True
        self._offset_samples: List[float] = []
        self._offset_lock = threading.Lock()
        self._triggered = threading.Event()
        self._node = rclpy.create_node("ee_moveit_force_z_monitor")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop_spin = threading.Event()
        self._cancel_clients = [
            self._node.create_client(CancelGoal, service_name)
            for service_name in cancel_services
        ]
        self._sub = self._node.create_subscription(WrenchStamped, topic_name, self._force_cb, 10)
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    def start(self) -> None:
        self._spin_thread.start()
        logger.info(
            f"Collecting force Z offset from {self.topic_name} for "
            f"{FORCE_Z_OFFSET_COLLECTION_SEC:.1f} s"
        )
        start_time = time.monotonic()
        next_status_time = start_time
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= FORCE_Z_OFFSET_COLLECTION_SEC:
                break
            if time.monotonic() >= next_status_time:
                progress = min(elapsed / FORCE_Z_OFFSET_COLLECTION_SEC, 1.0)
                bar_width = 20
                filled = int(progress * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                logger.info(
                    f"Force Z offset collection [{bar}] "
                    f"{elapsed:.1f}/{FORCE_Z_OFFSET_COLLECTION_SEC:.1f} s"
                )
                next_status_time += 0.5
            time.sleep(0.05)

        with self._offset_lock:
            self._collecting_offset = False
            sample_count = len(self._offset_samples)
            if sample_count:
                self.force_offset = sum(self._offset_samples) / sample_count
            else:
                self.force_offset = 0.0
        logger.info(
            f"Force Z offset: {self.force_offset:.3f} N from {sample_count} samples"
        )
        logger.info(
            f"Monitoring {self.topic_name}; approach stops below {self.force_limit:.1f} N"
        )

    def stop(self) -> None:
        self._stop_spin.set()
        self._executor.wake()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._executor.remove_node(self._node)
        self._node.destroy_subscription(self._sub)
        self._node.destroy_node()

    def _spin(self) -> None:
        while not self._stop_spin.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _force_cb(self, msg: WrenchStamped) -> None:
        raw_force_z = msg.wrench.force.z

        with self._offset_lock:
            if self._collecting_offset:
                self._offset_samples.append(raw_force_z)
                return
            force_offset = self.force_offset

        force_z = raw_force_z - force_offset

        if LOG_FORCE_Z_INFO:
            logger.info(f"Force Z: {force_z:.3f} N (raw: {raw_force_z:.3f} N)")

        if force_z >= self.force_limit or self._triggered.is_set():
            return

        self._triggered.set()
        logger.warning(
            f"Force Z {force_z:.3f} N fell below {self.force_limit:.3f} N; "
            "canceling active approach trajectory"
        )
        self._cancel_active_goals()

    def _cancel_active_goals(self) -> None:
        request = CancelGoal.Request()
        for client in self._cancel_clients:
            if not client.service_is_ready():
                continue
            client.call_async(request)


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


def generate_approach_waypoints(distance: float) -> List[Tuple[float, float, float]]:
    """Generate waypoints for an approach move along local Z."""
    return [(0.0, 0.0, distance)]


def create_pose_stamped(
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    orientation,
    planning_frame: str
) -> PoseStamped:
    """Create PoseStamped message for waypoint in local frame.

    Args:
        origin: Origin position in global frame
        x_axis: Normalized X-axis of local frame
        y_axis: Normalized Y-axis of local frame
        z_axis: Normalized Z-axis of local frame
        dx: Displacement along local X-axis
        dy: Displacement along local Y-axis
        dz: Displacement along local Z-axis
        orientation: Original orientation to maintain
        planning_frame: Planning frame ID

    Returns:
        PoseStamped message for the waypoint
    """
    position = origin + dx * x_axis + dy * y_axis + dz * z_axis

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
    controllers: List[str] = CONTROLLER_NAMES,
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


def execute_trajectory_with_force_stop(
    robot: MoveItPy,
    trajectory,
    force_monitor: ForceZMonitor,
    controllers: List[str] = CONTROLLER_NAMES,
) -> bool:
    """Execute a trajectory while allowing force-triggered cancellation."""
    for controller in controllers:
        try:
            if controller:
                robot.execute(trajectory, controllers=[controller])
            else:
                robot.execute(trajectory)
            if force_monitor.triggered:
                logger.info("Trajectory execution stopped after force threshold was reached")
            return True
        except Exception as e:
            if force_monitor.triggered:
                logger.info("Trajectory execution canceled after force threshold was reached")
                return True
            logger.warning(f"Controller '{controller}' failed: {e}")
            continue

    logger.error("All controllers failed")
    return False


def plan_waypoint(robot: MoveItPy, arm, waypoint_pose: PoseStamped, tip_link: str):
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=waypoint_pose, pose_link=tip_link)

    plan_params = PlanRequestParameters(robot, "")
    plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
    plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

    return arm.plan(single_plan_parameters=plan_params)


def get_trajectory_parameter() -> str:
    """Read trajectory from ROS parameters without relying on MoveItPy internals."""
    node = rclpy.create_node(
        "ee_moveit_parameter_reader",
        automatically_declare_parameters_from_overrides=True,
    )
    try:
        if not node.has_parameter("trajectory"):
            node.declare_parameter("trajectory", DEFAULT_TRAJECTORY)
        return node.get_parameter("trajectory").get_parameter_value().string_value
    finally:
        node.destroy_node()


def main() -> None:
    """Main execution function for the selected trajectory."""
    rclpy.init()

    try:
        trajectory_name = get_trajectory_parameter()
        logger.info(f"Selected trajectory: {trajectory_name}")
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
        z_axis = rotation_matrix[:, 2] / np.linalg.norm(rotation_matrix[:, 2])
        origin = transform_matrix[:3, 3]

        if trajectory_name == "square":
            local_waypoints = [
                (dx, dy, 0.0)
                for dx, dy in generate_square_waypoints(SQUARE_EDGE_LENGTH)
            ]
        elif trajectory_name == "approach":
            local_waypoints = generate_approach_waypoints(APPROACH_DISTANCE)
        else:
            raise ValueError(f"Unsupported trajectory: {trajectory_name}")

        # Create pose messages for each waypoint
        waypoint_poses = [
            create_pose_stamped(
                origin, x_axis, y_axis, z_axis, dx, dy, dz,
                current_pose.orientation, planning_frame
            )
            for dx, dy, dz in local_waypoints
        ]

        if trajectory_name == "approach":
            force_monitor = ForceZMonitor()
            force_monitor.start()
            stopped_by_force = False
            try:
                for i, waypoint_pose in enumerate(waypoint_poses):
                    plan_result = plan_waypoint(robot, arm, waypoint_pose, tip_link)
                    if not plan_result:
                        logger.error(f"Planning to waypoint {i} failed; aborting")
                        break

                    if force_monitor.triggered:
                        logger.info("Approach force threshold already reached; skipping execution")
                        stopped_by_force = True
                        break

                    if not execute_trajectory_with_force_stop(
                        robot,
                        plan_result.trajectory,
                        force_monitor,
                    ):
                        logger.error(f"Execution to waypoint {i} failed; aborting")
                        break

                    if force_monitor.triggered:
                        logger.info(f"Approach stopped at force threshold after waypoint {i + 1}")
                        stopped_by_force = True
                        break

                    logger.info(f"Reached waypoint {i + 1}/{len(waypoint_poses)}")
            finally:
                force_monitor.stop()

            if stopped_by_force:
                logger.info(f"{trajectory_name} trajectory stopped at force threshold")
            else:
                logger.info(f"{trajectory_name} trajectory completed successfully")
        else:
            for i, waypoint_pose in enumerate(waypoint_poses):
                plan_result = plan_waypoint(robot, arm, waypoint_pose, tip_link)
                if not plan_result:
                    logger.error(f"Planning to waypoint {i} failed; aborting")
                    break

                if not execute_trajectory_with_fallback(robot, plan_result.trajectory):
                    logger.error(f"Execution to waypoint {i} failed; aborting")
                    break

                logger.info(f"Reached waypoint {i + 1}/{len(waypoint_poses)}")
            logger.info(f"{trajectory_name} trajectory completed successfully")

    except Exception as e:
        logger.error(f"Trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
