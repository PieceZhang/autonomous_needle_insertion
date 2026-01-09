"""End-effector needle-finding and task4 execution module (Subtask 1 & Subtask 2).

Supported modes:
    - Subtask 1 (mode: "one"):
        1) Done: compute and move to p1
        2) Done: apply random perturbation, get p2, move to p2
        3) Execute standard_action_pose_sequence (tilt/fan + rock + sweep + compression)
           defined in transducer_motions.py
        4) Ensure returning to p2 after the sequence
        5) Move from p2 back to p1

    - Subtask 2 (mode: "two"):
        1) Done: compute and move to p1
        2) Done: apply random perturbation, get p2, move to p2
        3) From p2, run task42 steps 5-7 (z sweep, closed-loop tip z, closed-loop tip x/z,
           ry sweep, closed-loop base z)
        4) Finally move back to p1

Usage:
    TASK_MODE environment variable or first CLI argument selects mode:
        "one" -> Subtask 1, "two" -> Subtask 2 (default: "two")
"""

import logging
import os
import sys
import time
import threading
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_link_constraint
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.node import Node

from auto_needle_insertion.utils.needle import Needle
from auto_needle_insertion.utils.optical_tracking import read_instrument_pose
from auto_needle_insertion.utils.find_needle import (
    align_image_to_needle_axis,
    center_needle_in_image,
    needle_segment_in_image,
)
from auto_needle_insertion.utils.pose_representations import (
    homogeneous_to_pose_stamped,
    quat_to_T,
)
from auto_needle_insertion.utils.transducer_motions import (
    apply_random_small_perturbation,
    random_small_perturbation_sequence,
    transducer_motions,
    sweep_z_waypoints,
    rotate_waypoints,
    standard_action_pose_sequence,
)
from auto_needle_insertion.utils.us_probe import USProbe
from std_msgs.msg import String
from rclpy.executors import SingleThreadedExecutor
from auto_needle_insertion.rosbag_recorder_control import (
    TaskInfoPublisher,
    RosbagController,
    sleep_with_spin,
)

# Module constants
NODE_NAME = "auto_needle_insertion"
SQUARE_EDGE_LENGTH = 0.2  # meters
MAX_VELOCITY_SCALING = 0.2
MAX_ACCELERATION_SCALING = 0.2
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds
MAX_PERTURBATION_TRIALS = 20
IMAGE_WIDTH_PX = 1920
IMAGE_HEIGHT_PX = 1080

# Controller fallback order
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",  # UR hardware typical
    "",  # Default controller
    "joint_trajectory_controller"  # Simulation/common
]

# Preferred tip link names in order of preference
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

# Random perturbation parameters
P2_ROT_RANGE_DEG = (-10.0, 10.0)  # deg
P2_SWEEP_RANGE_MM = (-20.0, 20.0)  # mm
P2_SLIDE_RANGE_MM = (-20.0, 20.0)  # mm

# Target position in image plane for needle centering (in meters)
Y_TARGET_IN_PLANE_M = 0.07

STEP5_SWEEP_MM = 20.0    # sweep amplitude for z sweep (positive, mm)
STEP6_SLIDE_MM = 20.0    # total slide length used to compute x/2 target (mm)
STEP7_ROTATE_DEG = 10.0   # rotation amplitude for ry sweep (deg)

# Task 4.1 standard action parameters
TASK41_TILT_DEG = 10.0        # tilt/fan about X
TASK41_ROCK_DEG = 10.0        # rock about Z
TASK41_SWEEP_MM = 25.0        # sweep along Z (mm)
# TASK41_COMPRESSION_MM = 5.0  # compression along Y (mm)

DELAY_AFTER_ROSBAG_SEC = 0.5
ROSBAG_STOP_WAIT_SEC = 0.5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskProcedurePublisher(rclpy.node.Node):
    """Publish step changes on 'task_procedure'."""
    def __init__(self, topic_name: str = "task_procedure") -> None:
        super().__init__("task_procedure_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._last: Optional[str] = None

    def publish_step(self, step: str) -> None:
        if step != self._last:
            self._last = step
            msg = String()
            msg.data = step
            self._pub.publish(msg)


class _SpinThread:
    """Background spinner for rclpy executor."""
    def __init__(self, executor: SingleThreadedExecutor) -> None:
        self._exec = executor
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            self._exec.spin_once(timeout_sec=0.1)


# ---------------------- Planning helpers ----------------------
def get_planning_group_name(robot: MoveItPy) -> str:
    group_names = robot.get_robot_model().joint_model_group_names
    if not group_names:
        raise RuntimeError("No planning groups available")
    logger.info(f"Available planning groups: {group_names}")
    for group_name in group_names:
        if "manipulator" in group_name or "ur" in group_name:
            return group_name
    return group_names[0]


def get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    group = robot.get_robot_model().get_joint_model_group(group_name)
    if not group:
        raise RuntimeError(f"Planning group '{group_name}' not found")
    link_names = list(group.link_model_names)
    if not link_names:
        raise RuntimeError(f"No links found in planning group '{group_name}'")
    for preferred_link in PREFERRED_TIP_LINKS:
        if preferred_link in link_names:
            return preferred_link
    return link_names[-1]


def execute_trajectory_with_fallback(
    robot: MoveItPy,
    trajectory,
    controllers: List[str] = CONTROLLER_NAMES
) -> bool:
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


def plan_and_execute_pose(
    robot: MoveItPy,
    arm,
    tip_link: str,
    planning_frame: str,
    T_target: np.ndarray,
    vel_scale: float = MAX_VELOCITY_SCALING,
    acc_scale: float = MAX_ACCELERATION_SCALING,
) -> bool:
    pose_goal = homogeneous_to_pose_stamped(T_target, planning_frame)
    pos = pose_goal.pose.position
    ori = pose_goal.pose.orientation
    goal_c = construct_link_constraint(
        link_name=tip_link,
        source_frame=planning_frame,
        cartesian_position=[pos.x, pos.y, pos.z],
        cartesian_position_tolerance=1e-4,
        orientation=[ori.x, ori.y, ori.z, ori.w],
        orientation_tolerance=1e-4,
    )
    arm.set_start_state_to_current_state()
    arm.set_goal_state(motion_plan_constraints=[goal_c])
    params = PlanRequestParameters(robot, "")
    params.max_velocity_scaling_factor = vel_scale
    params.max_acceleration_scaling_factor = acc_scale
    plan_result = arm.plan(single_plan_parameters=params)
    if not plan_result:
        logger.error("Planning failed")
        return False
    if not execute_trajectory_with_fallback(robot, plan_result.trajectory):
        logger.error("Execution failed")
        return False
    return True


def get_current_ee_transform(robot: MoveItPy, tip_link: str) -> np.ndarray:
    with robot.get_planning_scene_monitor().read_only() as scene:
        scene.current_state.update()
        return scene.current_state.get_global_link_transform(tip_link)


def point_in_local(T_local_in_base: np.ndarray, p_in_base: np.ndarray) -> np.ndarray:
    T_base_in_local = np.linalg.inv(T_local_in_base)
    p_h = np.concatenate([p_in_base, [1.0]])
    p_loc = T_base_in_local @ p_h
    return p_loc[:3]


def execute_probe_pose_sequence(
    robot: MoveItPy,
    arm,
    tip_link: str,
    planning_frame: str,
    to_in_ee: np.ndarray,
    probe_poses: List[np.ndarray],
) -> bool:
    """Execute a list of probe poses (in base frame) sequentially."""
    for i, T_probe in enumerate(probe_poses):
        ee_target = T_probe @ np.linalg.inv(to_in_ee)
        ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
        if not ok:
            logger.error(f"Probe pose sequence failed at index {i}")
            return False
    return True


# ---------------------- Subtask 1 workflow ----------------------
def run_subtask_1(
    robot: MoveItPy,
    arm,
    tip_link: str,
    planning_frame: str,
    to_in_ee: np.ndarray,
    ee_target_pose_in_base: np.ndarray,
    task_proc_pub: TaskProcedurePublisher,
):
    task_proc_pub.publish_step("subtask1_start")
    """Subtask 1: standard action sequence then return to p1."""
    # Currently at p2
    current_ee_transform = get_current_ee_transform(robot, tip_link)
    to_in_base = current_ee_transform @ to_in_ee  # transducer pose in base

    task_proc_pub.publish_step("subtask1_standard_action")
    # Step 5: execute standard action sequence (tilt/fan + rock + sweep + compression)
    probe_poses = standard_action_pose_sequence(
        to_in_base,
        tilt_deg=TASK41_TILT_DEG,
        rock_deg=TASK41_ROCK_DEG,
        sweep_mm=TASK41_SWEEP_MM,
        # compression_mm=TASK41_COMPRESSION_MM,
    )
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, probe_poses[1:])
    if not ok:
        logger.error("[Subtask 1] Standard action sequence failed")
        task_proc_pub.publish_step("subtask1_failed")
        return

    # Step 6: return to p1
    task_proc_pub.publish_step("subtask1_return_p1")
    ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
    if not ok:
        logger.error("[Subtask 1] Return to p1 failed")
        task_proc_pub.publish_step("subtask1_failed")
        return
    logger.info("[Subtask 1] Returned to p1")
    task_proc_pub.publish_step("subtask1_done")


def tip_in_to_frame(to_in_base: np.ndarray, tracker_in_base: np.ndarray, tip_in_tracker: np.ndarray) -> np.ndarray:
    tip_in_base_h = tracker_in_base @ np.concatenate([tip_in_tracker, [1.0]])
    return point_in_local(to_in_base, tip_in_base_h[:3])

def base_in_to_frame(to_in_base: np.ndarray, tracker_in_base: np.ndarray, needle_pose_tracker: np.ndarray) -> np.ndarray:
    base_tracker = np.array(needle_pose_tracker[0:3])
    base_in_base_h = tracker_in_base @ np.concatenate([base_tracker, [1.0]])
    return point_in_local(to_in_base, base_in_base_h[:3])

def move_tip_z_to_zero_known(
    robot, arm, tip_link, planning_frame,
    to_in_base, to_in_ee,
    tracker_in_base, tip_in_tracker,
):
    """
    Let the needle tip approach to z = 0 in the tracker frame.
    """
    tip_in_to = tip_in_to_frame(to_in_base, tracker_in_base, tip_in_tracker)
    err_z = float(tip_in_to[2])
    T_step = transducer_motions("sweep", err_z*1000.0)
    to_target = to_in_base @ T_step
    ee_target = to_target @ np.linalg.inv(to_in_ee)

    ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    to_in_base = to_target
    if not ok:
        logger.error("move_tip_z_to_zero_known: execution failed")
    return to_in_base


def move_tip_x_to_image_center(robot, arm, tip_link, planning_frame,
    to_in_base, to_in_ee,
    tracker_in_base, tip_in_tracker
):
    """
    Let the needle tip approach to x = 0 in the tracker frame.
    """
    tip_in_to = tip_in_to_frame(to_in_base, tracker_in_base, tip_in_tracker)
    err_x = float(tip_in_to[0])
    T_step = transducer_motions("slide", err_x*1000.0)
    to_target = to_in_base @ T_step
    ee_target = to_target @ np.linalg.inv(to_in_ee)
    ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    to_in_base = to_target
    if not ok:
        logger.error("move_tip_z_to_zero_known: execution failed")
    return to_in_base


def rotate_base_z_to_zero_known(robot, arm, tip_link, planning_frame, to_in_base, to_in_base_p1, to_in_ee, 
                                tracker_in_base, needle_pose_tracker):
    x_ref = to_in_base_p1[0:3, 0]
    base_in_to = base_in_to_frame(to_in_base, tracker_in_base, needle_pose_tracker)
    err_z = float(base_in_to[2])
    x_cur = to_in_base[0:3, 0]
    dot1 = float(np.dot(x_ref, x_cur))
    angle = abs(np.arccos(dot1/(np.linalg.norm(x_ref)*np.linalg.norm(x_cur))))
    dot1 = max(min(dot1, 1.0), -1.0)  
    if err_z < 0 and dot1 > 0:
        delta_deg = angle * (180.0/np.pi)
    else:
        delta_deg = -angle * (180.0/np.pi)
    T_step = transducer_motions("rotation", delta_deg)
    to_target = to_in_base @ T_step
    ee_target = to_target @ np.linalg.inv(to_in_ee)
    ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    to_in_base = to_target
    if not ok:
        logger.error("move_tip_z_to_zero_known: execution failed")
    return to_in_base


# ---------------------- Subtask 2 workflow ----------------------
def run_subtask_2(
    robot,
    arm,
    tip_link: str,
    planning_frame: str,
    to_in_ee: np.ndarray,
    to_in_base_p1: np.ndarray,
    tracker_in_base: np.ndarray,
    needle_tip_position: np.ndarray,
    needle_pose: np.ndarray,
    task_proc_pub: TaskProcedurePublisher,
    sweep_mm: float = STEP5_SWEEP_MM,
    rotate_deg: float = STEP7_ROTATE_DEG,
):
    # Current transducer pose in base
    current_ee_transfrorm = get_current_ee_transform(robot, tip_link)
    to_in_base = current_ee_transfrorm @ to_in_ee
    task_proc_pub.publish_step("subtask2_start")
    task_proc_pub.publish_step("subtask2_step5-1")
    logger.info("----------Step 5-1: sweep along local z axis-------------")
    # Sweep -z to +z
    poses_sweep = sweep_z_waypoints(to_in_base, sweep_mm=sweep_mm)
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, poses_sweep[1:])
    if not ok:
        logger.error("[known] z sweep failed")
        task_proc_pub.publish_step("subtask2_failed")
        return to_in_base
    to_in_base = poses_sweep[-1]
    logger.info("Step 5-1 finished")

    # Tip to ( *, *, 0 )
    task_proc_pub.publish_step("subtask2_step5-2")
    logger.info("-------- Step 5-2: sweep to let tip z to 0--------")
    to_in_base = move_tip_z_to_zero_known(
        robot, arm, tip_link, planning_frame,
        to_in_base, to_in_ee,
        tracker_in_base,
        needle_tip_position,
    )
    logger.info("Step 5-2 finished")

    # Tip to ( x/2, *, 0 )
    task_proc_pub.publish_step("subtask2_step6")
    logger.info("-------- Step 6: slide to (x/2, *, 0)) -----------")
    to_in_base = move_tip_x_to_image_center(
        robot, arm, tip_link, planning_frame,
        to_in_base, to_in_ee,
        tracker_in_base,
        needle_tip_position
    )
    logger.info("Step 6 finished")

    # Rotate -theta to +theta
    task_proc_pub.publish_step("subtask2_step7-1")
    logger.info("--------- Step 7-1: rotate around y axis---------")
    poses_rotate = rotate_waypoints(to_in_base, rotate_deg=rotate_deg)
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, poses_rotate[1:])
    if not ok:
        logger.error("[known] ry sweep failed")
        task_proc_pub.publish_step("subtask2_failed")
        return to_in_base
    to_in_base = poses_rotate[-1]
    logger.info("Step 7-1 finished")

    # Adjust needle origin z to 0 via ry
    task_proc_pub.publish_step("subtask2_step7-2")
    logger.info("--------- Step 7-2: rotate to let base z to 0--------")
    to_in_base = rotate_base_z_to_zero_known(
        robot, arm, tip_link, planning_frame,
        to_in_base, to_in_base_p1, to_in_ee,
        tracker_in_base,
        needle_pose,
    )
    logger.info("Step 7-2 finished")
    task_proc_pub.publish_step("subtask2_done")


# ---------------------- Main ----------------------
def main() -> None:
    rclpy.init()

    executor = SingleThreadedExecutor()
    task_info_pub = TaskInfoPublisher(topic_name="task_info_collection_states")
    task_proc_pub = TaskProcedurePublisher(topic_name="task_procedure")
    executor.add_node(task_info_pub)
    executor.add_node(task_proc_pub)
    spinner = _SpinThread(executor)
    spinner.start()

    rosbag_controller = RosbagController()
    rosbag_active = False

    def start_rosbag_recording() -> None:
        nonlocal rosbag_active
        if rosbag_active:
            return
        logger.info("Starting rosbag recording before move to p1")
        task_info_pub.set_state("started")
        rosbag_controller.start_recording()
        sleep_with_spin(executor, DELAY_AFTER_ROSBAG_SEC)
        rosbag_active = True

    def stop_rosbag_recording(success: bool) -> None:
        nonlocal rosbag_active
        if not rosbag_active:
            return
        state = "stopped_success" if success else "stopped_failure"
        reason = "Success" if success else "Failure"
        task_info_pub.set_state(state)
        sleep_with_spin(executor, ROSBAG_STOP_WAIT_SEC)
        rosbag_controller.stop_recording(reason)
        rosbag_active = False

    # Replace task_mode selection with TASK4_SUBTASK env var
    _subtask_raw = (os.getenv("TASK4_SUBTASK", "") or "").strip()
    if _subtask_raw == "":
        task_subtask = 1
    else:
        try:
            task_subtask = int(_subtask_raw)
        except ValueError as e:
            raise ValueError(f"TASK4_SUBTASK must be '1' or '2' (got {_subtask_raw!r})") from e
    if task_subtask not in (1, 2):
        raise ValueError(f"TASK4_SUBTASK must be '1' or '2' (got {task_subtask})")

    logger.info(f"Selected TASK4_SUBTASK={task_subtask} (1 -> Subtask 1, 2 -> Subtask 2)")

    try:
        task_info_pub.set_state("started")
        task_proc_pub.publish_step("task4_init")

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

        # Load probe calibrations
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
        if us_probe.top_in_to is None:
            raise RuntimeError("US probe calibration missing TransducerOriginPixel calibration.")
        probe_pose = us_probe.report_pose(timeout_sec=2.0)

        # Load needle
        needle = Needle()
        needle.load_tip_offset("./calibration/needle_1_tip_offset.json")
        needle_pose = needle.report_pose(timeout_sec=2.0)
        needle_tip_position = needle.tip_position_in_tracker(needle_pose)

        # Image metrics
        pixel_spacing_x = abs(float(us_probe.top_in_to[0, 0])) / 1000.0
        pixel_spacing_y = abs(float(us_probe.top_in_to[1, 1])) / 1000.0
        image_width_m = IMAGE_WIDTH_PX * pixel_spacing_x
        image_height_m = IMAGE_HEIGHT_PX * pixel_spacing_y

        while True:
            # Frames
            tracker_in_base = current_ee_transform @ probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))
            to_in_tracker = quat_to_T(probe_pose) @ to_in_probe

            # Align and center image frame
            image_in_tracker_after_alignment = align_image_to_needle_axis(
                to_in_tracker, needle_pose[0:3], needle_tip_position
            )
            image_in_tracker_after_centering = center_needle_in_image(
                image_in_tracker_after_alignment, needle_pose[0:3], needle_tip_position,
                x_center_in_plane=0.0, y_target_in_plane=Y_TARGET_IN_PLANE_M
            )

            # Apply small random perturbations to get candidate image pose (p2)
            candidate_image_in_tracker = None
            rng = np.random.default_rng()
            for attempt in range(1, MAX_PERTURBATION_TRIALS + 1):
                seed = int(rng.integers(0, 2**32 - 1))
                poses, _ = apply_random_small_perturbation(
                    image_in_tracker_after_centering,
                    rot_range_deg=P2_ROT_RANGE_DEG,
                    sweep_range_mm=P2_SWEEP_RANGE_MM,
                    slide_range_mm=P2_SLIDE_RANGE_MM,
                    rng=np.random.default_rng(seed),
                )
                if not poses:
                    logger.warning("No perturbation poses generated; retrying.")
                    continue
                candidate_image_in_tracker = poses[-1]
                needle_visible = needle_segment_in_image(
                    candidate_image_in_tracker,
                    needle_pose[0:3],
                    needle_tip_position,
                    image_width=image_width_m,
                    image_height=image_height_m,
                    position_unit="m",
                )
                if needle_visible:
                    logger.info(f"Accepted perturbation sequence")
                    break
                logger.info(f"Rejected perturbation sequence")

            if candidate_image_in_tracker is None:
                raise RuntimeError("Failed to find a valid perturbation with the needle in view.")

            # p1
            ee_target_pose_in_base = tracker_in_base @ image_in_tracker_after_centering @ np.linalg.inv(to_in_ee)
            logger.info(f"GT pose p1 (EE in base):\n{ee_target_pose_in_base}")

            # p2
            ee_target_pose_in_base_p2 = tracker_in_base @ candidate_image_in_tracker @ np.linalg.inv(to_in_ee)
            logger.info(f"Random pose p2 (EE in base):\n{ee_target_pose_in_base_p2}")

            task_proc_pub.publish_step("move_p1")
            # Plan and execute to p1
            ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
            if not ok:
                logger.error("Execution to p1 failed; aborting")
                stop_rosbag_recording(success=False)
                return
            logger.info("Reached target pose p1")
            task_proc_pub.publish_step("p1_reached")
            to_in_base_p1 = ee_target_pose_in_base @ to_in_ee

            task_proc_pub.publish_step("move_p2")
            # Plan and execute to p2
            ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base_p2)
            if not ok:
                logger.error("Execution to p2 failed; aborting")
                stop_rosbag_recording(success=False)
                return
            logger.info("Reached target pose p2")
            task_proc_pub.publish_step("p2_reached")

            # Start rosbag recording after reaching p2
            print('Starting rosbag')
            start_rosbag_recording()

            if task_subtask == 1:
                logger.info("start task 4.1")
                print("start task 4.1")
                run_subtask_1(
                    robot, arm, tip_link, planning_frame,
                    to_in_ee,
                    ee_target_pose_in_base,
                    task_proc_pub,
                )
            elif task_subtask == 2:
                logger.info("start task 4.2")
                print("start task 4.2")
                run_subtask_2(
                    robot=robot,
                    arm=arm,
                    tip_link=tip_link,
                    planning_frame=planning_frame,
                    to_in_ee=to_in_ee,
                    to_in_base_p1=to_in_base_p1,
                    tracker_in_base=tracker_in_base,
                    needle_tip_position=needle_tip_position,
                    needle_pose=needle_pose,
                    task_proc_pub=task_proc_pub,
                )

            stop_rosbag_recording(success=True)
            task_info_pub.set_state("Success")

    except Exception as e:
        logger.error(f"Trajectory execution failed: {e}")
        stop_rosbag_recording(success=False)
        task_info_pub.set_state("Failure")
        raise
    finally:
        spinner.stop()
        executor.shutdown()
        task_info_pub.destroy_node()
        task_proc_pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

