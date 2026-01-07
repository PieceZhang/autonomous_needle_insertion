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
from typing import List, Tuple

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
    slide_x_waypoints,
    rotate_waypoints,
    standard_action_pose_sequence,
)
from auto_needle_insertion.utils.us_probe import USProbe

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

# # Task42 parameters and tolerances
# TIP_Z_TOL = 1e-3        # 1 mm
# TIP_X_TOL = 1e-3        # 1 mm
# BASE_Z_TOL = 1e-3       # 1 mm
# RY_STEP_DEG = 0.3       # ry fine-tune step
# MAX_ITER_TIP = 8
# MAX_ITER_BASE = 12

STEP5_SWEEP_MM = 20.0    # sweep amplitude for z sweep (positive, mm)
# STEP5_NUM = 10           # number of points (>=2) for z sweep
STEP6_SLIDE_MM = 20.0    # total slide length used to compute x/2 target (mm)
STEP7_ROTATE_DEG = 10.0   # rotation amplitude for ry sweep (deg)
# STEP7_NUM = 5            # number of points (>=2) for ry sweep

TIP_TARGET_X = (STEP6_SLIDE_MM / 1000.0) * 0.5  # meters (x/2 target for step6)

# Task 4.1 standard action parameters
TASK41_TILT_DEG = 10.0        # tilt/fan about X
TASK41_ROCK_DEG = 10.0        # rock about Z
TASK41_SWEEP_MM = 4.0        # sweep along Z (mm)
TASK41_COMPRESSION_MM = 1.0  # compression along Y (mm)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def apply_local_step(to_in_base: np.ndarray, to_in_ee: np.ndarray, step: Tuple[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    T_step = transducer_motions(step[0], step[1])
    to_in_base_new = to_in_base @ T_step
    ee_target = to_in_base_new @ np.linalg.inv(to_in_ee)
    return to_in_base_new, ee_target


def point_in_local(T_local_in_base: np.ndarray, p_in_base: np.ndarray) -> np.ndarray:
    T_base_in_local = np.linalg.inv(T_local_in_base)
    p_h = np.concatenate([p_in_base, [1.0]])
    p_loc = T_base_in_local @ p_h
    return p_loc[:3]


# ---------------------- Closed-loop adjusters (Subtask 2) ----------------------
# def closed_loop_tip_z_zero(
#     robot: MoveItPy,
#     arm,
#     tip_link: str,
#     planning_frame: str,
#     to_in_base: np.ndarray,
#     to_in_ee: np.ndarray,
#     needle: Needle,
#     tol_z: float,
#     max_iter: int,
# ) -> np.ndarray:
#     for it in range(max_iter):
#         needle_pose = needle.report_pose(timeout_sec=2.0)
#         tip_tracker = needle.tip_position_in_tracker(needle_pose)
#         tip_in_to = point_in_local(to_in_base, np.array(tip_tracker))
#         err_z = tip_in_to[2]
#         logger.info(f"[tip z->0] iter {it} err_z={err_z*1000:.2f} mm")
#         if abs(err_z) < tol_z:
#             return to_in_base
#         dz_mm = (-err_z) * 1000.0
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error("Tip z adjust failed")
#             return to_in_base
#     logger.warning("Tip z adjust reached max_iter")
#     return to_in_base
#
#
# def closed_loop_tip_to_x_and_z(
#     robot: MoveItPy,
#     arm,
#     tip_link: str,
#     planning_frame: str,
#     to_in_base: np.ndarray,
#     to_in_ee: np.ndarray,
#     needle: Needle,
#     x_target: float,
#     tol_x: float,
#     tol_z: float,
#     max_iter: int,
# ) -> np.ndarray:
#     for it in range(max_iter):
#         needle_pose = needle.report_pose(timeout_sec=2.0)
#         tip_tracker = needle.tip_position_in_tracker(needle_pose)
#         tip_in_to = point_in_local(to_in_base, np.array(tip_tracker))
#         err_x = tip_in_to[0] - x_target
#         err_z = tip_in_to[2]
#         logger.info(f"[tip x,z] iter {it} err_x={err_x*1000:.2f} mm err_z={err_z*1000:.2f} mm")
#         if abs(err_x) < tol_x and abs(err_z) < tol_z:
#             return to_in_base
#         dx_mm = (-err_x) * 1000.0
#         dz_mm = (-err_z) * 1000.0
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("x", dx_mm))
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error("Tip x adjust failed")
#             return to_in_base
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error("Tip z adjust failed")
#             return to_in_base
#     logger.warning("Tip x,z adjust reached max_iter")
#     return to_in_base
#
#
# def closed_loop_base_z_zero_via_ry(
#     robot: MoveItPy,
#     arm,
#     tip_link: str,
#     planning_frame: str,
#     to_in_base: np.ndarray,
#     to_in_ee: np.ndarray,
#     needle: Needle,
#     tol_z: float,
#     max_iter: int,
#     ry_step_deg: float,
# ) -> np.ndarray:
#     for it in range(max_iter):
#         needle_pose = needle.report_pose(timeout_sec=2.0)
#         base_tracker = np.array(needle_pose[0:3])
#         base_in_to = point_in_local(to_in_base, base_tracker)
#         err_z = base_in_to[2]
#         logger.info(f"[base z->0] iter {it} err_z={err_z*1000:.2f} mm")
#         if abs(err_z) < tol_z:
#             return to_in_base
#         delta_deg = -ry_step_deg if err_z > 0 else ry_step_deg
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("ry", delta_deg))
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error("Base z adjust failed")
#             return to_in_base
#     logger.warning("Base z adjust reached max_iter")
#     return to_in_base


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
    ee_target_pose_in_base_p2: np.ndarray,
):
    """Subtask 1: standard action sequence then return to p1."""
    # Currently at p2
    current_ee_transform = get_current_ee_transform(robot, tip_link)
    to_in_base = current_ee_transform @ to_in_ee  # transducer pose in base

    logger.info("[Subtask 1] Start rosbag recording (placeholder, integrate actual rosbag separately)")

    # Step 5: execute standard action sequence (tilt/fan + rock + sweep + compression)
    probe_poses = standard_action_pose_sequence(
        to_in_base,
        tilt_deg=TASK41_TILT_DEG,
        rock_deg=TASK41_ROCK_DEG,
        sweep_mm=TASK41_SWEEP_MM,
        compression_mm=TASK41_COMPRESSION_MM,
    )
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, probe_poses[1:])
    if not ok:
        logger.error("[Subtask 1] Standard action sequence failed")
        return

    # Step 6: return to p1
    ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
    if not ok:
        logger.error("[Subtask 1] Return to p1 failed")
        return
    logger.info("[Subtask 1] Returned to p1")

    logger.info("[Subtask 1] Stop rosbag recording (placeholder, integrate actual rosbag separately)")


# ---------------------- Subtask 2 workflow ----------------------
# def run_subtask_2(
#     robot: MoveItPy,
#     arm,
#     tip_link: str,
#     planning_frame: str,
#     to_in_ee: np.ndarray,
#     needle: Needle,
#     ee_target_pose_in_base: np.ndarray,
# ):
#     """Subtask 2: dynamic procedure."""
#     # Currently at p2
#     current_ee_transform = get_current_ee_transform(robot, tip_link)
#     to_in_base = current_ee_transform @ to_in_ee
#
#     # Step 5: z sweep
#     _, seq5 = sweep_z_waypoints(np.eye(4), sweep_mm=STEP5_SWEEP_MM, num_points=STEP5_NUM)
#     for i, step in enumerate(seq5):
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, step)
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error(f"Step 5 failed at index {i}")
#             return
#
#     # Closed-loop: tip z -> 0
#     to_in_base = closed_loop_tip_z_zero(
#         robot, arm, tip_link, planning_frame,
#         to_in_base, to_in_ee,
#         needle,
#         tol_z=TIP_Z_TOL, max_iter=MAX_ITER_TIP,
#     )
#
#     # Step 6: tip to (x/2, *, 0)
#     to_in_base = closed_loop_tip_to_x_and_z(
#         robot, arm, tip_link, planning_frame,
#         to_in_base, to_in_ee,
#         needle,
#         x_target=TIP_TARGET_X,
#         tol_x=TIP_X_TOL,
#         tol_z=TIP_Z_TOL,
#         max_iter=MAX_ITER_TIP,
#     )
#
#     # Step 7: ry sweep
#     _, seq7 = rotate_waypoints(np.eye(4), rotate_deg=STEP7_ROTATE_DEG, num_points=STEP7_NUM)
#     for i, step in enumerate(seq7):
#         to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, step)
#         ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#         if not ok:
#             logger.error(f"Step 7 rotation failed at index {i}")
#             return
#
#     # Closed-loop: base z -> 0 via ry micro steps
#     to_in_base = closed_loop_base_z_zero_via_ry(
#         robot, arm, tip_link, planning_frame,
#         to_in_base, to_in_ee,
#         needle,
#         tol_z=BASE_Z_TOL,
#         max_iter=MAX_ITER_BASE,
#         ry_step_deg=RY_STEP_DEG,
#     )
#
#     logger.info("Subtask 2 steps 5-7 completed from p2 toward p1 path.")
#
#     # Final return to p1
#     ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
#     if not ok:
#         logger.error("Final return to p1 failed; aborting")
#         return
#     logger.info("Returned to p1.")


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
    max_step_mm=3.0,
    tol=5e-4,          # 1 mm
    max_iter=50,
):
    """
    Let the needle tip approach to z = 0 in the tracker frame.
    Iteratively compute tip z error
    """
    for _ in range(max_iter):
        tip_in_to = tip_in_to_frame(to_in_base, tracker_in_base, tip_in_tracker)
        err_z = float(tip_in_to[2])
        if abs(err_z) < tol:
            break
        dz_mm = float(np.clip(err_z * 1000.0, -max_step_mm, max_step_mm))
        T_step = transducer_motions("sweep", dz_mm)
        to_target = to_in_base @ T_step
        ee_target = to_target @ np.linalg.inv(to_in_ee)

        # to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
        ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
        to_in_base = to_target
        if not ok:
            logger.error("move_tip_z_to_zero_known: execution failed")
            break
    return to_in_base

# def move_tip_z_to_zero_known(robot, arm, tip_link, planning_frame, to_in_base, to_in_ee, tracker_in_base, tip_in_tracker, max_step_mm=5.0):
#     tip_in_to = tip_in_to_frame(to_in_base, tracker_in_base, tip_in_tracker)
#     dz_mm = float(np.clip(-tip_in_to[2] * 1000.0, -max_step_mm, max_step_mm))
#     to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
#     plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
#     return to_in_base

def move_tip_to_x_over_2_known(robot, arm, tip_link, planning_frame, to_in_base, to_in_ee, tracker_in_base, tip_in_tracker, x_target, max_step_mm=5.0):
    tip_in_to = tip_in_to_frame(to_in_base, tracker_in_base, tip_in_tracker)
    dx_mm = float(np.clip((x_target - tip_in_to[0]) * 1000.0, -max_step_mm, max_step_mm))
    dz_mm = float(np.clip(-tip_in_to[2] * 1000.0, -max_step_mm, max_step_mm))
    to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("x", dx_mm))
    plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
    plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    return to_in_base

def rotate_base_z_to_zero_known(robot, arm, tip_link, planning_frame, to_in_base, to_in_base_p1, to_in_ee, 
                                tracker_in_base, needle_pose_tracker,
                                ry_step_deg=0.4, tol_z=5e-4, max_iter=50):
    x_ref = to_in_base_p1[0:3, 0]
    for _ in range(max_iter):
        base_in_to = base_in_to_frame(to_in_base, tracker_in_base, needle_pose_tracker)
        err_z = float(base_in_to[2])
        if abs(err_z) < tol_z:
            break
        x_cur = to_in_base[0:3, 0]
        dot1 = float(np.dot(x_ref, x_cur))
        dot1 = max(min(dot1, 1.0), -1.0)  
        if err_z < 0 and dot1 > 0:
            delta_deg = ry_step_deg
        else:
            delta_deg = -ry_step_deg
        T_step = transducer_motions("rotation", delta_deg)
        to_target = to_in_base @ T_step
        ee_target = to_target @ np.linalg.inv(to_in_ee)

        # to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("z", dz_mm))
        ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
        to_in_base = to_target
        if not ok:
            logger.error("move_tip_z_to_zero_known: execution failed")
            break
        # to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, ("ry", delta_deg))
        # plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target)
    return to_in_base

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
    sweep_mm: float = STEP5_SWEEP_MM,
    rotate_deg: float = STEP7_ROTATE_DEG,
):
    # Current transducer pose in base
    current_ee_transfrorm = get_current_ee_transform(robot, tip_link)
    to_in_base = current_ee_transfrorm @ to_in_ee
    logger.info("----------Step 5-1: sweep along local z axis-------------")
    # Sweep -z to +z
    poses_sweep = sweep_z_waypoints(to_in_base, sweep_mm=sweep_mm)
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, poses_sweep[1:])
    if not ok:
        logger.error("[known] z sweep failed")
        return to_in_base
    to_in_base = poses_sweep[-1]
    logger.info("Step 5-1 finished")

    # _, seq = sweep_z_waypoints(np.eye(4), sweep_mm=sweep_mm, num_points=sweep_num)
    # for i, step in enumerate(seq):
    #     to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, step)
    #     if not plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target):
    #         logger.error(f"[known] z sweep failed at index {i}")
    #         return to_in_base

    # Tip to ( *, *, 0 )
    logger.info("-------- Step 5-2: sweep to let tip z to 0--------")
    to_in_base = move_tip_z_to_zero_known(
        robot, arm, tip_link, planning_frame,
        to_in_base, to_in_ee,
        tracker_in_base,
        needle_tip_position,
    )
    logger.info("Step 5-2 finished")

    # Tip to ( x/2, *, 0 )
    logger.info("-------- Step 6: slide to (x/2, *, 0)) -----------")
    poses_slide = slide_x_waypoints(to_in_base, slide_mm=STEP6_SLIDE_MM/2)
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, poses_slide[1:])
    if ok:
        to_in_base = poses_slide[-1]
    logger.info("Step 6 finished")

    # to_in_base = move_tip_to_x_over_2_known(
    #     robot, arm, tip_link, planning_frame,
    #     to_in_base, to_in_ee,
    #     tracker_in_base,
    #     needle_tip_position,
    #     x_target=x_target,
    # )

    # Rotate -theta to +theta
    logger.info("--------- Step 7-1: rotate around y axis---------")
    poses_rotate = rotate_waypoints(to_in_base, rotate_deg=rotate_deg)
    ok = execute_probe_pose_sequence(robot, arm, tip_link, planning_frame, to_in_ee, poses_rotate[1:])
    if not ok:
        logger.error("[known] ry sweep failed")
        return to_in_base
    to_in_base = poses_rotate[-1]
    logger.info("Step 7-1 finished")

    # _, seq7 = rotate_waypoints(np.eye(4), rotate_deg=rotate_deg, num_points=rotate_num)
    # for i, step in enumerate(seq7):
    #     to_in_base, ee_target = apply_local_step(to_in_base, to_in_ee, step)
    #     if not plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target):
    #         logger.error(f"[known] ry sweep failed at index {i}")
    #         return to_in_base

    # Adjust needle origin z to 0 via ry
    logger.info("--------- Step 7-2: rotate to let base z to 0--------")
    to_in_base = rotate_base_z_to_zero_known(
        robot, arm, tip_link, planning_frame,
        to_in_base, to_in_base_p1, to_in_ee,
        tracker_in_base,
        needle_pose,
    )
    logger.info("Step 7-2 finished")
    # return to_in_base


# ---------------------- Main ----------------------
def main() -> None:
    rclpy.init()

    if len(sys.argv) > 1:
        task_mode = sys.argv[1].strip().lower()
    logger.info(f"Selected task mode: {task_mode} ('one' -> Subtask 1, 'two' -> Subtask 2)")

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

        # Frames
        tracker_in_base = current_ee_transform @ probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))
        to_in_tracker = quat_to_T(probe_pose) @ to_in_probe

        # Align and center image frame
        image_in_tracker_after_alignment = align_image_to_needle_axis(
            to_in_tracker, needle_pose[0:3], needle_tip_position
        )
        image_in_tracker_after_centering = center_needle_in_image(
            image_in_tracker_after_alignment, needle_pose[0:3], needle_tip_position,
            x_center_in_plane=0.0, y_target_in_plane=0.1
        )

        # Apply small random perturbations to get candidate image pose (p2)
        candidate_image_in_tracker = None
        rng = np.random.default_rng()
        for attempt in range(1, MAX_PERTURBATION_TRIALS + 1):
            seed = int(rng.integers(0, 2**32 - 1))
            poses, _ = apply_random_small_perturbation(
                image_in_tracker_after_centering,
                rot_range_deg=(-10.0, 10.0),
                sweep_range_mm=(-20.0, 20.0),
                slide_range_mm=(-20.0, 20.0),
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

        # Plan and execute to p1
        ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
        if not ok:
            logger.error("Execution to p1 failed; aborting")
            return
        logger.info("Reached target pose p1")
        to_in_base_p1 = ee_target_pose_in_base @ to_in_ee

        # Plan and execute to p2
        ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base_p2)
        if not ok:
            logger.error("Execution to p2 failed; aborting")
            return
        logger.info("Reached target pose p2")

        # run_subtask_1(
        #         robot, arm, tip_link, planning_frame,
        #         to_in_ee,
        #         ee_target_pose_in_base,
        #         ee_target_pose_in_base_p2,
        # )

        logger.info('start task 4.2')

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
        )

        # Final return to p1
        # ok = plan_and_execute_pose(robot, arm, tip_link, planning_frame, ee_target_pose_in_base)
        # if not ok:
        #     logger.error("Final return to p1 failed; aborting")
        #     return
        # logger.info("Returned to p1.")

    except Exception as e:
        logger.error(f"Trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

