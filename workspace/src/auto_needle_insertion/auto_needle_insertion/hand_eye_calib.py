#!/usr/bin/env python3
"""
End-effector pose sequence execution + pose logging.

This module commands a UR5e (MoveIt 2 + MoveItPy on ROS 2 Jazzy) through
a series of small, cumulative pose deltas expressed in the EE local frame,
and logs the *achieved* end-effector poses (in the planning frame) to CSV.

Motion-safety intent:
  - Small per-step deltas (<= 3 cm translation, <= ~5° orientation by default)
  - Conservative velocity/acceleration scaling
  - Controller fallback (scaled_joint_trajectory_controller -> default -> joint_trajectory_controller)

Logging:
  - Achieved robot EE poses are kept in-memory as an array shaped (N, 7): [x, y, z, qx, qy, qz, qw].

References:
  - MoveItPy Motion Planning Python API and PlanRequestParameters fields.
  - Orientation/path constraints are *not* used here to keep planning simple and robust,
    but can be added later if you prefer constraint-based planning.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit.planning import MoveItPy, PlanRequestParameters

# ---------------------- Module constants ----------------------

NODE_NAME = "ee_pose_sequence_logger"

# Conservative planning scales
MAX_VELOCITY_SCALING = 0.20
MAX_ACCELERATION_SCALING = 0.20

# Allow time for the planning scene to sync joint states
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds

# Controller fallback order (hardware -> default -> sim/common)
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",
    "",
    "joint_trajectory_controller",
]

# Preferred tip link names in order of preference
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

# ---------------------- Logging ----------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Math utilities ----------------------

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert RPY (rad) to quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)

def _quat_multiply(q1: Tuple[float, float, float, float],
                   q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Hamilton product q = q1 ⊗ q2, (x,y,z,w) convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    # Normalize to be safe
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/n, y/n, z/n, w/n)

def _quat_to_rot(q: Tuple[float, float, float, float]) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [2*(xy+wz),         1 - 2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),         2*(yz+wx),       1 - 2*(xx+yy)],
    ], dtype=float)

def _array_to_quat_msg(q: Tuple[float, float, float, float]) -> Quaternion:
    x, y, z, w = q
    msg = Quaternion()
    msg.x = float(x); msg.y = float(y); msg.z = float(z); msg.w = float(w)
    return msg

# ---------------------- MoveIt helpers ----------------------

def _get_planning_group_name(robot: MoveItPy) -> str:
    names = robot.get_robot_model().joint_model_group_names
    if not names:
        raise RuntimeError("No planning groups available")
    logger.info(f"Available planning groups: {names}")
    for g in names:
        if "manipulator" in g or "ur" in g:
            return g
    return names[0]

def _get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    group = robot.get_robot_model().get_joint_model_group(group_name)
    links = list(group.link_model_names) if group else []
    if not links:
        raise RuntimeError(f"No links in planning group '{group_name}'")
    for preferred in PREFERRED_TIP_LINKS:
        if preferred in links:
            return preferred
    return links[-1]

def _execute_with_fallback(robot: MoveItPy, trajectory, controllers: List[str]) -> bool:
    for c in controllers:
        try:
            if c:
                robot.execute(trajectory, controllers=[c])
            else:
                robot.execute(trajectory)
            return True
        except Exception as e:
            logger.warning(f"Controller '{c}' failed: {e}")
    logger.error("All controllers failed")
    return False

# ---------------------- Pose sequence definition ----------------------

@dataclass
class LocalDelta:
    """Pose delta in the *current EE local frame*."""
    dx: float; dy: float; dz: float      # meters
    droll: float; dpitch: float; dyaw: float  # radians

def _default_local_deltas() -> List[LocalDelta]:
    """Return a well-spread sequence of gentle pose deltas for calibration.

    Design goals (per step):
      - small translations (≈ 2 cm) and small rotations (≈ 3°)
      - balanced coverage around all axes and plane diagonals
      - interleaved signs to avoid large cumulative drift

    Returns ~30 deltas mixing pure translations, pure rotations, and
    small translation+rotation combos.
    """
    deg = math.radians
    t = 0.02           # 2 cm per step (local frame)
    r = deg(3.0)       # ~3 degrees per step (local frame)

    deltas: List[LocalDelta] = []

    # 1) Axis translations (±X, ±Y, ±Z) — 6
    deltas += [
        LocalDelta(+t, 0.0, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, 0.0, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, -t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, 0.0, +t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, 0.0, -t, 0.0, 0.0, 0.0),
    ]

    # 2) Diagonals in XY plane — 4
    deltas += [
        LocalDelta(+t, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, -t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(+t, -t, 0.0, 0.0, 0.0, 0.0),
    ]

    # 3) Diagonals in XZ and YZ planes — 8
    deltas += [
        LocalDelta(+t, 0.0, +t, 0.0, 0.0, 0.0),
        LocalDelta(-t, 0.0, +t, 0.0, 0.0, 0.0),
        LocalDelta(-t, 0.0, -t, 0.0, 0.0, 0.0),
        LocalDelta(+t, 0.0, -t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, +t, +t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, -t, +t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, -t, -t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, +t, -t, 0.0, 0.0, 0.0),
    ]

    # 4) Small pure rotations about local axes — 6
    deltas += [
        LocalDelta(0.0, 0.0, 0.0, +r, 0.0, 0.0),
        LocalDelta(0.0, 0.0, 0.0, -r, 0.0, 0.0),
        LocalDelta(0.0, 0.0, 0.0, 0.0, +r, 0.0),
        LocalDelta(0.0, 0.0, 0.0, 0.0, -r, 0.0),
        LocalDelta(0.0, 0.0, 0.0, 0.0, 0.0, +r),
        LocalDelta(0.0, 0.0, 0.0, 0.0, 0.0, -r),
    ]

    # 5) Translation + rotation combos (orthogonal axes) — 6
    deltas += [
        LocalDelta(+t, 0.0, 0.0, 0.0, 0.0, +r),
        LocalDelta(0.0, +t, 0.0, +r, 0.0, 0.0),
        LocalDelta(0.0, 0.0, +t, 0.0, +r, 0.0),
        LocalDelta(-t, 0.0, 0.0, 0.0, 0.0, -r),
        LocalDelta(0.0, -t, 0.0, -r, 0.0, 0.0),
        LocalDelta(0.0, 0.0, -t, 0.0, -r, 0.0),
    ]

    return deltas

# ---------------------- Core routine ----------------------

def main() -> None:
    rclpy.init()

    try:
        robot = MoveItPy(node_name=NODE_NAME)

        # Allow time for joint states to sync into the planning scene
        time.sleep(PLANNING_SCENE_SYNC_DELAY)

        psm = robot.get_planning_scene_monitor()
        with psm.read_write() as scene_rw:
            scene_rw.current_state.update()

        # Get planning frame
        with psm.read_only() as scene_ro:
            planning_frame = scene_ro.planning_frame
        logger.info(f"Planning frame: {planning_frame}")

        # Setup planning component
        group_name = _get_planning_group_name(robot)
        tip_link = _get_tip_link_name(robot, group_name)
        arm = robot.get_planning_component(group_name)
        logger.info(f"Using planning group: {group_name}")
        logger.info(f"Using tip link: {tip_link}")

        # Initial state and transform
        arm.set_start_state_to_current_state()
        with psm.read_only() as scene:
            scene.current_state.update()
            T0 = scene.current_state.get_global_link_transform(tip_link)  # 4x4
            current_pose = scene.current_state.get_pose(tip_link)

        # Extract origin and current orientation
        origin = T0[:3, 3].copy()
        q_cur = (current_pose.orientation.x, current_pose.orientation.y,
                 current_pose.orientation.z, current_pose.orientation.w)
        R_cur = _quat_to_rot(q_cur)

        # Build a sequence of small, cumulative targets in world frame
        local_deltas = _default_local_deltas()
        targets: List[PoseStamped] = []

        pos_cur = origin.copy()
        quat_cur = q_cur
        R_world_from_local = R_cur.copy()

        for d in local_deltas:
            # translate in *current local* frame
            t_local = np.array([d.dx, d.dy, d.dz], dtype=float)
            t_world = R_world_from_local @ t_local
            pos_next = pos_cur + t_world

            # rotate by local delta
            q_rel = _euler_to_quat(d.droll, d.dpitch, d.dyaw)
            quat_next = _quat_multiply(quat_cur, q_rel)
            R_world_from_local = _quat_to_rot(quat_next)

            # assemble PoseStamped in planning frame
            ps = PoseStamped()
            ps.header.frame_id = planning_frame
            ps.pose.position.x = float(pos_next[0])
            ps.pose.position.y = float(pos_next[1])
            ps.pose.position.z = float(pos_next[2])
            ps.pose.orientation = _array_to_quat_msg(quat_next)
            targets.append(ps)

            # update "current" for next step
            pos_cur = pos_next
            quat_cur = quat_next

        # Plan/execution parameters
        params = PlanRequestParameters(robot, "")
        params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

        # In-memory storage of achieved poses (Nx7: x,y,z,qx,qy,qz,qw)
        achieved_list: List[np.ndarray] = []

        # Execute sequence step-by-step, logging achieved state after each
        for i, goal in enumerate(targets):
            arm.set_start_state_to_current_state()
            arm.set_goal_state(pose_stamped_msg=goal, pose_link=tip_link)

            plan_result = arm.plan(single_plan_parameters=params)
            if not plan_result:
                logger.error(f"Planning failed at step {i}; aborting.")
                break

            if not _execute_with_fallback(robot, plan_result.trajectory, CONTROLLER_NAMES):
                logger.error(f"Execution failed at step {i}; aborting.")
                break

            # After execution, read back achieved EE pose and log it
            time.sleep(0.05)  # small settle delay
            with psm.read_only() as scene_after:
                scene_after.current_state.update()
                achieved = scene_after.current_state.get_pose(tip_link)

            pos = np.array([achieved.position.x, achieved.position.y, achieved.position.z], dtype=float)
            q = np.array([achieved.orientation.x, achieved.orientation.y, achieved.orientation.z, achieved.orientation.w], dtype=float)
            achieved_list.append(np.concatenate([pos, q]))

            logger.info(f"Step {i+1}/{len(targets)} complete: "
                        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

        logger.info("Pose sequence completed.")

        # Convert to a single numpy array for downstream use
        achieved_np = np.vstack(achieved_list) if achieved_list else np.empty((0, 7), dtype=float)
        logger.info(f"Collected {achieved_np.shape[0]} achieved poses in memory (shape: {achieved_np.shape}).")

    except Exception as e:
        logger.error(f"Pose sequence execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()