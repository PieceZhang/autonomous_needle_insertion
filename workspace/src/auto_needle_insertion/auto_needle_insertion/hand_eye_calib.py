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
import cv2
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ---------------------- Module constants ----------------------

NODE_NAME = "ee_pose_sequence_logger"

# Conservative planning scales
MAX_VELOCITY_SCALING = 0.60
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

# PoseStamped topic from NDI Polaris pose_broadcaster for the US tracker
US_TRACKER_TOPIC = "/ndi/us_tracker_pose"

# ---------------------- Logging ----------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- Tracker subscriber helper ----------------------
class _LatestPose:
    """Stores the most recent PoseStamped received on a topic."""
    def __init__(self):
        self.msg: PoseStamped | None = None
    def cb(self, msg: PoseStamped) -> None:
        self.msg = msg

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


# ---------------------- SE(3) helpers ----------------------
def _pose7_to_T(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qx,qy,qz,qw] -> 4x4 homogeneous transform (float64)."""
    if pose7.shape[-1] != 7:
        raise ValueError("pose7 must have 7 elements: x,y,z,qx,qy,qz,qw")
    x, y, z, qx, qy, qz, qw = [float(v) for v in pose7]
    R = _quat_to_rot((qx, qy, qz, qw))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    """SE(3) inverse using block structure."""
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=float)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def _relative_motion(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Compute relative motion from pose a to pose b: A = Ta^{-1} Tb."""
    return _invert_T(Ta) @ Tb

# ---------------------- Validation utilities ----------------------

def _pose7_is_finite(vec: np.ndarray) -> bool:
    """Return True if [x,y,z,qx,qy,qz,qw] has only finite values and a nonzero quaternion."""
    if vec.shape[-1] != 7:
        return False
    if not np.isfinite(vec).all():
        return False
    # Guard against zero-norm quaternions
    return float(np.linalg.norm(vec[3:7])) > 0.0

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
    """Return a well-spread sequence with rotational excitation.

    Design goals (per step):
      - small translations (~2 cm) and small rotations (~3°)
      - coverage within about ±45° cumulative about local axes
      - interleaved signs/translations to avoid large drift and planner issues

    Structure:
      1) Baseline translations & diagonals (18)
      2) Rotational sweeps to reach ~±45° cumulatively:
         - +45° about roll (15 × +3°)
         - −45° about pitch (15 × −3°)
         - +45° about yaw (15 × +3°)
      Each rotation step includes a small orthogonal translation to keep the scene changing.
    """
    deg = math.radians
    t = 0.02           # 2 cm per step (local frame)
    r = deg(3.0)       # ~3 degrees per step (local frame)

    deltas: List[LocalDelta] = []

    # ---- 1) Baseline: axis translations (±X, ±Y, ±Z) — 6 ----
    deltas += [
        LocalDelta(+t, 0.0, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, 0.0, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, -t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(0.0, 0.0, +t, 0.0, 0.0, 0.0),
        LocalDelta(0.0, 0.0, -t, 0.0, 0.0, 0.0),
    ]

    # ---- 2) Baseline: diagonals in XY plane — 4 ----
    deltas += [
        LocalDelta(+t, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, +t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(-t, -t, 0.0, 0.0, 0.0, 0.0),
        LocalDelta(+t, -t, 0.0, 0.0, 0.0, 0.0),
    ]

    # ---- 3) Baseline: diagonals in XZ and YZ planes — 8 ----
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

    # ---- 4) Rotational sweeps with small orthogonal translations ----
    def add_rot_sweep(axis: str, steps: int, sign: int, trans_axis: str) -> None:
        for k in range(steps):
            dx = dy = dz = 0.0
            droll = dpitch = dyaw = 0.0

            # small alternating translation along an orthogonal axis
            s = +1.0 if (k % 2 == 0) else -1.0
            if trans_axis == 'x':
                dx = s * (t * 0.5)
            elif trans_axis == 'y':
                dy = s * (t * 0.5)
            else:
                dz = s * (t * 0.5)

            if axis == 'roll':
                droll = sign * r
            elif axis == 'pitch':
                dpitch = sign * r
            else:  # 'yaw'
                dyaw = sign * r

            deltas.append(LocalDelta(dx, dy, dz, droll, dpitch, dyaw))

    steps = 15
    # +45° about roll (15 × +3°)
    deltas.append(LocalDelta(0.0, 0.0, 0.0, r * steps, 0.0, 0.0))
    add_rot_sweep(axis='roll', steps=steps*2, sign=-1, trans_axis='y')
    deltas.append(LocalDelta(0.0, 0.0, 0.0, r * steps, 0.0, 0.0))
    # −45° about pitch (15 × −3°)
    deltas.append(LocalDelta(0.0, 0.0, 0.0, 0.0, r * steps, 0.0))
    add_rot_sweep(axis='pitch', steps=steps*2, sign=-1, trans_axis='x')
    deltas.append(LocalDelta(0.0, 0.0, 0.0, 0.0, r * steps, 0.0))
    # +45° about yaw (15 × +3°)
    deltas.append(LocalDelta(0.0, 0.0, 0.0, 0.0, 0.0, r * steps))
    add_rot_sweep(axis='yaw', steps=steps*2, sign=-1, trans_axis='z')
    deltas.append(LocalDelta(0.0, 0.0, 0.0, 0.0, 0.0, r * steps))

    return deltas[0:6]

# ---------------------- Core routine ----------------------

def main() -> None:
    rclpy.init()

    # Subscribe to the US tracker PoseStamped (Polaris)
    sub_node = rclpy.create_node("us_tracker_listener")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    latest_tracker = _LatestPose()
    sub_node.create_subscription(PoseStamped, US_TRACKER_TOPIC, latest_tracker.cb, qos)

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

        # In-memory storage of US tracker poses (Nx7: x,y,z,qx,qy,qz,qw)
        us_tracker_list: List[np.ndarray] = []

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
            ee_pose7 = np.concatenate([pos, q])

            # Sample the latest tracker pose right after motion execution
            rclpy.spin_once(sub_node, timeout_sec=0.05)
            if latest_tracker.msg is not None:
                tp = latest_tracker.msg.pose
                tpos = np.array([tp.position.x, tp.position.y, tp.position.z], dtype=float)
                tquat = np.array([tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w], dtype=float)
                tracker_pose7 = np.concatenate([tpos, tquat])

                if _pose7_is_finite(tracker_pose7) and _pose7_is_finite(ee_pose7):
                    achieved_list.append(ee_pose7)
                    us_tracker_list.append(tracker_pose7)
                else:
                    logger.warning("Detected NaN/invalid in tracker or EE pose; dropping this pair.")
            else:
                logger.warning("No us_tracker PoseStamped received; dropping this pair.")

            logger.info(f"Step {i+1}/{len(targets)} complete: "
                        f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
            logger.info(f"Kept pairs so far: {len(achieved_list)}")

        logger.info("Pose sequence completed.")

        # Convert to a single numpy array for downstream use
        achieved_np = np.vstack(achieved_list) if achieved_list else np.empty((0, 7), dtype=float)
        us_tracker_np = np.vstack(us_tracker_list) if us_tracker_list else np.empty((0, 7), dtype=float)
        logger.info(f"Collected {achieved_np.shape[0]} achieved poses in memory (shape: {achieved_np.shape}).")
        logger.info(f"Collected {us_tracker_np.shape[0]} us_tracker poses in memory (shape: {us_tracker_np.shape}).")
        # logger.info(f"Collected EE pose: {achieved_np}")
        # logger.info(f"Collected tracker pose: {us_tracker_np}")

        # ---------------------- Relative motions for hand-eye ----------------------
        # Build A_i (robot/EE motion) and B_i (tracker/marker motion) as consecutive relative transforms
        # A_i = T_ee(i)^{-1} * T_ee(i+1),  B_i = T_tr(i)^{-1} * T_tr(i+1)
        if achieved_np.shape[0] >= 2 and us_tracker_np.shape[0] >= 2:
            T_ee_seq = [_pose7_to_T(p) for p in achieved_np]
            T_tr_seq = [_pose7_to_T(p) for p in us_tracker_np]

            # Ensure equal length pairing (they should be, since we append pairs together)
            n = min(len(T_ee_seq), len(T_tr_seq))
            T_ee_seq = T_ee_seq[:n]
            T_tr_seq = T_tr_seq[:n]

            A_list = [_relative_motion(T_ee_seq[i], T_ee_seq[i+1]) for i in range(n - 1)]
            B_list = [_relative_motion(T_tr_seq[i], T_tr_seq[i+1]) for i in range(n - 1)]

            # Stack to contiguous arrays for downstream solvers (AX = XB, AX = YB, etc.)
            A_array = np.stack(A_list, axis=0) if A_list else np.empty((0, 4, 4), dtype=float)
            B_array = np.stack(B_list, axis=0) if B_list else np.empty((0, 4, 4), dtype=float)

            # Optional: also expose R/t blocks for convenience
            A_R = A_array[:, :3, :3]
            A_t = A_array[:, :3, 3]
            B_R = B_array[:, :3, :3]
            B_t = B_array[:, :3, 3]

            logger.info(f"Built {A_array.shape[0]} paired relative motions (A_i, B_i) for AX=XB-style solvers.")
            logger.info(f"Example norms: |A_t0|={np.linalg.norm(A_t[0]):.4f} m, |B_t0|={np.linalg.norm(B_t[0]):.4f} m" if A_array.shape[0] > 0 else "")
        else:
            A_array = np.empty((0, 4, 4), dtype=float)
            B_array = np.empty((0, 4, 4), dtype=float)
            A_R = np.empty((0, 3, 3), dtype=float)
            A_t = np.empty((0, 3), dtype=float)
            B_R = np.empty((0, 3, 3), dtype=float)
            B_t = np.empty((0, 3), dtype=float)
            logger.warning("Not enough pose pairs to build relative motions (need ≥2).")

        logger.info(f"Collected relative EE pose: {A_array}")
        logger.info(f"Collected relative tracker pose: {B_array}")

    except Exception as e:
        logger.error(f"Pose sequence execution failed: {e}")
        raise
    finally:
        try:
            sub_node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()