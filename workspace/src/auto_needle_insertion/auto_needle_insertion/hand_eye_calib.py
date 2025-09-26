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
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from control_msgs.msg import JointTrajectoryControllerState
from action_msgs.msg import GoalStatusArray, GoalStatus
from control_msgs.action import FollowJointTrajectory  # for reference
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener

from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
import threading

# ---------------------- Module constants ----------------------

NODE_NAME = "ee_pose_sequence_logger"

# Robot base frame used for TF lookup of the EE pose
BASE_FRAME = "base"  # controller base frame (matches pendant)

# Conservative planning scales
MAX_VELOCITY_SCALING = 0.10
MAX_ACCELERATION_SCALING = 0.10

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

# Candidate TF frame names (try in order)
BASE_CANDIDATES = ["base", "base_link"]
TIP_TF_CANDIDATES = ["tool0", "tool0_controller", "flange"]

def resolve_tf_pair(tf_buffer: Buffer, node, timeout_sec: float = 2.0) -> Tuple[str, str]:
    """Probe TF and return a (base, tip) pair that exists in the buffer.
    Spins the node while probing so the buffer can fill. Raises if none found."""
    import time as _time
    deadline = _time.time() + timeout_sec
    while _time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        for base in BASE_CANDIDATES:
            for tip in TIP_TF_CANDIDATES:
                if tf_buffer.can_transform(base, tip, rclpy.time.Time()):
                    return base, tip
    # One last spin + check
    rclpy.spin_once(node, timeout_sec=0.05)
    for base in BASE_CANDIDATES:
        for tip in TIP_TF_CANDIDATES:
            if tf_buffer.can_transform(base, tip, rclpy.time.Time()):
                return base, tip
    raise RuntimeError("Could not resolve any (base, tip) TF pair. Inspect the TF tree with 'ros2 run tf2_tools view_frames'.")

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

# ---------------------- JTC state subscriber helper ----------------------
class _LatestJtcState:
    def __init__(self):
        self.msg: JointTrajectoryControllerState | None = None
        self.src_topic: str | None = None
    def cb_scaled(self, msg: JointTrajectoryControllerState) -> None:
        self.msg = msg
        self.src_topic = "/scaled_joint_trajectory_controller/state"
    def cb_plain(self, msg: JointTrajectoryControllerState) -> None:
        self.msg = msg
        self.src_topic = "/joint_trajectory_controller/state"

# ---------------------- ExecuteTrajectory status helper ----------------------
class _LatestExecStatus:
    def __init__(self):
        self.msg: GoalStatusArray | None = None
        self.src_topic: str | None = None
    def cb(self, msg: GoalStatusArray) -> None:
        self.msg = msg
        # 'goal_info' carries header.stamp but not the topic; store best-effort source via rclpy API if available
        try:
            self.src_topic = msg._topic_name  # not public API; may be absent
        except Exception:
            pass

# ---------------------- FollowJointTrajectory status helper ------------------
class _LatestFjtStatus:
    def __init__(self):
        self.msg: GoalStatusArray | None = None
        self.src_topic: str | None = None
    def cb(self, msg: GoalStatusArray) -> None:
        self.msg = msg
        try:
            self.src_topic = msg._topic_name
        except Exception:
            pass

def _wait_for_succeeded_after(node, latest: "_LatestExecStatus | _LatestFjtStatus", start_time: rclpy.time.Time, timeout: float = 5.0) -> tuple[bool, str | None]:
    """Wait until we see STATUS_SUCCEEDED for a goal newer than start_time.
    Returns (ok, src_topic).
    """
    import time as _time
    deadline = _time.time() + float(timeout)
    st_sec = int(start_time.nanoseconds // 1_000_000_000)
    st_nsec = int(start_time.nanoseconds % 1_000_000_000)

    def _is_newer(stamp) -> bool:
        return (stamp.sec > st_sec) or (stamp.sec == st_sec and stamp.nanosec > st_nsec)

    last_diag = None
    while _time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        msg = latest.msg
        if msg is None:
            continue
        for gs in msg.status_list:
            last_diag = (gs.status, gs.goal_info.stamp.sec, gs.goal_info.stamp.nanosec)
            if gs.status == GoalStatus.STATUS_SUCCEEDED and _is_newer(gs.goal_info.stamp):
                return True, getattr(latest, 'src_topic', None)
    if last_diag is not None:
        st, s, ns = last_diag
        logger.debug(f"Status wait timed out; last status={st} at {s}.{ns}s from {getattr(latest, 'src_topic', None)}")
    return False, getattr(latest, 'src_topic', None)


def _wait_for_execution_via_actions(node, exec_status: "_LatestExecStatus", fjt_status: "_LatestFjtStatus", start_time: rclpy.time.Time, timeout_total: float = 6.0) -> bool:
    """Try MoveIt ExecuteTrajectory status first, then fall back to controller FollowJointTrajectory status."""
    # Split budget: 2/3 for MoveIt, 1/3 for controller-level
    ok, src = _wait_for_succeeded_after(node, exec_status, start_time, timeout=timeout_total * (2.0/3.0))
    if ok:
        logger.info(f"Observed SUCCEEDED on MoveIt action status ({src or '/execute_trajectory/_action/status'}).")
        return True
    ok, src = _wait_for_succeeded_after(node, fjt_status, start_time, timeout=timeout_total * (1.0/3.0))
    if ok:
        logger.info(f"Observed SUCCEEDED on controller FollowJointTrajectory action status ({src}).")
        return True
    return False

# ---------------------- Execution settle helper ----------------------
from typing import Sequence

def _wait_until_controller_settled(
    node,
    latest_state: "_LatestJtcState",
    target_joint_names: Sequence[str],
    pos_tol_rad: float = 2e-3,
    consecutive: int = 5,
    timeout: float = 5.0,
) -> bool:
    """Wait until |actual.position - desired.position| < pos_tol_rad for
    all target joints for `consecutive` samples of the JTC state.
    Returns True if condition satisfied within timeout, else False.
    """
    import time as _time
    deadline = _time.time() + float(timeout)
    ok_count = 0
    idx_map: dict[str, int] | None = None

    while _time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        st = latest_state.msg
        if st is None:
            continue
        # Build index map once from controller's joint order
        if idx_map is None:
            idx_map = {name: i for i, name in enumerate(st.joint_names)}
        # Check only joints present in both sets
        errors = []
        for jn in target_joint_names:
            i = idx_map.get(jn) if idx_map else None
            if i is None:
                continue
            if i >= len(st.actual.positions) or i >= len(st.desired.positions):
                continue
            err = abs(float(st.actual.positions[i]) - float(st.desired.positions[i]))
            errors.append(err)
        if errors and max(errors) < pos_tol_rad:
            ok_count += 1
            if ok_count >= consecutive:
                return True
        else:
            ok_count = 0
    return False

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


# ---------------------- Rotation angle helper ----------------------
def _average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Average an array of quaternions (N,4)."""
    # [Function body omitted for brevity]
    pass  # Placeholder for context

def _rot_angle_deg(R: np.ndarray) -> float:
    """Return rotation angle (degrees) from 3x3 rotation matrix."""
    tr = (np.trace(R) - 1.0) * 0.5
    tr = max(-1.0, min(1.0, float(tr)))
    return math.degrees(math.acos(tr))


# ---------------------- Quaternion and averaging helpers ----------------------
def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    m = R
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
    # normalize
    n = math.sqrt(x*x + y*y + z*z + w*w)
    return (x/n, y/n, z/n, w/n)


def _average_quaternions(q_list: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Average unit quaternions via eigenvector of scatter matrix (Markley method)."""
    if not q_list:
        return (0.0, 0.0, 0.0, 1.0)
    A = np.zeros((4, 4), dtype=float)
    for q in q_list:
        qv = np.array(q, dtype=float).reshape(4, 1)
        A += qv @ qv.T
    # Principal eigenvector
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    # Ensure w >= 0 for consistency
    if q_avg[3] < 0:
        q_avg = -q_avg
    # normalize
    q_avg = q_avg / np.linalg.norm(q_avg)
    return (float(q_avg[0]), float(q_avg[1]), float(q_avg[2]), float(q_avg[3]))

# ---------------------- Validation utilities ----------------------

def _pose7_is_finite(vec: np.ndarray) -> bool:
    """Return True if [x,y,z,qx,qy,qz,qw] has only finite values and a nonzero quaternion."""
    if vec.shape[-1] != 7:
        return False
    if not np.isfinite(vec).all():
        return False
    # Guard against zero-norm quaternions
    return float(np.linalg.norm(vec[3:7])) > 0.0

# ---------------------- CSV helpers ----------------------

def _save_pose_array_csv(path: Path, poses: np.ndarray, header: str = "idx,x,y,z,qx,qy,qz,qw") -> None:
    """Save an Nx7 pose array to CSV with a header and index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header.split(","))
        for i, row in enumerate(poses):
            w.writerow([i] + [float(v) for v in row.tolist()])


def _save_Ts_csv(path: Path, Ts: np.ndarray, header: str = "idx,T00,T01,T02,T03,T10,T11,T12,T13,T20,T21,T22,T23,T30,T31,T32,T33") -> None:
    """Save a stack of 4x4 matrices (M,4,4) row-major flattened per row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header.split(","))
        for i, T in enumerate(Ts):
            w.writerow([i] + [float(x) for x in T.reshape(-1).tolist()])

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

    return deltas


# ---------------------- TF helpers ----------------------
from typing import Optional

def _lookup_latest_tf(
    tf_buffer: Buffer,
    node,
    base: str,
    tip: str,
    last_stamp: Optional[tuple] = None,
    timeout_total: float = 1.0,
):
    """Return (pos[3], quat[4], stamp_pair) for the *latest* TF, waiting until
    a **new** sample arrives (i.e., header.stamp != last_stamp) or until timeout.
    This prevents re-logging identical transforms when the buffer hasn't updated yet.
    """
    import time as _time
    deadline = _time.time() + float(timeout_total)
    pos = np.zeros(3, dtype=float)
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    stamp_pair = last_stamp
    while _time.time() < deadline:
        # Service callbacks so /tf is ingested
        rclpy.spin_once(node, timeout_sec=0.02)
        try:
            tf_trans = tf_buffer.lookup_transform(
                base, tip, rclpy.time.Time(), timeout=Duration(seconds=0.2)
            )
        except Exception:
            continue
        t = (tf_trans.header.stamp.sec, tf_trans.header.stamp.nanosec)
        pos = np.array([
            tf_trans.transform.translation.x,
            tf_trans.transform.translation.y,
            tf_trans.transform.translation.z,
        ], dtype=float)
        q = np.array([
            tf_trans.transform.rotation.x,
            tf_trans.transform.rotation.y,
            tf_trans.transform.rotation.z,
            tf_trans.transform.rotation.w,
        ], dtype=float)
        # Accept if it's a *new* TF sample
        if (last_stamp is None) or (t != last_stamp):
            return pos, q, t
        # else: wait a bit and try again
    # Timed out: return the latest we have (may be unchanged)
    return pos, q, stamp_pair

# ---------------------- Core routine ----------------------

def main() -> None:
    rclpy.init()

    # Subscribe to the US tracker PoseStamped (Polaris)
    sub_node = rclpy.create_node("us_tracker_listener")
    # Spin the TF/subscription node in the background so /tf and /ndi/us_tracker_pose
    # callbacks are processed continuously while we plan/execute motions.
    executor = SingleThreadedExecutor()
    executor.add_node(sub_node)
    _spin_thread = threading.Thread(target=executor.spin, daemon=True)
    _spin_thread.start()
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, sub_node)
    # Warm up TF buffer and resolve (base, tip) TF frames to use for logging
    try:
        BASE_FRAME_RESOLVED, TIP_LINK_TF = resolve_tf_pair(tf_buffer, sub_node, timeout_sec=2.0)
        logger.info(f"Resolved TF pair for EE logging: base='{BASE_FRAME_RESOLVED}', tip='{TIP_LINK_TF}'")
    except Exception as e:
        # Fall back to defaults if resolution fails; will still try lookup later
        BASE_FRAME_RESOLVED, TIP_LINK_TF = BASE_FRAME, "tool0_controller"
        logger.warning(f"TF auto-resolution failed ({e}); falling back to '{BASE_FRAME_RESOLVED}'<-'{TIP_LINK_TF}'.")
    # Prefer 'tool0' (pendant-consistent) if available
    try:
        if tf_buffer.can_transform(BASE_FRAME_RESOLVED, 'tool0', rclpy.time.Time()):
            TIP_LINK_TF = 'tool0'
            logger.info("Overriding TIP_LINK_TF to 'tool0' for pendant parity.")
    except Exception:
        pass
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    latest_tracker = _LatestPose()
    sub_node.create_subscription(PoseStamped, US_TRACKER_TOPIC, latest_tracker.cb, qos)

    # Subscribe to controller state to know when execution has *actually* finished
    latest_jtc = _LatestJtcState()
    sub_node.create_subscription(JointTrajectoryControllerState,
                                 "/scaled_joint_trajectory_controller/state",
                                 latest_jtc.cb_scaled, qos)
    sub_node.create_subscription(JointTrajectoryControllerState,
                                 "/joint_trajectory_controller/state",
                                 latest_jtc.cb_plain, qos)

    # Subscribe to MoveIt ExecuteTrajectory action status (root and namespaced) to detect completion
    latest_exec = _LatestExecStatus()
    sub_node.create_subscription(GoalStatusArray, "/execute_trajectory/_action/status", latest_exec.cb, qos)
    sub_node.create_subscription(GoalStatusArray, "/move_group/execute_trajectory/_action/status", latest_exec.cb, qos)

    # Subscribe to controller-level FollowJointTrajectory action status as a fallback
    latest_fjt = _LatestFjtStatus()
    sub_node.create_subscription(GoalStatusArray, "/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status", latest_fjt.cb, qos)
    sub_node.create_subscription(GoalStatusArray, "/joint_trajectory_controller/follow_joint_trajectory/_action/status", latest_fjt.cb, qos)

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
        # Joint order from MoveIt group (used to select a subset in controller state)
        group_joint_names = list(robot.get_robot_model().get_joint_model_group(group_name).active_joint_model_names)
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
            # Record time to correlate with the upcoming execute_trajectory goal
            exec_start_t = sub_node.get_clock().now()

            if not _execute_with_fallback(robot, plan_result.trajectory, CONTROLLER_NAMES):
                logger.error(f"Execution failed at step {i}; aborting.")
                break
            logger.info(f"Execution succeed at step {i}.")

            # Observe MoveIt's execute_trajectory action status and controller-level fallback to confirm completion
            if not _wait_for_execution_via_actions(sub_node, latest_exec, latest_fjt, exec_start_t, timeout_total=6.0):
                logger.warning("Did not observe SUCCEEDED on MoveIt or controller action status within timeout; proceeding to log pose regardless.")

            # Wait for a *new* TF sample to avoid re-logging identical data
            if 'last_tf_stamp' not in locals():
                last_tf_stamp = None

            pos, q, new_stamp = _lookup_latest_tf(
                tf_buffer, sub_node, BASE_FRAME_RESOLVED, TIP_LINK_TF,
                last_stamp=last_tf_stamp, timeout_total=1.0
            )

            # If TF didn't advance, fall back to MoveIt's current_state
            used_fallback = False
            if (last_tf_stamp is not None) and (new_stamp == last_tf_stamp):
                try:
                    with psm.read_write() as scene_rw:
                        scene_rw.current_state.update()
                        T_now = scene_rw.current_state.get_global_link_transform(tip_link)
                    pos = T_now[:3, 3].astype(float)
                    q = np.array(_rot_to_quat(T_now[:3, :3]), dtype=float)
                    used_fallback = True
                    logger.warning("TF timestamp did not advance; used MoveIt current_state as fallback for this sample.")
                except Exception as _e:
                    logger.warning(f"Fallback to current_state failed: {_e}")

            ee_pose7 = np.concatenate([pos, q])
            last_tf_stamp = new_stamp if new_stamp is not None else last_tf_stamp

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
                        f"pos=({pos[0]*1000:.3f},{pos[1]*1000:.3f},{pos[2]*1000:.3f})")
            try:
                logger.info(f"TF stamp used: {last_tf_stamp}, fallback={'yes' if used_fallback else 'no'}")
            except Exception:
                pass
            logger.info(f"Kept pairs so far: {len(achieved_list)}")

        logger.info("Pose sequence completed.")

        # Convert to a single numpy array for downstream use
        achieved_np = np.vstack(achieved_list) if achieved_list else np.empty((0, 7), dtype=float)
        us_tracker_np = np.vstack(us_tracker_list) if us_tracker_list else np.empty((0, 7), dtype=float)
        logger.info(f"Collected {achieved_np.shape[0]} achieved poses in memory (shape: {achieved_np.shape}).")
        logger.info(f"Collected {us_tracker_np.shape[0]} us_tracker poses in memory (shape: {us_tracker_np.shape}).")
        # ---------------------- Quick sanity diagnostics ----------------------
        def _step_lengths_m(P: np.ndarray) -> np.ndarray:
            return np.linalg.norm(P[1:, 0:3] - P[:-1, 0:3], axis=1) if len(P) > 1 else np.array([])

        ee_steps = _step_lengths_m(achieved_np)
        tr_steps = _step_lengths_m(us_tracker_np)
        if ee_steps.size > 0 and tr_steps.size > 0:
            ee_med = float(np.median(ee_steps))
            tr_med = float(np.median(tr_steps))
            if ee_med > 0.0:
                ratio = tr_med / ee_med
                if ratio > 5.0 or ratio < 0.2:
                    logger.warning(
                        "Tracker step length is %.1fx of EE step length (median). Possible unit mismatch (mm vs m) or wrong frames.",
                        ratio,
                    )
                    logger.warning("NDI SDKs commonly report positions in millimeters; ensure your driver publishes meters (SI).")

        # ---------------------- Persist raw pose logs to CSV ----------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.cwd() / "handeye_logs"
        ee_csv = out_dir / f"ee_poses_{ts}.csv"
        tr_csv = out_dir / f"tracker_poses_{ts}.csv"
        if achieved_np.shape[0] > 0:
            _save_pose_array_csv(ee_csv, achieved_np)
            logger.info(f"Saved EE poses to: {ee_csv}")
        else:
            logger.warning("No EE poses to save.")
        if us_tracker_np.shape[0] > 0:
            _save_pose_array_csv(tr_csv, us_tracker_np)
            logger.info(f"Saved tracker poses to: {tr_csv}")
        else:
            logger.warning("No tracker poses to save.")

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
            # ---------------------- Persist relative motions to CSV ----------------------
            A_csv = out_dir / f"A_rel_motions_{ts}.csv"
            B_csv = out_dir / f"B_rel_motions_{ts}.csv"
            if A_array.shape[0] > 0:
                _save_Ts_csv(A_csv, A_array)
                logger.info(f"Saved relative EE motions (A_i) to: {A_csv}")
            else:
                logger.warning("No relative EE motions to save.")
            if B_array.shape[0] > 0:
                _save_Ts_csv(B_csv, B_array)
                logger.info(f"Saved relative tracker motions (B_i) to: {B_csv}")
            else:
                logger.warning("No relative tracker motions to save.")
        else:
            A_array = np.empty((0, 4, 4), dtype=float)
            B_array = np.empty((0, 4, 4), dtype=float)
            A_R = np.empty((0, 3, 3), dtype=float)
            A_t = np.empty((0, 3), dtype=float)
            B_R = np.empty((0, 3, 3), dtype=float)
            B_t = np.empty((0, 3), dtype=float)
            logger.warning("Not enough pose pairs to build relative motions (need ≥2).")
            # ---------------------- Persist relative motions to CSV ----------------------
            A_csv = out_dir / f"A_rel_motions_{ts}.csv"
            B_csv = out_dir / f"B_rel_motions_{ts}.csv"
            if A_array.shape[0] > 0:
                _save_Ts_csv(A_csv, A_array)
                logger.info(f"Saved relative EE motions (A_i) to: {A_csv}")
            else:
                logger.warning("No relative EE motions to save.")
            if B_array.shape[0] > 0:
                _save_Ts_csv(B_csv, B_array)
                logger.info(f"Saved relative tracker motions (B_i) to: {B_csv}")
            else:
                logger.warning("No relative tracker motions to save.")

        # logger.info(f"Collected relative EE pose: {A_array}")
        # logger.info(f"Collected relative tracker pose: {B_array}")

        # ---------------------- OpenCV hand–eye (eye-to-hand) ----------------------
        # We use OpenCV's Robot-World/Hand-Eye solver to estimate:
        #   ^wT_b  (base wrt world) and ^cT_g  (camera wrt gripper)
        # Here: world == tracker target (rigid body on the EE), camera == Polaris (fixed),
        #       base == robot base, gripper == UR5e EE.
        # API expects per-sample: R_world2cam, t_world2cam, R_base2gripper, t_base2gripper.
        #   R_world2cam,t_world2cam  correspond to  ^cT_t   from Polaris (tool/marker in camera).
        #   R_base2gripper,t_base2gripper correspond to  ^gT_b, the inverse of ^bT_g from MoveIt.
        try:
            import cv2
            nA = achieved_np.shape[0]
            nB = us_tracker_np.shape[0]
            n = min(nA, nB)
            if n < 3:
                logger.warning("Need at least 3 paired poses for hand–eye; skipping calibration.")
            else:
                R_world2cam_list, t_world2cam_list = [], []
                R_base2gripper_list, t_base2gripper_list = [], []

                for i in range(n):
                    # tracker/tool pose in camera frame: ^cT_t
                    x, y, z, qx, qy, qz, qw = us_tracker_np[i]
                    R_ct = _quat_to_rot((qx, qy, qz, qw))
                    t_ct = np.array([[x], [y], [z]], dtype=float)
                    R_world2cam_list.append(R_ct.astype(np.float64))
                    t_world2cam_list.append(t_ct.astype(np.float64))

                    # robot EE pose in base: ^bT_g  -> invert to ^gT_b
                    xb, yb, zb, qbx, qby, qbz, qbw = achieved_np[i]
                    R_bg = _quat_to_rot((qbx, qby, qbz, qbw))
                    t_bg = np.array([xb, yb, zb], dtype=float)
                    T_bg = np.eye(4, dtype=float)
                    T_bg[:3, :3] = R_bg
                    T_bg[:3, 3] = t_bg
                    T_gb = _invert_T(T_bg)
                    R_gb = T_gb[:3, :3].astype(np.float64)
                    t_gb = T_gb[:3, 3].reshape(3, 1).astype(np.float64)
                    R_base2gripper_list.append(R_gb)
                    t_base2gripper_list.append(t_gb)

                # Solve AX = ZB (Robot-World/Hand-Eye) -> outputs ^wT_b and ^cT_g
                R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
                    R_world2cam_list, t_world2cam_list,
                    R_base2gripper_list, t_base2gripper_list,
                    method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
                )

                # Build transforms
                T_b2w = np.eye(4); T_b2w[:3, :3] = R_b2w; T_b2w[:3, 3] = t_b2w.flatten()
                T_g2c = np.eye(4); T_g2c[:3, :3] = R_g2c; T_g2c[:3, 3] = t_g2c.flatten()
                T_c2g = _invert_T(T_g2c)

                # Compute constant ^gT_t from each sample:  ^gT_t(i) =  (^cT_g)^{-1}  ^cT_t(i)
                Rq_list, t_list = [], []
                for i in range(n):
                    T_ct = _pose7_to_T(us_tracker_np[i])
                    T_gt = T_c2g @ T_ct
                    Rq_list.append(_rot_to_quat(T_gt[:3, :3]))
                    t_list.append(T_gt[:3, 3])

                q_avg = _average_quaternions(Rq_list)
                R_gt_mean = _quat_to_rot(q_avg)
                t_gt_mean = np.mean(np.stack(t_list, axis=0), axis=0)

                T_gt_mean = np.eye(4)
                T_gt_mean[:3, :3] = R_gt_mean
                T_gt_mean[:3, 3] = t_gt_mean

                # Report results
                def fmt_T(T: np.ndarray) -> str:
                    return "\n" + "\n".join(["  [" + ", ".join(f"{v: .6f}" for v in row) + "]" for row in T])

                logger.info("OpenCV Robot-World/Hand–Eye results:")
                logger.info(f"  ^wT_b (base wrt world/target) = {fmt_T(T_b2w)}")
                logger.info(f"  ^cT_g (camera wrt gripper)     = {fmt_T(T_g2c)}")
                logger.info(f"Estimated constant transform from TRACKER/TOOL -> EE ( ^gT_t ): {fmt_T(T_gt_mean)}")

                # ---------------------- Residual metrics / measurement indices ----------------------
                # 1) Kinematic identity residual:
                #    Check  ^cT_g * ^gT_b(i)  ≈  ^cT_t(i) * ^wT_b
                rot_err_deg_list = []
                trans_err_mm_list = []

                for i in range(n):
                    # ^gT_b(i)
                    xb, yb, zb, qbx, qby, qbz, qbw = achieved_np[i]
                    R_bg = _quat_to_rot((qbx, qby, qbz, qbw))
                    T_bg = np.eye(4); T_bg[:3, :3] = R_bg; T_bg[:3, 3] = [xb, yb, zb]
                    T_gb = _invert_T(T_bg)

                    # ^cT_t(i)
                    T_ct = _pose7_to_T(us_tracker_np[i])

                    # Left and right sides
                    T_left  = T_g2c @ T_gb
                    T_right = T_ct @ T_b2w
                    T_delta = T_left @ _invert_T(T_right)

                    rot_err_deg_list.append(_rot_angle_deg(T_delta[:3, :3]))
                    trans_err_mm_list.append(float(np.linalg.norm(T_delta[:3, 3]) * 1e3))

                def _stats(arr: List[float]) -> Tuple[float, float, float, float]:
                    if len(arr) == 0:
                        return (float('nan'), float('nan'), float('nan'), float('nan'))
                    a = np.asarray(arr, dtype=float)
                    rms = float(np.sqrt(np.mean(np.square(a))))
                    med = float(np.median(a))
                    p95 = float(np.percentile(a, 95.0))
                    mx  = float(np.max(a))
                    return rms, med, p95, mx

                rms_rot, med_rot, p95_rot, max_rot = _stats(rot_err_deg_list)
                rms_trn, med_trn, p95_trn, max_trn = _stats(trans_err_mm_list)

                logger.info(
                    "Kinematic identity residual  (^cT_g * ^gT_b  vs  ^cT_t * ^wT_b):\n"
                    f"  rotation:  RMS={rms_rot:.3f} deg, median={med_rot:.3f}, p95={p95_rot:.3f}, max={max_rot:.3f}\n"
                    f"  translation: RMS={rms_trn:.3f} mm, median={med_trn:.3f}, p95={p95_trn:.3f}, max={max_trn:.3f}"
                )

                # 2) Constancy of ^gT_t across samples:  ^gT_t(i) ≈ ^gT_t(mean)
                const_rot_deg, const_trn_mm = [], []
                for i in range(n):
                    T_ct = _pose7_to_T(us_tracker_np[i])
                    T_gt_i = T_c2g @ T_ct   # ^gT_t(i)
                    T_dev = _invert_T(T_gt_mean) @ T_gt_i
                    const_rot_deg.append(_rot_angle_deg(T_dev[:3, :3]))
                    const_trn_mm.append(float(np.linalg.norm(T_dev[:3, 3]) * 1e3))

                rms_crot, med_crot, p95_crot, max_crot = _stats(const_rot_deg)
                rms_ctrn, med_ctrn, p95_ctrn, max_ctrn = _stats(const_trn_mm)

                logger.info(
                    "Constancy residual of ^gT_t across samples  (^gT_t(i) vs mean):\n"
                    f"  rotation:  RMS={rms_crot:.3f} deg, median={med_crot:.3f}, p95={p95_crot:.3f}, max={max_crot:.3f}\n"
                    f"  translation: RMS={rms_ctrn:.3f} mm, median={med_ctrn:.3f}, p95={p95_ctrn:.3f}, max={max_ctrn:.3f}"
                )

        except Exception as e:
            logger.error(f"OpenCV hand–eye calibration failed: {e}")

    except Exception as e:
        logger.error(f"Pose sequence execution failed: {e}")
        raise
    finally:
        # Stop background executor and join the spin thread, then clean up the node/ROS.
        try:
            if 'executor' in locals():
                executor.shutdown()
        except Exception:
            pass
        try:
            if '_spin_thread' in locals():
                _spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            sub_node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()