#!/usr/bin/env python3
"""
Hand–eye calibration for UR/MoveIt + NDI Polaris (eye-in-hand via OpenCV’s
calibrateHandEye), with motion excitation, AX=XB solving, residual checks, and result logging.

Summary
-------
This module drives a small, well-conditioned excitation of the robot end effector using
MoveItPy, and for each step pairs the achieved gripper pose with the
simultaneously received NDI PoseStamped of the tracked tool in the Polaris frame.
From these sequences it builds relative motions A_i and B_i, solves for the hand–eye
transform using OpenCV, and reports algebraic AX=XB residuals on SE(3) as a basic
consistency check.

Pipeline
--------
1) Warm up TF and resolve frames (base, tip); subscribe to /ndi/us_tracker_pose and to
   controller/MoveIt status topics for execution confirmation.
2) Generate a well-spread local-frame excitation sequence via _default_local_deltas()
   (small translations + small rotations).
3) For each target: plan & execute; confirm completion by observing the MoveIt
   /execute_trajectory/_action/status and (fallback) */follow_joint_trajectory/_action/status
   streams; then log the achieved gripper pose and the latest tracker pose.
4) Build relative motions A_i (gripper) and B_i (tracker) and call
   cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
   to estimate ^gT_c ("camera"/tracker with respect to the gripper).
5) Compute SE(3)-log residuals E_i and print median/95th-percentile rotation (deg)
   and translation (mm) as a quick diagnostic.

Frames & units
--------------
- Robot base:  'base' (or 'base_link' if available).
- Gripper/tip: MoveIt tip link (prefers 'tool0' when present).
- Polaris camera frame: fixed sensor frame.
- All translations are **meters** and rotations are **radians**. ndi_ros2_driver publishes in
  **metres**; the script emits a step-length ratio warning if a unit mismatch is suspected.

Inputs & topics
---------------
- Subscribes: /ndi/us_tracker_pose (geometry_msgs/PoseStamped).
- Observes status: /execute_trajectory/_action/status and
  */follow_joint_trajectory/_action/status.
- Requires a running UR ROS 2 driver/controller and MoveIt 2 planning scene.

Configuration highlights
------------------------
- Motion scaling: MAX_VELOCITY_SCALING, MAX_ACCELERATION_SCALING.
- Waits: PLANNING_SCENE_SYNC_DELAY, POST_EXECUTION_SETTLE_SEC.
- Excitation: _default_local_deltas() (both translational and rotational excitation)

Outputs
-------
- Logs the estimated ^gT_c and an AX=XB residual summary.
- Keeps in-memory arrays 'achieved_np' (N×7 for ^bT_g) and 'tracker_np' (N×7 for ^cT_t).

Usage
-----
Run with the auto_needle_insertion package with the UR driver and MoveIt running, i.eg.:

    ros2 launch auto_needle_insertion move_robot.launch.py mode:=hand_eye_calib

Notes
-----
- The OpenCV API expects (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam) and
  returns (R_cam2gripper, t_cam2gripper).
- In our settings the camera role and target role are inverted, thus the tracker pose needs
  to be inverted before passing to OpenCV.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Tuple
import os

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

NODE_NAME = "hand_eye_calib"

# Robot base frame used for TF lookup of the EE pose
BASE_FRAME = "base"

# Conservative planning scales
MAX_VELOCITY_SCALING = 1.00
MAX_ACCELERATION_SCALING = 0.20

# Allow time for the planning scene to sync joint states
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds
POST_EXECUTION_SETTLE_SEC = 0.75  # small wait to ensure state update after motion

# Controller fallback order (hardware -> default -> sim/common)
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",
    "",
    "joint_trajectory_controller",
]

# Preferred tip link names in order of preference
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

# PoseStamped topic from ndi_ros2_driver pose_broadcaster for the tracker
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

# --- Helper to match ee_pose_logger semantics ---
def relative_transform(parent_T: np.ndarray, child_T: np.ndarray) -> np.ndarray:
    """Return the homogeneous transform of the child expressed in the parent frame.
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
    T_rel = np.eye(4, dtype=float)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = t_rel
    return T_rel

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

# ---------------------- Validation utilities ----------------------

def _pose7_is_finite(vec: np.ndarray) -> bool:
    """Return True if [x,y,z,qx,qy,qz,qw] has only finite values and a nonzero quaternion."""
    if vec.shape[-1] != 7:
        return False
    if not np.isfinite(vec).all():
        return False
    # Guard against zero-norm quaternions
    return float(np.linalg.norm(vec[3:7])) > 0.0

# ---------------------- AX=XB algebraic consistency (SE(3) log) ----------------------
def _skew(w: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]], dtype=float)

def _so3_log(R: np.ndarray) -> np.ndarray:
    """Matrix logarithm for SO(3): returns the rotation vector ω (axis * angle, in radians).
    Robust near-zero rotation using first-order series.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    # Clamp trace for numerical stability
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        # Use first-order approximation: log(R) ≈ (R - R^T)/2 with vee operator
        W = 0.5 * (R - R.T)
        return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)
    else:
        W = (R - R.T) * (theta / (2.0 * math.sin(theta)))
        return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=float)

def _se3_log_xi(T: np.ndarray) -> np.ndarray:
    """Matrix logarithm for SE(3) returning the 6×1 twist coordinates ξ = [v; ω].
    Uses V^{-1} from Blanco-Claraco (2010) for translating t to v. Units: meters & radians.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    omega = _so3_log(R)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        # Series expansion for small angles: V^{-1} ≈ I - 1/2 Ω + 1/12 Ω^2
        Omega = _skew(omega)
        V_inv = np.eye(3) - 0.5 * Omega + (1.0 / 12.0) * (Omega @ Omega)
    else:
        Omega = _skew(omega)
        half_theta = 0.5 * theta
        # Avoid division by zero when sin(θ/2) is tiny
        s = math.sin(half_theta)
        if abs(s) < 1e-12:
            V_inv = np.eye(3) - 0.5 * Omega + (1.0 / 12.0) * (Omega @ Omega)
        else:
            A = (1.0 - (theta * math.cos(half_theta)) / (2.0 * s)) / (theta * theta)
            V_inv = np.eye(3) - 0.5 * Omega + A * (Omega @ Omega)
    v = V_inv @ t
    return np.hstack([v, omega])

def axxb_residuals(A_array: np.ndarray, B_array: np.ndarray, X: np.ndarray) -> dict:
    """Compute algebraic residuals for AX ≈ XB using the SE(3) log.

    Args:
        A_array: (M,4,4) relative motions of gripper, A_i = ^bT_g(i+1)^{-1} ^bT_g(i).
        B_array: (M,4,4) relative motions of target in camera, B_i = ^cT_t(i+1) ^cT_t(i)^{-1}.
        X:       (4,4) hand–eye transform, X = ^gT_c (camera2gripper).

    Returns:
        dict with arrays of per-pair rotation error (deg) and translation error (m),
        plus summary statistics.
    """
    A_array = np.asarray(A_array, dtype=float)
    B_array = np.asarray(B_array, dtype=float)
    X = np.asarray(X, dtype=float).reshape(4, 4)
    if A_array.ndim != 3 or B_array.ndim != 3 or A_array.shape != B_array.shape:
        raise ValueError("A_array and B_array must be (M,4,4) and same shape.")
    M = A_array.shape[0]
    if M == 0:
        return {
            "rot_deg": np.array([]),
            "trans_m": np.array([]),
            "median_deg": float("nan"),
            "p95_deg": float("nan"),
            "median_m": float("nan"),
            "p95_m": float("nan"),
        }

    X_inv = _invert_T(X)
    rot_deg = np.zeros(M, dtype=float)
    trans_m = np.zeros(M, dtype=float)

    for i in range(M):
        E = A_array[i] @ X @ _invert_T(B_array[i]) @ X_inv
        xi = _se3_log_xi(E)
        rot_deg[i] = np.degrees(np.linalg.norm(xi[3:]))
        trans_m[i] = float(np.linalg.norm(xi[:3]))

    def _pct(a: np.ndarray, p: float) -> float:
        return float(np.percentile(a, p)) if a.size else float("nan")

    summary = {
        "rot_deg": rot_deg,
        "trans_m": trans_m,
        "median_deg": float(np.median(rot_deg)) if rot_deg.size else float("nan"),
        "p95_deg": _pct(rot_deg, 95.0),
        "median_m": float(np.median(trans_m)) if trans_m.size else float("nan"),
        "p95_m": _pct(trans_m, 95.0),
    }
    return summary

def axxb_print_summary(res: dict, logger_obj=logger, label: str = "AX=XB residuals") -> None:
    """Pretty-print residual summary returned by axxb_residuals()."""
    if res["rot_deg"].size == 0:
        logger_obj.warning(f"{label}: no residuals to report (empty A/B arrays).")
        return
    logger_obj.info(
        f"{label}: rotation median={res['median_deg']:.3f}°, p95={res['p95_deg']:.3f}°; "
        f"translation median={res['median_m']*1000.0:.2f} mm, p95={res['p95_m']*1000.0:.2f} mm."
    )

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
      - small translations (~2 cm) and small rotations (~5°)
      - coverage within about ±45° cumulative about local axes
      - interleaved signs/translations to avoid large drift and planner issues
      - avoid large single-step rotations to remove angle inversion ambiguities in relative motions

    Structure:
      1) Baseline translations & diagonals (18)
      2) Rotational sweeps to reach ~±45° cumulatively:
         - +45° about roll
         - −45° about pitch
         - +45° about yaw
      Each rotation step includes a small orthogonal translation to keep the scene changing.
    """
    deg = math.radians
    t = 0.02           # 2 cm per step (local frame)
    r = deg(5.0)       # ~3 degrees per step (local frame)

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

    steps = 8
    # +/-40° about roll (8 × +5°)
    add_rot_sweep(axis='roll', steps=steps, sign=-1, trans_axis='y')
    add_rot_sweep(axis='roll', steps=steps, sign= 1, trans_axis='y')
    # +/−45° about pitch
    add_rot_sweep(axis='pitch', steps=steps, sign=-1, trans_axis='x')
    add_rot_sweep(axis='pitch', steps=steps, sign= 1, trans_axis='x')
    # +/-45° about yaw
    add_rot_sweep(axis='yaw', steps=steps, sign=-1, trans_axis='z')
    add_rot_sweep(axis='yaw', steps=steps, sign= 1, trans_axis='z')

    return deltas

def _square_local_deltas(
    side: float = 0.10,
    step: float = 0.02,
    axis_order: str = "xy",
    layers: int = 2,
    layer_gap: float = 0.05,
    vertical_step: float | None = None,
) -> List[LocalDelta]:
    """Return incremental *local-frame* deltas that trace one or more squares centered at start.

    The trajectory starts at the square center on layer 0 (z = 0), traverses the
    square perimeter, returns to the center, then (if layers > 1) moves along +Z
    by ``layer_gap`` to the next layer and repeats the same square. After the last
    layer, it returns to the original center at z = 0. Orientation is held
    constant (no rotation deltas).

    Args:
        side: Side length of each square in meters.
        step: Linear translation increment in meters along X/Y edges.
        axis_order: "xy" (go +X half, +Y, −X, −Y, +X half) or "yx" to swap axes.
        layers: Number of square layers to draw (>=1). Default 2 as requested.
        layer_gap: Separation (meters) between consecutive layers along +Z.
        vertical_step: Optional increment for Z moves (defaults to ``step`` if None).

    Returns:
        List[LocalDelta] defining the incremental motion sequence.
    """
    if side <= 0.0:
        raise ValueError("side must be > 0")
    if step <= 0.0:
        raise ValueError("step must be > 0")
    if axis_order not in ("xy", "yx"):
        raise ValueError("axis_order must be 'xy' or 'yx'")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if layer_gap < 0.0:
        raise ValueError("layer_gap must be >= 0")

    vz = step if vertical_step is None else float(vertical_step)
    if vz <= 0.0:
        raise ValueError("vertical_step must be > 0 (or None)")

    # Helper: produce increments along a single-axis segment so that the sum is exact.
    def _segment_xy(dx: float = 0.0, dy: float = 0.0) -> List[LocalDelta]:
        # Only one axis should be non-zero for a square edge.
        if dx != 0.0 and dy != 0.0:
            raise ValueError("Only one of dx or dy should be non-zero for a square edge.")
        dist = abs(dx if dx != 0.0 else dy)
        if dist == 0.0:
            return []
        signx = 0.0 if dx == 0.0 else (1.0 if dx > 0.0 else -1.0)
        signy = 0.0 if dy == 0.0 else (1.0 if dy > 0.0 else -1.0)
        n = int(dist // step)
        rem = dist - n * step
        incs = [step] * n
        if rem > 1e-9:
            incs.append(rem)
        return [LocalDelta(signx * inc, signy * inc, 0.0, 0.0, 0.0, 0.0) for inc in incs]

    def _segment_z(dz: float) -> List[LocalDelta]:
        dist = abs(dz)
        if dist == 0.0:
            return []
        signz = 0.0 if dz == 0.0 else (1.0 if dz > 0.0 else -1.0)
        n = int(dist // vz)
        rem = dist - n * vz
        incs = [vz] * n
        if rem > 1e-9:
            incs.append(rem)
        return [LocalDelta(0.0, 0.0, signz * inc, 0.0, 0.0, 0.0) for inc in incs]

    def _one_square_sequence() -> List[LocalDelta]:
        half = side * 0.5
        seq: List[LocalDelta] = []
        if axis_order == "xy":
            # Center -> +X half-side (to the middle of right edge)
            seq += _segment_xy(dx=+half)
            # +Y full side (right edge up)
            seq += _segment_xy(dy=+side)
            # -X full side (top edge left)
            seq += _segment_xy(dx=-side)
            # -Y full side (left edge down)
            seq += _segment_xy(dy=-side)
            # +X half-side back to center
            seq += _segment_xy(dx=+half)
        else:  # axis_order == "yx"
            # Center -> +Y half-side (to the middle of top edge)
            seq += _segment_xy(dy=+half)
            # +X full side (top edge right)
            seq += _segment_xy(dx=+side)
            # -Y full side (right edge down)
            seq += _segment_xy(dy=-side)
            # -X full side (bottom edge left)
            seq += _segment_xy(dx=-side)
            # +Y half-side back to center
            seq += _segment_xy(dy=+half)
        return seq

    full: List[LocalDelta] = []

    # First layer at z = 0
    full += _one_square_sequence()

    # Additional layers: move up along +Z by layer_gap, draw square, remain centered on that layer
    accumulated_z = 0.0
    for _ in range(1, layers):
        full += _segment_z(+layer_gap)
        accumulated_z += layer_gap
        full += _one_square_sequence()

    # Return to the original center height (z = 0)
    if accumulated_z > 0.0:
        full += _segment_z(-accumulated_z)

    return full

def main() -> None:
    rclpy.init()

    # Subscribe to the Polaris PoseStamped
    sub_node = rclpy.create_node("us_tracker_listener")
    # Spin the subscription node in the background so /ndi/us_tracker_pose
    # callbacks are processed continuously while we plan/execute motions.
    executor = SingleThreadedExecutor()
    executor.add_node(sub_node)
    _spin_thread = threading.Thread(target=executor.spin, daemon=True)
    _spin_thread.start()
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

        base_frame = robot.get_robot_model().model_frame
        logger.info(f"Using base frame: {base_frame}")

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

        # In-memory storage of achieved gripper poses (Nx7: x,y,z,qx,qy,qz,qw)
        achieved_list: List[np.ndarray] = []

        # In-memory storage of Polaris tracker poses (Nx7: x,y,z,qx,qy,qz,qw)
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
            if not _wait_for_execution_via_actions(sub_node, latest_exec, latest_fjt, exec_start_t, timeout_total=0.3):
                logger.warning("Did not observe SUCCEEDED on MoveIt or controller action status within timeout; proceeding to log pose regardless.")

            # --- Pose logging in base frame from MoveIt's planning scene ---
            time.sleep(POST_EXECUTION_SETTLE_SEC)
            with psm.read_write() as scene_rw:
                scene_rw.current_state.update()
                T_base = scene_rw.current_state.get_frame_transform(base_frame)
                T_tip = scene_rw.current_state.get_global_link_transform(tip_link)
            T_g2b = relative_transform(T_base, T_tip)
            g_pos = T_g2b[:3, 3].astype(float)
            g_q = np.array(_rot_to_quat(T_g2b[:3, :3]), dtype=float)
            gripper_pose7 = np.concatenate([g_pos, g_q])

            # Sample the latest tracker pose right after motion execution
            rclpy.spin_once(sub_node, timeout_sec=0.05)
            if latest_tracker.msg is not None:
                tracker_pose = latest_tracker.msg.pose
                t_pos = np.array([tracker_pose.position.x, tracker_pose.position.y, tracker_pose.position.z], dtype=float)
                t_q = np.array([tracker_pose.orientation.x, tracker_pose.orientation.y, tracker_pose.orientation.z, tracker_pose.orientation.w], dtype=float)
                tracker_pose7 = np.concatenate([t_pos, t_q])

                if _pose7_is_finite(tracker_pose7) and _pose7_is_finite(gripper_pose7):
                    achieved_list.append(gripper_pose7)
                    us_tracker_list.append(tracker_pose7)
                else:
                    logger.warning("Detected NaN/invalid in tracker or gripper pose; dropping this pair.")
            else:
                logger.warning("No us_tracker PoseStamped received; dropping this pair.")

            logger.info(f"Step {i+1}/{len(targets)} complete: "
                        f"pos=({g_pos[0]*1000:.2f},{g_pos[1]*1000:.2f},{g_pos[2]*1000:.2f}) mm")
            logger.info(f"Kept pairs so far: {len(achieved_list)}")

        logger.info("Pose sequence completed.")

        # Convert to a single numpy array for downstream use
        achieved_np = np.vstack(achieved_list) if achieved_list else np.empty((0, 7), dtype=float)
        tracker_np = np.vstack(us_tracker_list) if us_tracker_list else np.empty((0, 7), dtype=float)
        logger.info(f"Collected {achieved_np.shape[0]} gripper poses in memory (shape: {achieved_np.shape}).")
        logger.info(f"Collected {tracker_np.shape[0]} tracker poses in memory (shape: {tracker_np.shape}).")

        # ---------------------- Quick sanity diagnostics ----------------------
        def _step_lengths_m(P: np.ndarray) -> np.ndarray:
            return np.linalg.norm(P[1:, 0:3] - P[:-1, 0:3], axis=1) if len(P) > 1 else np.array([])

        gripper_steps = _step_lengths_m(achieved_np)
        tracker_steps = _step_lengths_m(tracker_np)
        if gripper_steps.size > 0 and tracker_steps.size > 0:
            gripper_med = float(np.median(gripper_steps))
            tracker_med = float(np.median(tracker_steps))
            if gripper_med > 0.0:
                ratio = tracker_med / gripper_med
                if ratio > 5.0 or ratio < 0.2:
                    logger.warning(
                        "Tracker step length is %.1fx of gripper step length (median). Possible unit mismatch (mm vs m) or wrong frames.",
                        ratio,
                    )
                    logger.warning("NDI SDKs commonly report positions in millimeters; ensure your driver publishes meters (SI).")

        # ---------------------- Relative motions for calibration quality check ----------------------
        # Build A_i (gripper motion) and B_i (tracker motion) as consecutive relative transforms.
        # The tracker motion is inverted.
        # Source: https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
        if achieved_np.shape[0] >= 2 and tracker_np.shape[0] >= 2:
            T_gripper_seq = [_pose7_to_T(p) for p in achieved_np]
            T_camera_seq = [_invert_T(_pose7_to_T(p)) for p in tracker_np]

            # Ensure equal length pairing
            n = min(len(T_gripper_seq), len(T_camera_seq))
            T_gripper_seq = T_gripper_seq[:n]
            T_camera_seq = T_camera_seq[:n]

            A_list = [_invert_T(T_gripper_seq[i + 1]) @ T_gripper_seq[i] for i in range(n - 1)]
            B_list = [T_camera_seq[i + 1] @ _invert_T(T_camera_seq[i]) for i in range(n - 1)]

            # Stack to contiguous arrays for downstream solvers (AX = XB, AX = YB, etc.)
            A_array = np.stack(A_list, axis=0) if A_list else np.empty((0, 4, 4), dtype=float)
            B_array = np.stack(B_list, axis=0) if B_list else np.empty((0, 4, 4), dtype=float)

            logger.info(f"Built {A_array.shape[0]} paired relative motions (A_i, B_i) for AX=XB-style solvers.")
        else:
            A_array = np.empty((0, 4, 4), dtype=float)
            B_array = np.empty((0, 4, 4), dtype=float)
            logger.warning("Not enough pose pairs to build relative motions (need ≥2).")

        # ---------------------- OpenCV hand–eye (eye-in-hand) ----------------------
        # Use OpenCV's calibrateHandEye solver to estimate: ^gT_t  (tracker wrt gripper).
        # Here we use the eye-in-hand model, where the camera is fixed on the gripper.
        # In our settings, the Polaris sensor is static in the world, and the tracked marker is
        # mounted on the robot end effector. Thus, we use the eye-in-hand solver with
        # inverted roles: the tracker is the "camera" and the Polaris sensor is the "target".
        # The collected tracker poses in the Polaris frame should be inverted before feeding
        # into the solver.
        # API expects: R_gripper2base, t_gripper2base, R_target2cam, t_target2cam.
        # Naming example:
        # T_g2b means "gripper to base" (end effector in robot base frame)
        # T_c2t means "camera to target" (Polaris in tracker frame)
        try:
            nA = achieved_np.shape[0]
            nB = tracker_np.shape[0]
            n = min(nA, nB)
            if n < 3:
                logger.warning("Need at least 3 paired poses for hand–eye; skipping calibration.")
            else:
                R_g2b_list, t_g2b_list = [], []
                R_t2c_list, t_t2c_list = [], []

                for i in range(n):
                    camera_pose7 = tracker_np[i]
                    T_t2c = _invert_T(_pose7_to_T(camera_pose7))
                    R_t2c_list.append(T_t2c[:3, :3].astype(np.float64))
                    t_t2c_list.append(T_t2c[:3, 3].astype(np.float64))

                    gripper_pose7 = achieved_np[i]
                    T_g2b = _pose7_to_T(gripper_pose7)
                    R_g2b_list.append(T_g2b[:3, :3].astype(np.float64))
                    t_g2b_list.append(T_g2b[:3, 3].astype(np.float64))

                # Solve Hand–Eye (eye-in-hand)
                R_c2g, t_c2g= cv2.calibrateHandEye(
                    R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list,
                    method=cv2.CALIB_HAND_EYE_PARK,
                )

                # Build transforms
                T_c2g = np.eye(4); T_c2g[:3, :3] = R_c2g; T_c2g[:3, 3] = t_c2g.flatten()

                # Report results
                def fmt_T(T: np.ndarray) -> str:
                    return "\n" + "\n".join(["  [" + ", ".join(f"{v: .6f}" for v in row) + "]" for row in T])

                logger.info(f"Calibration result: ^gT_c (tracker wrt end effector) = {fmt_T(T_c2g)}")

                # ---------------------- AX=XB algebraic consistency check ----------------------
                try:
                    if A_array.shape[0] > 0 and B_array.shape[0] > 0:
                        res = axxb_residuals(A_array, B_array, T_c2g)
                        axxb_print_summary(res)
                    else:
                        logger.warning("AX=XB residuals skipped: need relative motion stacks A_array and B_array.")
                except Exception as _e:
                    logger.error(f"AX=XB residual computation failed: {_e}")

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