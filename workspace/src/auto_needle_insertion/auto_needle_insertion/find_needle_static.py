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

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple
from threading import Event
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from rclpy.node import Node

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Pose
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


def load_probe_image_transform(calibration_xml_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load Image->Probe (and its inverse Probe->Image) from a PLUS-style calibration XML.

    Expected XML element:
        <Transform From="Image" To="Probe" Matrix="...16 numbers..." />

    PLUS/IGT conventions represent rigid transforms as 4x4 homogeneous matrices.
    """
    calibration_xml_path = Path(calibration_xml_path)
    if not calibration_xml_path.exists():
        raise FileNotFoundError(str(calibration_xml_path))

    root = ET.parse(str(calibration_xml_path)).getroot()

    def find_transform(frm: str, to: str) -> np.ndarray:
        elem = root.find(f".//Transform[@From='{frm}'][@To='{to}']")
        if elem is None:
            raise RuntimeError(
                f"Transform From='{frm}' To='{to}' not found in {calibration_xml_path}"
            )
        matrix_str = elem.get("Matrix")
        if not matrix_str:
            raise RuntimeError(
                f"Transform From='{frm}' To='{to}' has no Matrix attribute in {calibration_xml_path}"
            )

        values = [float(x) for x in matrix_str.replace(",", " ").split()]
        if len(values) != 16:
            raise RuntimeError(
                f"Expected 16 matrix values (4x4), got {len(values)} for From='{frm}' To='{to}'"
            )

        mat = np.array(values, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(mat)):
            raise RuntimeError(
                f"Transform From='{frm}' To='{to}' contains NaN/Inf in {calibration_xml_path}"
            )
        return mat

    # Image -> Probe (mm)
    T_probe_from_image = find_transform("Image", "Probe")

    # Convert translation to meters
    T_probe_from_image_m = T_probe_from_image.copy()
    T_probe_from_image_m[:3, 3] *= 1e-3

    # Probe -> Image (inverse) (mm)
    try:
        T_image_from_probe_m = np.linalg.inv(T_probe_from_image_m)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Probe -> Image inverse failed (singular matrix): {e}") from e

    if not np.all(np.isfinite(T_image_from_probe_m)):
        raise RuntimeError("Probe -> Image inverse contains NaN/Inf")

    # return T_probe_from_image, T_image_from_probe
    return T_image_from_probe_m


# --- Needle tip offset helper ---
def load_needle_tip_offset_mm(json_path: str | Path) -> np.ndarray:
    """Load needle tip offset (in mm) from a JSON file.
    Expected JSON schema:
        {
          "tip_offset_mm": [x, y, z],
          ... optional metrics ...
        }

    Args:
        json_path: Path to the JSON file.

    Returns:
        A numpy array of shape (3,) in millimeters: [x_mm, y_mm, z_mm].

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the JSON is missing the expected key or has invalid content.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(str(json_path))

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object at top-level in {json_path}")

    if "tip_offset_mm" not in data:
        raise RuntimeError(f"Missing 'tip_offset_mm' in {json_path}")

    tip = data["tip_offset_mm"]
    if not isinstance(tip, (list, tuple)) or len(tip) != 3:
        raise RuntimeError(
            f"'tip_offset_mm' must be a list of 3 numbers in {json_path}; got: {tip!r}"
        )

    try:
        tip_vec = np.array([float(tip[0]), float(tip[1]), float(tip[2])], dtype=float)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Invalid numeric values in 'tip_offset_mm' in {json_path}: {e}") from e

    if not np.all(np.isfinite(tip_vec)):
        raise RuntimeError(f"'tip_offset_mm' contains NaN/Inf in {json_path}")

    return tip_vec


# --- Hand-eye calibration helper ---
def load_hand_eye_transform(json_path: str | Path) -> np.ndarray:
    """Load hand-eye calibration result T_c2g from a JSON file.

    Expected JSON schema (minimum):
        {
          "T_c2g": [[...],[...],[...],[...]],
          ... optional keys like "timestamp" ...
        }

    Returns:
        T_c2g as a (4,4) numpy array (float).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the JSON is missing the expected key or has invalid content.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(str(json_path))

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object at top-level in {json_path}")

    if "T_c2g" not in data:
        raise RuntimeError(f"Missing 'T_c2g' in {json_path}")

    T = data["T_c2g"]

    # Accept either a 4x4 nested list or a flat list of 16 values
    if isinstance(T, (list, tuple)) and len(T) == 16 and not any(isinstance(x, (list, tuple)) for x in T):
        T_mat = np.array([float(x) for x in T], dtype=float).reshape(4, 4)
    else:
        try:
            T_mat = np.asarray(T, dtype=float)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid numeric values in 'T_c2g' in {json_path}: {e}") from e

        if T_mat.shape != (4, 4):
            raise RuntimeError(
                f"'T_c2g' must be a 4x4 matrix (nested list) in {json_path}; got shape {T_mat.shape}"
            )

    if not np.all(np.isfinite(T_mat)):
        raise RuntimeError(f"'T_c2g' contains NaN/Inf in {json_path}")

    return T_mat


# Instrument pose reading utilities
def read_instrument_pose(
        instrument: str = "needle",
        topic: Optional[str] = None,
        timeout_sec: float = 2.0,
        node: Optional[Node] = None,
        qos_depth: int = 1,
) -> Tuple[float, float, float, float, float, float, float]:
    """Read a single instrument pose and return it as a position+quaternion tuple.

    This helper subscribes to a `geometry_msgs/msg/PoseStamped` topic and waits
    until one message arrives (or times out). If `topic` is not provided, a
    default topic is selected based on `instrument`:

      - instrument == "needle"   -> /ndi/needle_pose
      - instrument == "us_probe" -> /ndi/us_probe_pose

    The returned pose is:
        (px, py, pz, qx, qy, qz, qw)
    where the quaternion follows ROS conventions (x, y, z, w).

    Args:
        instrument: Instrument name selector ("needle" or "us_probe").
        topic: Optional explicit topic name. If provided, it overrides `instrument`.
        timeout_sec: Maximum time to wait for one message.
        node: Optional existing rclpy Node to use. If not provided, a temporary
            node is created and destroyed inside this function.
        qos_depth: Queue depth for the subscription.

    Returns:
        Tuple (px, py, pz, qx, qy, qz, qw).

    Raises:
        ValueError: If `instrument` is unknown and `topic` is not provided.
        TimeoutError: If no message is received within `timeout_sec`.
    """
    topic_map = {
        "needle": "/ndi/needle_pose",
        "us_probe": "/ndi/us_probe_pose",
    }

    if topic is None:
        key = instrument.strip().lower()
        if key not in topic_map:
            raise ValueError(
                f"Unknown instrument '{instrument}'. Provide topic=... or use one of: {sorted(topic_map.keys())}"
            )
        topic = topic_map[key]

    owns_node = False

    # Be robust if this utility gets called outside of main().
    if not rclpy.ok():
        rclpy.init()

    if node is None:
        node = rclpy.create_node(f"{NODE_NAME}_instrument_pose_reader")
        owns_node = True

    got_msg = Event()
    last_msg: dict = {}

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=qos_depth,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    def _cb(msg: PoseStamped) -> None:
        last_msg["msg"] = msg
        got_msg.set()

    sub = node.create_subscription(PoseStamped, topic, _cb, qos)

    start = time.monotonic()
    while rclpy.ok() and not got_msg.is_set():
        remaining = timeout_sec - (time.monotonic() - start)
        if remaining <= 0.0:
            break
        # spin_once() is blocking; keep a small timeout to remain responsive.
        rclpy.spin_once(node, timeout_sec=min(0.1, remaining))

    node.destroy_subscription(sub)

    if owns_node:
        node.destroy_node()

    if not got_msg.is_set():
        raise TimeoutError(f"Timed out waiting for PoseStamped on '{topic}'")

    msg: PoseStamped = last_msg["msg"]
    p = msg.pose.position
    q = msg.pose.orientation

    vals = np.array(
        [
            float(p.x), float(p.y), float(p.z),
            float(q.x), float(q.y), float(q.z), float(q.w),
        ],
        dtype=float,
    )

    # Fail fast if tracker publishes invalid numbers (NaN/Inf).
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(
            f"Received NaN/Inf from tracker on '{topic}' for instrument='{instrument}': {vals.tolist()}"
        )

    return tuple(float(x) for x in vals)


# --- Needle tip computation helpers ---
def _quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a ROS quaternion (x,y,z,w) to a 3x3 rotation matrix.

    The input quaternion is normalized internally.

    Args:
        qx, qy, qz, qw: Quaternion components in ROS order (x, y, z, w).

    Returns:
        (3,3) rotation matrix.

    Raises:
        RuntimeError: If the quaternion norm is too small or contains NaN/Inf.
    """
    q = np.array([qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Quaternion contains NaN/Inf")

    n = float(np.dot(q, q))
    if n < 1e-12:
        raise RuntimeError(f"Quaternion norm too small: {n}")

    q *= 1.0 / np.sqrt(n)
    x, y, z, w = q  # ROS order

    # Standard unit-quaternion to rotation-matrix conversion
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),         2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz),   2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),         1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(R)):
        raise RuntimeError("Rotation matrix contains NaN/Inf")

    return R


def pose_to_homogeneous(pose: Pose) -> np.ndarray:
    """Convert geometry_msgs/Pose to a 4x4 homogeneous transform.

    Quaternion convention: (x, y, z, w) as in ROS.
    """
    # Translation
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)

    # Quaternion (x, y, z, w)
    q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Pose quaternion contains NaN/Inf")

    n = float(np.dot(q, q))
    if n < 1e-12:
        raise RuntimeError("Pose quaternion norm too small")
    q /= np.sqrt(n)

    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=float)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p

    if not np.all(np.isfinite(T)):
        raise RuntimeError("Homogeneous transform contains NaN/Inf")

    return T


def quat_to_T(quat: Tuple[float, float, float, float, float, float, float]) -> np.ndarray:
    """Convert a pose (px,py,pz,qx,qy,qz,qw) to a 4x4 homogeneous transform.

    Assumes ROS quaternion ordering (x, y, z, w).  [oai_citation:0‡ROS Docs](https://docs.ros.org/en/diamondback/api/geometry_msgs/html/msg/Pose.html?utm_source=chatgpt.com)

    Args:
        quat: (px, py, pz, qx, qy, qz, qw)

    Returns:
        T: (4,4) homogeneous transform with rotation from quaternion and translation (p).  [oai_citation:1‡MathWorks](https://www.mathworks.com/help/ros/ug/convert-a-ros-pose-message-to-homogenous-transform.html?utm_source=chatgpt.com)

    Raises:
        RuntimeError: if inputs contain NaN/Inf.
    """
    px, py, pz, qx, qy, qz, qw = quat
    vals = np.array([px, py, pz, qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Pose contains NaN/Inf: {vals.tolist()}")

    # Uses your existing helper (expects x,y,z,w)
    R = _quat_xyzw_to_rotmat(qx, qy, qz, qw)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]

    if not np.all(np.isfinite(T)):
        raise RuntimeError("Computed transform contains NaN/Inf")

    return T


# --- New helpers: rotation matrix to quaternion and homogeneous to Pose/PoseStamped ---
def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (x, y, z, w) in ROS ordering."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise RuntimeError(f"Rotation matrix must be (3,3), got {R.shape}")
    if not np.all(np.isfinite(R)):
        raise RuntimeError("Rotation matrix contains NaN/Inf")

    # Optional: tolerate small numerical drift
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation matrix is not orthonormal")

    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        # Find the largest diagonal element and proceed accordingly (more stable near 180 deg)
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Quaternion contains NaN/Inf")

    n = float(np.dot(q, q))
    if n < 1e-12 or not np.isfinite(n):
        raise RuntimeError(f"Quaternion norm too small: {n}")

    q /= np.sqrt(n)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def homogeneous_to_pose_msg(T: np.ndarray) -> Pose:
    """Convert a 4x4 homogeneous transform into a geometry_msgs/Pose message.

    The returned quaternion uses ROS ordering (x, y, z, w).
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"Transform must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Transform contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-8):
        raise RuntimeError(f"Transform last row must be [0 0 0 1], got {T[3, :]}")

    R = T[:3, :3]
    p = T[:3, 3]

    qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)

    pose = Pose()
    pose.position.x = float(p[0])
    pose.position.y = float(p[1])
    pose.position.z = float(p[2])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def homogeneous_to_pose_stamped(
    T: np.ndarray,
    frame_id: str,
    node: Optional[Node] = None,
) -> PoseStamped:
    """Convert a 4x4 homogeneous transform into a geometry_msgs/PoseStamped.

    If `node` is provided, the stamp uses `node.get_clock().now().to_msg()`.
    Otherwise, it uses `rclpy.clock.Clock().now().to_msg()`.
    """
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    if node is not None:
        ps.header.stamp = node.get_clock().now().to_msg()
    else:
        ps.header.stamp = rclpy.clock.Clock().now().to_msg()

    ps.pose = homogeneous_to_pose_msg(T)
    return ps


def probe_from_transducer_origin_pose(
    T_probe_from_image: np.ndarray,
    T_top_from_image: np.ndarray,
    T_to_from_top: np.ndarray,
    *,
    translation_in: str = "mm",
    translation_out: str = "m",
    orthonormalize_rotation: bool = False,
):
    """
    Compute the pose of the *physical* image-plane frame (TransducerOrigin, TO)
    expressed in the Probe frame.

    Inputs (PLUS/fCal conventions):
      - T_probe_from_image: Transform From="Image" To="Probe"
      - T_top_from_image:  Transform From="Image" To="TransducerOriginPixel" (TOP)
      - T_to_from_top:     Transform From="TransducerOriginPixel" To="TransducerOrigin" (TO)
        This typically encodes pixel spacing (mm per pixel), akin to standard pixel spacing concepts.  [oai_citation:1‡GitHub](https://github.com/PlusToolkit/PlusLib/issues/370)

    Math:
      T_to_from_image = T_to_from_top @ T_top_from_image
      T_probe_from_to = T_probe_from_image @ inv(T_to_from_image)

    Returns:
      (T_probe_from_to, pose_dict) where pose_dict has:
        position: (x,y,z) in translation_out
        orientation: quaternion (x,y,z,w)

    Notes:
      - If your T_probe_from_image translation is already in mm (typical), keep translation_in="mm".
      - If you need a strictly rigid pose (rotation orthonormal), set orthonormalize_rotation=True.
    """
    _check_hmat(T_probe_from_image, "T_probe_from_image")
    _check_hmat(T_top_from_image, "T_top_from_image")
    _check_hmat(T_to_from_top, "T_to_from_top")

    T_probe_from_image = np.asarray(T_probe_from_image, dtype=float)
    T_top_from_image = np.asarray(T_top_from_image, dtype=float)
    T_to_from_top = np.asarray(T_to_from_top, dtype=float)

    # Image -> TransducerOrigin (metric) via origin shift + spacing
    T_to_from_image = T_to_from_top @ T_top_from_image

    # Probe <- TO
    T_probe_from_to = T_probe_from_image @ np.linalg.inv(T_to_from_image)

    R = T_probe_from_to[:3, :3].copy()
    t = T_probe_from_to[:3, 3].copy()

    if orthonormalize_rotation:
        R = _closest_rotation_svd(R)
        T_probe_from_to[:3, :3] = R

    q_xyzw = _rot_to_quat_xyzw(R)

    # Unit conversion for translation
    scale = 1.0
    if translation_in == "mm" and translation_out == "m":
        scale = 1e-3
    elif translation_in == "m" and translation_out == "mm":
        scale = 1e3
    elif translation_in == translation_out:
        scale = 1.0
    else:
        raise ValueError(f"Unsupported translation conversion: {translation_in} -> {translation_out}")

    pose = {
        "position": {"x": float(t[0] * scale), "y": float(t[1] * scale), "z": float(t[2] * scale)},
        "orientation": {"x": float(q_xyzw[0]), "y": float(q_xyzw[1]), "z": float(q_xyzw[2]), "w": float(q_xyzw[3])},
    }

    return T_probe_from_to, pose


def get_needle_tip_pos_in_tracker(
    needle_pose_in_tracker: Tuple[float, float, float, float, float, float, float],
    needle_tip_offset_mm: np.ndarray,
    position_unit: str = "m",
) -> np.ndarray:
    """Compute needle tip position in the tracker frame.

    Assumptions:
      - `needle_marker_pose_tracker` is the pose of the *needle marker origin* in the tracker frame,
        formatted as (px, py, pz, qx, qy, qz, qw) following ROS conventions.
      - `needle_tip_offset_mm_marker` is a 3D offset vector from marker origin to needle tip,
        expressed in the *marker's local frame* in millimeters.
      - The returned tip position is expressed in the same units as the marker position.
        By default (ROS REP-103), positions are in meters.

    Args:
        needle_pose_in_tracker: (px, py, pz, qx, qy, qz, qw) in tracker frame.
        needle_tip_offset_mm: (3,) offset vector in marker frame, in mm.
        position_unit: Unit of (px,py,pz). Use "m" (default) or "mm".

    Returns:
        tip_pos_in_tracker: (3,) numpy array of needle tip position in tracker frame.

    Raises:
        RuntimeError: If inputs are invalid or contain NaN/Inf.
        ValueError: If `position_unit` is not supported.
    """
    px, py, pz, qx, qy, qz, qw = needle_pose_in_tracker
    p_marker = np.array([px, py, pz], dtype=float)
    if not np.all(np.isfinite(p_marker)):
        raise RuntimeError("Marker position contains NaN/Inf")

    tip = np.asarray(needle_tip_offset_mm, dtype=float).reshape(-1)
    if tip.shape != (3,):
        raise RuntimeError(f"needle_tip_offset_mm_marker must be shape (3,), got {tip.shape}")
    if not np.all(np.isfinite(tip)):
        raise RuntimeError("Needle tip offset contains NaN/Inf")

    # Convert offset into the same distance unit as the marker position
    if position_unit == "m":
        tip_offset = tip / 1000.0
    elif position_unit == "mm":
        tip_offset = tip
    else:
        raise ValueError("position_unit must be 'm' or 'mm'")

    R_tracker_from_marker = _quat_xyzw_to_rotmat(qx, qy, qz, qw)

    tip_pos_in_tracker = p_marker + (R_tracker_from_marker @ tip_offset)

    if not np.all(np.isfinite(tip_pos_in_tracker)):
        raise RuntimeError("Computed tip position contains NaN/Inf")

    return tip_pos_in_tracker


def align_image_to_needle_axis(
        T_image_in_tracker: np.ndarray,
        needle_marker_origin_in_tracker: np.ndarray,
        needle_tip_pos_in_tracker: np.ndarray,
        position_unit: str = "m",
) -> np.ndarray:
    """Compute the target pose in tracker after aligning the image plane to the needle axis.

    Goal:
      - Move/rotate the image plane so that it contains both points:
            P0 = needle_marker_origin_in_tracker
            P1 = needle_tip_pos_in_tracker
      - Keep the image plane's +Y axis direction unchanged in the tracker frame.

    Args:
        T_image_in_tracker: Current image plane pose in tracker frame as a (4,4)
            homogeneous transform:
                [ R  p ]
                [ 0  1 ]
        needle_marker_origin_in_tracker: (3,) point P0 in tracker frame.
        needle_tip_pos_in_tracker: (3,) point P1 in tracker frame.
        position_unit: Unit of the provided positions. Use "m" (default) or "mm".

    Returns:
        T_target_in_tracker: (4,4) homogeneous transform in tracker that places the image-plane
            in alignment with the needle axis.

    Raises:
        RuntimeError: If inputs are invalid (NaN/Inf) or the geometry is degenerate.
        ValueError: If position_unit is not supported.
    """
    # Validate input transform
    T = np.asarray(T_image_in_tracker, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_image_in_tracker must be shape (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("T_image_in_tracker contains NaN/Inf")

    # Optional: enforce proper homogeneous last row
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-9):
        raise RuntimeError(f"T_image_in_tracker last row must be [0 0 0 1], got {T[3, :]}")

    # Extract current pose
    R_cur = T[:3, :3]
    p_cur = T[:3, 3]

    # Keep current Y axis direction (in tracker frame)
    y_keep = R_cur[:, 1]
    y_norm = float(np.linalg.norm(y_keep))
    if y_norm < 1e-12 or not np.isfinite(y_norm):
        raise RuntimeError("Image plane current Y axis is invalid")
    y_keep = y_keep / y_norm

    # Input points
    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("needle_marker_origin_in_tracker and needle_tip_pos_in_tracker must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    # Units sanity (no conversion performed; only validation)
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")

    v = P1 - P0
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9 or not np.isfinite(v_norm):
        raise RuntimeError("Needle marker origin and tip are too close (degenerate line)")

    # Choose x as the component of v orthogonal to y_keep: v_proj = v - (v·y) y
    v_proj = v - float(np.dot(v, y_keep)) * y_keep
    proj_norm = float(np.linalg.norm(v_proj))

    x_cur = R_cur[:, 0]

    if proj_norm < 1e-9:
        # v is (almost) parallel to y_keep; any x orthogonal to y_keep works.
        # Keep current x but re-orthogonalize to y_keep.
        x_target = x_cur - float(np.dot(x_cur, y_keep)) * y_keep
        x_n = float(np.linalg.norm(x_target))
        if x_n < 1e-9:
            raise RuntimeError("Cannot construct X axis: current X is parallel to Y")
        x_target = x_target / x_n
    else:
        x_target = v_proj / proj_norm
        # Minimize rotation by keeping x direction consistent with current x
        if float(np.dot(x_target, x_cur)) < 0.0:
            x_target = -x_target

    # Right-handed completion
    z_target = np.cross(x_target, y_keep)
    z_n = float(np.linalg.norm(z_target))
    if z_n < 1e-9 or not np.isfinite(z_n):
        raise RuntimeError("Cannot construct Z axis (degenerate cross product)")
    z_target = z_target / z_n

    # Recompute x to enforce orthonormality
    x_target = np.cross(y_keep, z_target)
    x_n2 = float(np.linalg.norm(x_target))
    if x_n2 < 1e-9 or not np.isfinite(x_n2):
        raise RuntimeError("Cannot re-orthonormalize X axis")
    x_target = x_target / x_n2

    R_tgt = np.column_stack((x_target, y_keep, z_target))

    # Target origin at the needle marker origin (plane passes through both points by construction)
    p_tgt = P0

    # Absolute target pose in tracker
    T_tgt = np.eye(4, dtype=float)
    T_tgt[:3, :3] = R_tgt
    T_tgt[:3, 3] = p_tgt

    if not np.all(np.isfinite(T_tgt)):
        raise RuntimeError("Computed target pose contains NaN/Inf")

    return T_tgt


def center_needle_in_image(
        T_image_in_tracker: np.ndarray,
        needle_marker_origin_in_tracker: np.ndarray,
        needle_tip_pos_in_tracker: np.ndarray,
        x_center_in_plane: float = 0.0,
        y_target_in_plane: float = 0.0,
        reference: str = "tip",
        position_unit: str = "m",
) -> np.ndarray:
    """Translate the image plane (no rotation) so the needle tip lands at a desired (x,y) in the plane frame.

    Args:
        T_image_in_tracker: Current image plane pose in tracker as a (4,4) homogeneous transform.
            It maps coordinates expressed in the image-plane frame into the tracker frame.
        needle_marker_origin_in_tracker: Needle marker origin point P0 in tracker frame (3,).
        needle_tip_pos_in_tracker: Needle tip point P1 in tracker frame (3,).
        x_center_in_plane: Desired x coordinate of the needle tip in the plane frame.
        y_target_in_plane: Desired y coordinate of the needle tip in the plane frame.
        reference: Kept for backward compatibility. Must be "tip".
        position_unit: Unit of the provided positions. Use "m" (default) or "mm".

    Returns:
        T_target_in_tracker: (4,4) homogeneous transform of the *target image plane pose* in the tracker frame.
            Orientation is unchanged; translation is along the plane +X and +Y axes.
    """
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")

    T_cur = np.asarray(T_image_in_tracker, dtype=float)
    if T_cur.shape != (4, 4):
        raise RuntimeError(f"T_image_in_tracker must be shape (4,4), got {T_cur.shape}")
    if not np.all(np.isfinite(T_cur)):
        raise RuntimeError("T_image_in_tracker contains NaN/Inf")
    if not np.allclose(T_cur[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError("T_image_in_tracker last row must be [0, 0, 0, 1]")

    R_cur = T_cur[:3, :3]
    p_cur = T_cur[:3, 3]

    # Optional orthonormality check
    if not np.allclose(R_cur.T @ R_cur, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation part of T_image_in_tracker is not orthonormal")

    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("Needle points must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    ref = reference.strip().lower()
    if ref not in ("tip", "needle_tip"):
        raise ValueError("center_needle_in_image is intended to center the needle tip; set reference='tip'.")
    P_ref = P1

    # Express needle tip in the current plane frame: p_plane = R^T (p_world - p_origin)
    p_ref_plane = R_cur.T @ (P_ref - p_cur)
    if not np.all(np.isfinite(p_ref_plane)):
        raise RuntimeError("Reference point projection produced NaN/Inf")

    tol = 2e-3 if position_unit == "m" else 2.0  # 2 mm
    if abs(float(p_ref_plane[2])) > tol:
        raise RuntimeError(
            f"Reference point is not on the image plane (z={p_ref_plane[2]:.6g} {position_unit}). "
            "Make sure you have already applied the plane alignment motion."
        )

    x_ref = float(p_ref_plane[0])
    y_ref = float(p_ref_plane[1])

    dx = x_ref - float(x_center_in_plane)
    dy = y_ref - float(y_target_in_plane)

    # Translate plane along its own +X and +Y axes (expressed in tracker)
    x_axis_tracker = R_cur[:, 0]
    y_axis_tracker = R_cur[:, 1]
    p_tgt = p_cur + dx * x_axis_tracker + dy * y_axis_tracker

    T_tgt = np.eye(4, dtype=float)
    T_tgt[:3, :3] = R_cur
    T_tgt[:3, 3] = p_tgt

    if not np.all(np.isfinite(T_tgt)):
        raise RuntimeError("Computed target pose contains NaN/Inf")

    return T_tgt


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
            current_ee_pose = pose_to_homogeneous(scene.current_state.get_pose(tip_link))

        # Extract and normalize local coordinate frame
        rotation_matrix = transform_matrix[:3, :3]
        x_axis = rotation_matrix[:, 0] / np.linalg.norm(rotation_matrix[:, 0])
        y_axis = rotation_matrix[:, 1] / np.linalg.norm(rotation_matrix[:, 1])
        origin = transform_matrix[:3, 3]

        # Read and report tracker poses for probe and needle
        def _fmt_pose(name: str, pose: Tuple[float, float, float, float, float, float, float]) -> str:
            px, py, pz, qx, qy, qz, qw = pose
            return (
                f"{name}: pos=({px:+.4f}, {py:+.4f}, {pz:+.4f}) m, "
                f"quat(xyzw)=({qx:+.5f}, {qy:+.5f}, {qz:+.5f}, {qw:+.5f})"
            )

        # Load calibrations
        image_in_probe = load_probe_image_transform("./calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20251212_144952_SRIL.xml")
        # logger.info(f"probe in image: {image_in_probe}")
        # logger.info(f"image in probe: {np.linalg.inv(image_in_probe)}")
        probe_in_ee = load_hand_eye_transform("./calibration/hand_eye_20251228_124205.json")
        # logger.info(f"probe in EE: {probe_in_ee}")
        needle_tip_offset = load_needle_tip_offset_mm("./calibration/needle_1_tip_offset.json")

        # Acquire poses
        needle_pose = read_instrument_pose(instrument="needle", timeout_sec=2.0)
        # logger.info(_fmt_pose("Needle (tracker)", needle_pose))
        probe_pose = read_instrument_pose(instrument="us_probe", timeout_sec=2.0)
        # logger.info(f"probe pose: {quat_to_T(probe_pose)}")


        # image_in_probe = np.array([
        #     [0.107175, 0.872407, 0.476885, 0.103681629],
        #     [-0.211703, 0.488672, -0.846393, 0.071431616],
        #     [-0.971440, -0.010246, 0.237064, 0.019185049],
        #     [0.0, 0.0, 0.0, 1.0],
        # ], dtype=float)
        # image_in_probe = np.array([
        #         [0.123036 , 0.862034 , 0.476885 , 0.114707699],
        #         [-0.206725 , 0.478400 , -0.846390 , 0.052905636],
        #         [-0.985577 , -0.026049 , 0.237064 , -0.069138901],
        #         [0 , 0 , 0 , 1],
        # ], dtype=float)
        # image_in_probe = np.array([
        #     [0.107175, 0.872407, 0.476885, 0.114707699],
        #     [-0.211703, 0.488672, -0.846393, 0.052905636],
        #     [-0.971440, -0.010246, 0.237064, -0.069138901],
        #     [0.0, 0.0, 0.0, 1.0],
        # ], dtype=float)
        # image_in_probe = np.array([
        #     [0.0254156, 0.8490541, 0.5276912, 0.0876791],
        #     [-0.0739174, 0.5280166, -0.8460127, 0.0255441],
        #     [-0.9969407, -0.0175038, 0.0761794, -0.0664658],
        #     [0.0, 0.0, 0.0, 1.0]
        # ], dtype=float)
        image_in_probe = np.array([
            [0.02541557, 0.84905477, 0.52769313, 0.08767910],
            [-0.07391722, 0.52801478, -0.84601220, 0.02554410],
            [-0.99694047, -0.01750372, 0.07617956, -0.06646580],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        image_in_ee = probe_in_ee @ image_in_probe
        logger.info(f"Image in EE: {image_in_ee}")
        needle_tip_position = get_needle_tip_pos_in_tracker(needle_pose, needle_tip_offset)
        # logger.info(f"Needle tip (tracker): {needle_tip_position}")

        # Calibrate tracker frame and robot base frame
        tracker_in_base = current_ee_pose @ probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))
        # logger.info(f"tracker in base: {tracker_in_base}")
        logger.info(f"current ee pose: {current_ee_pose}")

        image_in_tracker = quat_to_T(probe_pose) @ image_in_probe
        # logger.info(f"image in tracker: {image_in_tracker}")
        image_in_tracker_after_alignment = align_image_to_needle_axis(image_in_tracker, needle_pose[0:3], needle_tip_position)
        # logger.info(f"image pose when aligned: {image_in_tracker_after_alignment}")
        image_in_tracker_after_centering = center_needle_in_image(image_in_tracker_after_alignment, needle_pose[0:3], needle_tip_position, x_center_in_plane=0.0, y_target_in_plane=0.060)
        # logger.info(f"image pose when centered: {image_in_tracker_after_centering}")

        ee_target_pose_in_base = tracker_in_base @ image_in_tracker_after_centering @ np.linalg.inv(image_in_ee)
        logger.info(f"Target pose of EE in base: {ee_target_pose_in_base}")

        pose_goal = homogeneous_to_pose_stamped(ee_target_pose_in_base, planning_frame)
        arm.set_start_state_to_current_state()
        arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=tip_link)

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

