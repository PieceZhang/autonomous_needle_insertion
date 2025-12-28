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


def load_probe_image_transforms(calibration_xml_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
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

    # Probe -> Image (inverse) (mm)
    try:
        T_image_from_probe = np.linalg.inv(T_probe_from_image)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Probe -> Image inverse failed (singular matrix): {e}") from e

    if not np.all(np.isfinite(T_image_from_probe)):
        raise RuntimeError("Probe -> Image inverse contains NaN/Inf")

    return T_probe_from_image, T_image_from_probe


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


# Instrument pose reading utilities
def get_instrument_pose(
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
    return (
        float(p.x), float(p.y), float(p.z),
        float(q.x), float(q.y), float(q.z), float(q.w),
    )


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


def compute_needle_tip_position_in_tracker(
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


def compute_image_plane_relative_motion_from_points(
    image_plane_pose_in_tracker: Tuple[float, float, float, float, float, float, float],
    needle_marker_origin_in_tracker: np.ndarray,
    needle_tip_pos_in_tracker: np.ndarray,
    position_unit: str = "m",
) -> np.ndarray:
    """Compute relative motion (target-from-current) for the image plane.

    Goal:
      - Move/rotate the image plane so that it contains both points:
            P0 = needle_marker_origin_in_tracker
            P1 = needle_tip_pos_in_tracker
      - Keep the image plane's +Y axis direction unchanged in the tracker frame.

    Construction:
      - Let y_keep be the current image plane y-axis expressed in the tracker frame.
      - Let v = (P1 - P0). To ensure both points lie in the plane, v must lie in the span of
        the plane x- and y-axes. Since y is fixed, choose x as the component of v orthogonal
        to y_keep (i.e., v projected onto the plane normal to y_keep).
      - Define z = x × y for a right-handed frame.
      - Choose the target plane origin to be P0 (needle marker origin), which guarantees
        P0 lies on the plane and (by construction) P1 lies on the plane.

    Args:
        image_plane_pose_in_tracker: Current image plane pose in tracker frame as
            (px, py, pz, qx, qy, qz, qw) with ROS quaternion order (x,y,z,w).
        needle_marker_origin_in_tracker: (3,) point P0 in tracker frame.
        needle_tip_pos_in_tracker: (3,) point P1 in tracker frame.
        position_unit: Unit of the provided positions. Use "m" (default) or "mm".

    Returns:
        T_target_from_current: (4,4) homogeneous transform that maps coordinates from the
            current image-plane frame into the target image-plane frame.

    Raises:
        RuntimeError: If inputs are invalid (NaN/Inf) or the geometry is degenerate.
        ValueError: If position_unit is not supported.
    """
    # Current pose
    px, py, pz, qx, qy, qz, qw = image_plane_pose_in_tracker
    p_cur = np.array([px, py, pz], dtype=float)
    if not np.all(np.isfinite(p_cur)):
        raise RuntimeError("Image plane current position contains NaN/Inf")

    R_cur = _quat_xyzw_to_rotmat(qx, qy, qz, qw)

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
        raise RuntimeError("needle_marker_origin_tracker and needle_tip_pos_tracker must be shape (3,)")
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

    # Relative transform: T_target_from_current = inv(T_cur) * T_tgt
    # Where T = [R p; 0 1]. In block form:
    #   R_rel = R_cur^T * R_tgt
    #   p_rel = R_cur^T * (p_tgt - p_cur)
    R_rel = R_cur.T @ R_tgt
    p_rel = R_cur.T @ (p_tgt - p_cur)

    T_rel = np.eye(4, dtype=float)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = p_rel

    if not np.all(np.isfinite(T_rel)):
        raise RuntimeError("Computed relative motion contains NaN/Inf")

    return T_rel


def compute_image_plane_inplane_centering_motion(
    image_plane_pose_in_tracker: Tuple[float, float, float, float, float, float, float],
    needle_marker_origin_in_tracker: np.ndarray,
    needle_tip_pos_in_tracker: np.ndarray,
    x_center_in_plane: float = 0.0,
    reference: str = "midpoint",
    position_unit: str = "m",
) -> np.ndarray:
    """Compute an additional *in-plane* relative motion to center the needle on the plane's vertical center line.

    Context:
      - You already oriented the image plane so the needle lies in the plane.
      - Now you want to translate the plane *within its own xOy plane* (no rotation) so that the needle is
        on the plane's vertical center line, i.e., the line x = x_center_in_plane in the plane frame.
      - The plane's +Y axis direction is preserved automatically because this is a pure translation.

    Practical interpretation:
      - "Vertical center line" means the plane-frame y-axis direction, and the center line is defined by
        x = constant.
      - This function computes a translation along the plane X axis such that a chosen needle reference
        point has x-coordinate equal to `x_center_in_plane` when expressed in the plane frame.

    Args:
        image_plane_pose_in_tracker: Current image plane pose in tracker frame as
            (px, py, pz, qx, qy, qz, qw) with ROS quaternion order (x,y,z,w).
        needle_marker_origin_in_tracker: Needle marker origin point P0 in tracker frame (3,).
        needle_tip_pos_in_tracker: Needle tip point P1 in tracker frame (3,).
        x_center_in_plane: Desired x coordinate of the needle reference point in the plane frame.
            Use 0.0 if the plane frame origin is at the image horizontal center.
        reference: Which needle point to center. One of: "origin", "tip", "midpoint" (default).
        position_unit: Unit of the provided positions. Use "m" (default) or "mm".

    Returns:
        T_target_from_current: (4,4) homogeneous transform representing the desired *relative* motion
            from current image-plane frame to target image-plane frame.
            This is a pure translation in the current plane frame: +dx along X.

    Raises:
        RuntimeError: If inputs are invalid or degenerate.
        ValueError: If `reference` or `position_unit` is not supported.
    """
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")

    # Current plane pose
    px, py, pz, qx, qy, qz, qw = image_plane_pose_in_tracker
    p_cur = np.array([px, py, pz], dtype=float)
    if not np.all(np.isfinite(p_cur)):
        raise RuntimeError("Image plane current position contains NaN/Inf")

    R_cur = _quat_xyzw_to_rotmat(qx, qy, qz, qw)

    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("Needle points must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    ref = reference.strip().lower()
    if ref == "origin":
        P_ref = P0
    elif ref == "tip":
        P_ref = P1
    elif ref == "midpoint":
        P_ref = 0.5 * (P0 + P1)
    else:
        raise ValueError("reference must be one of: 'origin', 'tip', 'midpoint'")

    # Express the reference point in the *current plane frame*.
    # p_plane = R^T (p_world - p_origin)
    p_ref_plane = R_cur.T @ (P_ref - p_cur)
    if not np.all(np.isfinite(p_ref_plane)):
        raise RuntimeError("Reference point projection produced NaN/Inf")

    # Optional sanity check: needle should already lie on the plane (z approximately 0)
    # Tolerance depends on units.
    tol = 2e-3 if position_unit == "m" else 2.0  # 2 mm
    if abs(float(p_ref_plane[2])) > tol:
        raise RuntimeError(
            f"Reference point is not on the image plane (z={p_ref_plane[2]:.6g} {position_unit}). "
            "Make sure you have already applied the plane alignment motion."
        )

    x_ref = float(p_ref_plane[0])
    dx = x_ref - float(x_center_in_plane)

    # Relative transform in the *current* plane frame is a pure translation along +X by dx.
    T_rel = np.eye(4, dtype=float)
    T_rel[0, 3] = dx

    if not np.all(np.isfinite(T_rel)):
        raise RuntimeError("Computed in-plane centering motion contains NaN/Inf")

    return T_rel


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

        # Read and report tracker poses for probe and needle
        def _fmt_pose(name: str, pose: Tuple[float, float, float, float, float, float, float]) -> str:
            px, py, pz, qx, qy, qz, qw = pose
            return (
                f"{name}: pos=({px:+.4f}, {py:+.4f}, {pz:+.4f}) m, "
                f"quat(xyzw)=({qx:+.5f}, {qy:+.5f}, {qz:+.5f}, {qw:+.5f})"
            )

        # Report instrument poses
        needle_pose = get_instrument_pose(instrument="needle", timeout_sec=2.0)
        probe_pose = get_instrument_pose(instrument="us_probe", timeout_sec=2.0)
        logger.info(_fmt_pose("Needle (tracker)", needle_pose))
        logger.info(_fmt_pose("Probe  (tracker)", probe_pose))

    except Exception as e:
        logger.error(f"Square trajectory execution failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

