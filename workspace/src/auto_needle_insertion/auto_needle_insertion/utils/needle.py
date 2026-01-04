"""Needle utilities for tracked needle operations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy


class Needle:
    """Tracked needle helper for pose and tip position utilities."""

    def __init__(
        self,
        gauge: str = "18G",
        length_mm: float = 200.0,
        pose_topic: str = "/ndi/needle_pose",
        tip_offset_mm: Optional[np.ndarray] = None,
    ) -> None:
        self.gauge = gauge
        self.length_mm = float(length_mm)
        self.pose_topic = pose_topic
        self.tip_offset_mm = tip_offset_mm

    def load_tip_offset(self, calibration_file: str | Path) -> np.ndarray:
        """Load and store needle tip offset from a calibration JSON file.

        Args:
            calibration_file: Path to the JSON file containing "tip_offset_mm".

        Returns:
            The loaded tip offset in millimeters.
        """
        self.tip_offset_mm = self._load_needle_tip_offset_mm(calibration_file)
        return self.tip_offset_mm

    def report_pose(
        self,
        timeout_sec: float = 2.0,
        node: Optional[Node] = None,
        qos_depth: int = 1,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Report the latest needle pose from the tracking topic.

        Args:
            timeout_sec: Maximum time to wait for a pose message.
            node: Optional existing ROS2 node to reuse.
            qos_depth: Queue depth for the subscription.

        Returns:
            (px, py, pz, qx, qy, qz, qw) pose tuple in the tracker frame.
        """
        return self._read_instrument_pose(
            instrument="needle",
            topic=self.pose_topic,
            timeout_sec=timeout_sec,
            node=node,
            qos_depth=qos_depth,
        )

    def tip_position_in_tracker(
        self,
        needle_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None,
        position_unit: str = "m",
    ) -> np.ndarray:
        """Compute the needle tip position in the tracker frame.

        Args:
            needle_pose: Optional pose tuple. If omitted, the pose is read
                from the tracking topic.
            position_unit: Unit for the returned position. Use "m" or "mm".

        Returns:
            (3,) numpy array for the needle tip position in the tracker frame.
        """
        if self.tip_offset_mm is None:
            raise RuntimeError("Needle tip offset is not set. Call load_tip_offset() first.")

        pose = needle_pose or self.report_pose()
        return self._get_needle_tip_pos_in_tracker(pose, self.tip_offset_mm, position_unit=position_unit)

    @staticmethod
    def _load_needle_tip_offset_mm(json_path: str | Path) -> np.ndarray:
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

    @staticmethod
    def _read_instrument_pose(
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

        if not rclpy.ok():
            rclpy.init()

        if node is None:
            node = rclpy.create_node("auto_needle_insertion_instrument_pose_reader")
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

        if not np.all(np.isfinite(vals)):
            raise RuntimeError(
                f"Received NaN/Inf from tracker on '{topic}' for instrument='{instrument}': {vals.tolist()}"
            )

        return tuple(float(x) for x in vals)

    @staticmethod
    def _quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert a ROS quaternion (x,y,z,w) to a 3x3 rotation matrix."""
        q = np.array([qx, qy, qz, qw], dtype=float)
        if not np.all(np.isfinite(q)):
            raise RuntimeError("Quaternion contains NaN/Inf")

        norm = np.linalg.norm(q)
        if norm <= 0.0:
            raise RuntimeError("Quaternion has zero norm")
        q = q / norm
        qx, qy, qz, qw = q

        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    @staticmethod
    def _get_needle_tip_pos_in_tracker(
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

        if position_unit == "m":
            tip_offset = tip / 1000.0
        elif position_unit == "mm":
            tip_offset = tip
        else:
            raise ValueError("position_unit must be 'm' or 'mm'")

        R_tracker_from_marker = Needle._quat_xyzw_to_rotmat(qx, qy, qz, qw)

        tip_pos_in_tracker = p_marker + (R_tracker_from_marker @ tip_offset)

        if not np.all(np.isfinite(tip_pos_in_tracker)):
            raise RuntimeError("Computed tip position contains NaN/Inf")

        return tip_pos_in_tracker
