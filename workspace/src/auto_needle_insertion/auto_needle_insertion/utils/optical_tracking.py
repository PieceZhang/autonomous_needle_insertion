"""Utilities for reading tracked instrument poses from optical tracking topics."""

from __future__ import annotations

import time
from threading import Event
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy


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
