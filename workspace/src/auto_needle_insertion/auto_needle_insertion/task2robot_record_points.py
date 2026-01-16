import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import rclpy

from auto_needle_insertion.utils.us_probe import USProbe
from auto_needle_insertion.utils.optical_tracking import read_instrument_pose


# ----------------- Math helpers -----------------
def _check_hmat(T: np.ndarray, name: str = "T") -> None:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"{name} must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError(f"{name} contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError(f"{name} last row must be [0 0 0 1], got {T[3, :]}")


def quat_to_T(quat):
    px, py, pz, qx, qy, qz, qw = quat
    vals = np.array([px, py, pz, qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Pose contains NaN/Inf: {vals.tolist()}")
    T = np.eye(4, dtype=float)
    # rotation matrix from quaternion (xyzw)
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    T[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )
    T[:3, 3] = [px, py, pz]
    return T


# ----------------- Data container -----------------
@dataclass
class RecordedPoints:
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    P3: np.ndarray | None = None


# ----------------- Core logic -----------------
class PointRecorder:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("task2_record_points")

        self.us_probe = USProbe()
        calib_root = Path(__file__).resolve().parents[3] / "calibration"  # repo-root calibration dir
        xml_path = calib_root / "PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml"
        hand_eye_path = calib_root / "hand_eye_20251231_075559.json"
        self.us_probe.load_calibrations(xml_path, hand_eye_path)
        if self.us_probe.to_in_probe is None:
            raise RuntimeError("US probe calibration failed (to_in_probe missing).")

    def capture_to_in_tracker(self, label: str) -> np.ndarray:
        pose = read_instrument_pose(instrument="us_probe", node=self.node, timeout_sec=2.0)
        to_in_tracker = quat_to_T(pose) @ self.us_probe.to_in_probe
        _check_hmat(to_in_tracker, f"to_in_tracker_{label}")
        print(f"Captured {label}.")
        return to_in_tracker

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


# ----------------- Helpers -----------------
def _prompt(msg: str) -> None:
    if not sys.stdin or not sys.stdin.isatty():
        raise SystemExit("This script requires an interactive terminal for prompts.")
    input(msg)


def _timestamp_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ----------------- CLI -----------------
def main() -> None:
    recorder = PointRecorder()
    points = RecordedPoints()
    try:
        _prompt("Press Enter to record P1...")
        points.P1 = recorder.capture_to_in_tracker("P1")

        _prompt("Press Enter to record P2...")
        points.P2 = recorder.capture_to_in_tracker("P2")

        _prompt("Press Enter to record P3...")
        points.P3 = recorder.capture_to_in_tracker("P3")

        _prompt("Press Enter to save all points...")
        ts = _timestamp_str()
        out_path = Path(__file__).resolve().parent / f"task2_pose_{ts}.json"
        payload: Dict[str, object] = {
            "timestamp": ts,
            "points": {
                "P1": points.P1.tolist() if points.P1 is not None else None,
                "P2": points.P2.tolist() if points.P2 is not None else None,
                "P3": points.P3.tolist() if points.P3 is not None else None,
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved poses to {out_path}")
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
