"""Ultrasound probe utilities for tracked probe operations."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from auto_needle_insertion.utils.optical_tracking import read_instrument_pose
from auto_needle_insertion.utils.pose_representations import project_to_se3, translation_scale


class USProbe:
    """Tracked ultrasound probe helper for calibration and pose reporting."""

    def __init__(
        self,
        probe_type: str = "C5-1",
        pose_topic: str = "/ndi/us_probe_pose",
    ) -> None:
        self.probe_type = probe_type
        self.pose_topic = pose_topic

        self.image_in_probe: Optional[np.ndarray] = None
        self.image_in_top: Optional[np.ndarray] = None
        self.top_in_to: Optional[np.ndarray] = None
        self.to_in_probe: Optional[np.ndarray] = None
        self.probe_in_ee: Optional[np.ndarray] = None
        self.to_in_ee: Optional[np.ndarray] = None

    def load_image_calibration(
        self,
        calibration_xml_path: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load image/probe calibration matrices from a PLUS-style XML file."""
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

        self.image_in_probe = find_transform("Image", "Probe")
        self.image_in_top = find_transform("Image", "TransducerOriginPixel")
        self.top_in_to = find_transform("TransducerOriginPixel", "TransducerOrigin")

        return self.image_in_probe, self.image_in_top, self.top_in_to

    def compute_to_in_probe(
        self,
        *,
        translation_in: str = "mm",
        translation_out: str = "m",
        orthonormalize_rotation: bool = True,
    ) -> np.ndarray:
        """Compute transducer origin frame (TO) expressed in the probe frame."""
        if self.image_in_probe is None or self.image_in_top is None or self.top_in_to is None:
            raise RuntimeError("Image calibration matrices are not loaded. Call load_image_calibration() first.")

        to_in_probe = (
            self.image_in_probe
            @ np.linalg.inv(self.image_in_top)
            @ np.linalg.inv(self.top_in_to)
        )

        scale = translation_scale(translation_in, translation_out)
        to_in_probe = to_in_probe.copy()
        to_in_probe[:3, 3] *= scale

        if orthonormalize_rotation:
            to_in_probe = project_to_se3(to_in_probe)

        self.to_in_probe = to_in_probe
        return to_in_probe

    def load_hand_eye_transform(self, json_path: str | Path) -> np.ndarray:
        """Load and store probe-in-EE transform from a JSON file."""
        self.probe_in_ee = self._load_hand_eye_transform(json_path)
        return self.probe_in_ee

    def compute_to_in_ee(self) -> np.ndarray:
        """Compute transducer origin frame expressed in the end-effector frame."""
        if self.to_in_probe is None:
            raise RuntimeError("to_in_probe is not set. Call compute_to_in_probe() first.")
        if self.probe_in_ee is None:
            raise RuntimeError("probe_in_ee is not set. Call load_hand_eye_transform() first.")

        self.to_in_ee = self.probe_in_ee @ self.to_in_probe
        return self.to_in_ee

    def load_calibrations(
        self,
        calibration_xml_path: str | Path,
        hand_eye_json_path: Optional[str | Path] = None,
    ) -> None:
        """Load calibration files and compute TO in probe/EE if available."""
        self.load_image_calibration(calibration_xml_path)
        self.compute_to_in_probe()

        if hand_eye_json_path is not None:
            self.load_hand_eye_transform(hand_eye_json_path)
            self.compute_to_in_ee()

    def report_pose(
        self,
        timeout_sec: float = 2.0,
        node=None,
        qos_depth: int = 1,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Report the latest probe pose from the tracking topic."""
        return read_instrument_pose(
            instrument="us_probe",
            topic=self.pose_topic,
            timeout_sec=timeout_sec,
            node=node,
            qos_depth=qos_depth,
        )

    @staticmethod
    def _load_hand_eye_transform(json_path: str | Path) -> np.ndarray:
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
