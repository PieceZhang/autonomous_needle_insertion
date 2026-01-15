#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert rosbag2-exported multi-stream videos + ndjson topics into a LeRobot dataset
for task1: "Probe Placement".
"""

import argparse
import json
import math
from pathlib import Path
from bisect import bisect_left
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from tqdm import tqdm
import traceback

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------- PATCH: REALLY disable stats aggregation (patch both refs) ----------
import lerobot.datasets.compute_stats as cs
import lerobot.datasets.lerobot_dataset as lds

def _aggregate_stats_disabled(*args, **kwargs):
    return {}  # empty stats => no quantile/min/max validation

cs.aggregate_stats = _aggregate_stats_disabled
lds.aggregate_stats = _aggregate_stats_disabled
# -----------------------------------------------------------------------------


# -------------------------
# Helpers: NA placeholders
# -------------------------
def na_str() -> str:
    return "NA"

def na_floats(n: int) -> np.ndarray:
    return np.full((n,), np.nan, dtype=np.float32)

def na_mat4x4_flat() -> np.ndarray:
    return np.full((16,), np.nan, dtype=np.float32)


# -------------------------
# XML: read LocalTimeOffsetSec
# -------------------------
def read_local_time_offset_sec(xml_path: Path, device_id: str | None = None) -> float:
    """
    Parse PlusDeviceSet XML and find attribute LocalTimeOffsetSec.
    If device_id is provided, prefer <Device Id="device_id" ... LocalTimeOffsetSec="...">.
    Otherwise, return the first LocalTimeOffsetSec found during traversal.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    if device_id is not None:
        for elem in root.iter():
            if elem.tag.endswith("Device") and elem.attrib.get("Id", "") == device_id:
                if "LocalTimeOffsetSec" in elem.attrib:
                    return float(elem.attrib["LocalTimeOffsetSec"])
        raise ValueError(f"LocalTimeOffsetSec not found on Device Id='{device_id}' in {xml_path}")

    for elem in root.iter():
        if "LocalTimeOffsetSec" in elem.attrib:
            return float(elem.attrib["LocalTimeOffsetSec"])

    raise ValueError(f"LocalTimeOffsetSec not found in XML: {xml_path}")


# -------------------------
# NDJSON helpers
# -------------------------
def load_ndjson_times_and_payloads(path: Path):
    times = []
    payloads = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            times.append(int(obj["publish_time_ns"]))
            payloads.append(obj["data"])

    if len(times) == 0:
        raise ValueError(f"Empty ndjson: {path}")

    times = np.asarray(times, dtype=np.int64)
    idx = np.argsort(times)
    times = times[idx]
    payloads = [payloads[i] for i in idx]
    return times, payloads


def nearest_payload(times_ns: np.ndarray, payloads: list, t_ns: int):
    j = bisect_left(times_ns, t_ns)
    if j == 0:
        return payloads[0]
    if j >= len(times_ns):
        return payloads[-1]
    before = times_ns[j - 1]
    after = times_ns[j]
    return payloads[j - 1] if (t_ns - before) <= (after - t_ns) else payloads[j]


# -------------------------
# Video helpers
# -------------------------
def load_video_info(video_info_path: Path):
    info = json.loads(video_info_path.read_text(encoding="utf-8"))
    width = int(info["resolution"]["width"])
    height = int(info["resolution"]["height"])
    fps = float(info.get("measured_fps", 0.0))
    frame_count = int(info["frame_count"])
    start_time_ns = int(info["start_time_ns"])
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "start_time_ns": start_time_ns,
    }


def read_video_frames_rgb(video_path: Path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    n = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb.astype(np.uint8))
        n += 1
        if max_frames is not None and n >= max_frames:
            break

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return frames


def check_frame_shape(name: str, arr: np.ndarray, expected_shape):
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {expected_shape}")


def frame_times_from_info(info: dict, n_frames: int) -> np.ndarray:
    """Return per-frame timestamps (ns) for a stream based on start_time_ns and fps."""
    fps = info["fps"] if info["fps"] > 1e-6 else 30.0
    dt_ns = 1e9 / fps
    start = int(info["start_time_ns"])
    # Use float64 for accumulation, then round to int64.
    return (start + np.arange(n_frames, dtype=np.float64) * dt_ns).round().astype(np.int64)


def pick_frame_by_time(frames: list, times_ns: np.ndarray, t_ns: int) -> np.ndarray:
    """Pick nearest frame in `frames` by timestamp."""
    idx = bisect_left(times_ns, t_ns)
    if idx <= 0:
        return frames[0]
    if idx >= len(times_ns):
        return frames[-1]
    before = times_ns[idx - 1]
    after = times_ns[idx]
    j = idx - 1 if (t_ns - before) <= (after - t_ns) else idx
    return frames[j]


# -------------------------
# Pose helpers
# -------------------------
def pose_to_vec7(pose_dict):
    p = pose_dict["position"]
    q = pose_dict["orientation"]
    return np.array([p["x"], p["y"], p["z"], q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)


# -------------------------
# Robust JSON utilities
# -------------------------
def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_first_json_in_dir(d: Path) -> Path | None:
    if not d.exists() or not d.is_dir():
        return None
    cands = sorted(list(d.glob("*.json")))
    return cands[0] if cands else None


def extract_mat4x4_flat(obj) -> np.ndarray:
    """
    Accept common representations and return flattened 16 float32:
    - {"T_c2g": [[...],[...],[...],[...]]}
    - {"T_probe_from_image": [[...]]}
    - {"matrix": [[...]]}
    - [[...],[...],[...],[...]]
    - [16 floats]
    If cannot parse -> all-nan.
    """
    try:
        mat = None
        if isinstance(obj, dict):
            for k in ["T_c2g", "T_probe_from_image", "T_probe_from_image_cali_mtx", "matrix", "T", "mat", "transform"]:
                if k in obj:
                    mat = obj[k]
                    break
            if mat is None and len(obj) == 1:
                mat = list(obj.values())[0]
        else:
            mat = obj

        arr = np.asarray(mat, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr.reshape(-1).astype(np.float32)
        if arr.shape == (16,):
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
            return arr.reshape(-1)[:16].astype(np.float32)
    except Exception:
        pass
    return na_mat4x4_flat()


def extract_handeye_Tc2g_flat(handeye_json, device_key: str) -> np.ndarray:
    """
    For your hand-eye files:
      - probe_handeye_cali_mtx: device_key='probe', want ['T_c2g']
      - wristcam_handeye_cali_mtx: device_key='zed2', want ['T_c2g']

    Supports common layouts:
      A) {"probe": {"T_c2g": [[4x4]]}, "zed2": {"T_c2g": [[4x4]]}}
      B) {"T_c2g": [[4x4]]}  (fallback if device key not present)
      C) {"probe": {"T_c2g": [16 floats]}}
    Returns (16,) float32; if fail => all-nan.
    """
    if not isinstance(handeye_json, (dict, list)):
        return na_mat4x4_flat()

    if isinstance(handeye_json, dict) and device_key in handeye_json:
        sub = handeye_json.get(device_key)
        if isinstance(sub, dict):
            if "T_c2g" in sub:
                return extract_mat4x4_flat(sub["T_c2g"])
            return extract_mat4x4_flat(sub)
        return extract_mat4x4_flat(sub)

    if isinstance(handeye_json, dict) and "T_c2g" in handeye_json:
        return extract_mat4x4_flat(handeye_json["T_c2g"])

    return extract_mat4x4_flat(handeye_json)


def extract_probe_acq_param_from_spec(spec_json: dict | None) -> np.ndarray:
    """
    Supports your structure:
    {
      "probe_type": "Wisonic_C51",
      "specifications": {
        "center_frequency_mhz": "4",
        "num_elements": "NA",
        ...
      }
    }

    Output order (6):
      [center_frequency_mhz, num_elements, imaging_depth_cm, linear_fov_mm, convex_radius_mm, convex_fov_deg]
    """
    out = na_floats(6)
    if not isinstance(spec_json, dict):
        return out

    spec_block = spec_json.get("specifications", None)
    if not isinstance(spec_block, dict):
        spec_block = spec_json  # fallback

    def to_float(v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.upper() == "NA":
                return np.nan
            try:
                return float(s)
            except Exception:
                return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan

    def get_any(keys):
        for k in keys:
            if k in spec_block:
                return spec_block[k]
        for k in keys:
            if k in spec_json:
                return spec_json[k]
        return None

    out[0] = to_float(get_any(["center_frequency_mhz", "center_frequency_MHz", "center_frequency", "frequency_mhz", "frequency_MHz"]))
    out[1] = to_float(get_any(["num_elements", "elements", "n_elements"]))
    out[2] = to_float(get_any(["imaging_depth_cm", "depth_cm", "imaging_depth", "depth"]))
    out[3] = to_float(get_any(["linear_fov_mm", "linear_fov", "fov_mm", "width_mm"]))
    out[4] = to_float(get_any(["convex_radius_mm", "radius_mm"]))
    out[5] = to_float(get_any(["convex_fov_deg", "fov_deg", "scan_angle_deg"]))

    return out.astype(np.float32)


# -------------------------
# Features schema
# -------------------------
def lab_features_schema_task1_fixed_hw():
    return {
        "observation.description": {"dtype": "string", "shape": (1,), "names": ["task_description"]},
        "observation.images.ultrasound": {"dtype": "video", "shape": (1080, 1920, 3), "names": ["height", "width", "channels"]},
        "observation.images.room_rgb_camera": {"dtype": "video", "shape": (768, 1024, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_camera_depth": {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_camera_rgb": {"dtype": "video", "shape": (360, 640, 3), "names": ["height", "width", "channels"]},

        "observation.meta.force_torque": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
            "info": {"sensor": "wrist_ft", "units": "N + N m"},
        },

        "observation.state": {
            "dtype": "float32",
            "shape": (27,),
            "names": [
                "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                "probe_ur_x", "probe_ur_y", "probe_ur_z", "probe_ur_ux", "probe_ur_uy", "probe_ur_uz", "probe_ur_w",
                "probe_ndi_x", "probe_ndi_y", "probe_ndi_z", "probe_ndi_ux", "probe_ndi_uy", "probe_ndi_uz", "probe_ndi_w",
                "needle_ndi_x", "needle_ndi_y", "needle_ndi_z", "needle_ndi_ux", "needle_ndi_uy", "needle_ndi_uz", "needle_ndi_w",
            ],
        },

        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "probe_delta_x", "probe_delta_y", "probe_delta_z",
                "probe_delta_ux", "probe_delta_uy", "probe_delta_uz", "probe_delta_w",
                "needle_delta_x", "needle_delta_y", "needle_delta_z",
                "needle_delta_ux", "needle_delta_uy", "needle_delta_uz", "needle_delta_w",
            ],
        },

        "observation.meta.probe_type": {"dtype": "string", "shape": (1,), "names": ["company_model_endwith'linear'_or_'convex'"]},
        "observation.meta.probe_acquisition_param": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["center_frequency_mhz", "num_elements", "imaging_depth_cm", "linear_fov_mm", "convex_radius_mm", "convex_fov_deg"],
        },

        "observation.meta.probe_handeye_cali_mtx": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"t{i}{j}" for i in range(1, 5) for j in range(1, 5)],
        },
        "observation.meta.wristcam_handeye_cali_mtx": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"t{i}{j}" for i in range(1, 5) for j in range(1, 5)],
        },
        "observation.meta.probe_from_image_cali_mtx": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"t{i}{j}" for i in range(1, 5) for j in range(1, 5)],
        },

        "observation.meta.roomcam_cali_mtx_tracker_to_color": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"],
        },
        "observation.meta.tip_offset_mm": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
    }


# -------------------------
# Locate per-episode file-based meta
# -------------------------
def load_episode_task_description(episode_dir: Path) -> str:
    task_info_dir = episode_dir / "task_info"
    p = find_first_json_in_dir(task_info_dir)
    j = read_json_if_exists(p) if p else None
    if isinstance(j, dict):
        for k in ["description", "task_description", "prompt", "text", "note"]:
            if k in j and isinstance(j[k], str) and j[k].strip():
                return j[k].strip()
    return na_str()


def load_tip_offset_mm(episode_dir: Path, workspace_root: Path) -> np.ndarray:
    task_info_dir = episode_dir / "task_info"
    p = find_first_json_in_dir(task_info_dir)
    j = read_json_if_exists(p) if p else None
    if isinstance(j, dict):
        for k in ["tip_offset_mm", "needle_tip_offset_mm", "needle_1_tip_offset_mm"]:
            if k in j:
                arr = np.asarray(j[k], dtype=np.float32).reshape(-1)
                if arr.shape[0] >= 3:
                    return arr[:3].astype(np.float32)

    cand = workspace_root / "calibration" / "needle_1_tip_offset.json"
    jj = read_json_if_exists(cand)
    if isinstance(jj, dict) and "tip_offset_mm" in jj:
        arr = np.asarray(jj["tip_offset_mm"], dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 3:
            return arr[:3].astype(np.float32)

    return na_floats(3)


# -------------------------
# (1) roomcam tracker->color: HARD CODED vec7
# -------------------------
def load_roomcam_tracker_to_color_7_hardcoded() -> np.ndarray:
    return np.asarray(
        [-14.8879, 34.6886, -65.7274, 0.709725, 0.704474, -0.001833, 0.002027],
        dtype=np.float32,
    )


# -------------------------
# (2)(3) hand-eye matrices: load explicit files and keys
# -------------------------
def load_handeye_matrices_explicit(workspace_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    probe_handeye_cali_mtx:
      calibration/hand_eye_20251231_075559.json   -> probe:'T_c2g' -> (16,)
    wristcam_handeye_cali_mtx:
      calibration/hand_eye_20260112_071955.json   -> zed2:'T_c2g'  -> (16,)
    """
    calib = workspace_root / "calibration"

    probe_path = calib / "hand_eye_20251231_075559.json"
    wrist_path = calib / "hand_eye_20260112_071955.json"

    probe_mat = na_mat4x4_flat()
    wrist_mat = na_mat4x4_flat()

    pj = read_json_if_exists(probe_path)
    if pj is None:
        print(f"[WARN] probe hand-eye json not found or unreadable: {probe_path}")
    else:
        probe_mat = extract_handeye_Tc2g_flat(pj, device_key="probe")
        if np.isnan(probe_mat).all():
            print(f"[WARN] probe hand-eye parsed as all-NaN. Check json structure: {probe_path}")

    wj = read_json_if_exists(wrist_path)
    if wj is None:
        print(f"[WARN] wrist/zed2 hand-eye json not found or unreadable: {wrist_path}")
    else:
        wrist_mat = extract_handeye_Tc2g_flat(wj, device_key="zed2")
        if np.isnan(wrist_mat).all():
            print(f"[WARN] zed2 hand-eye parsed as all-NaN. Check json structure: {wrist_path}")

    return probe_mat.astype(np.float32), wrist_mat.astype(np.float32)


def load_probe_from_image_matrix(workspace_root: Path) -> np.ndarray:
    calib = workspace_root / "calibration"
    for p in sorted(calib.glob("probe_from_image*.json")):
        j = read_json_if_exists(p)
        m = extract_mat4x4_flat(j)
        if not np.isnan(m).all():
            return m
    for p in sorted(calib.glob("T_probe_from_image*.json")):
        j = read_json_if_exists(p)
        m = extract_mat4x4_flat(j)
        if not np.isnan(m).all():
            return m

    for xmlp in sorted(calib.glob("PlusDeviceSet*.xml")):
        try:
            tree = ET.parse(str(xmlp))
            root = tree.getroot()
            for elem in root.iter():
                if elem.attrib.get("From", "") == "Image" and elem.attrib.get("To", "") == "Probe":
                    for key in ["Matrix", "matrix", "Transform", "transform", "Value", "value"]:
                        if key in elem.attrib:
                            nums = [float(x) for x in elem.attrib[key].replace(",", " ").split()]
                            if len(nums) >= 16:
                                return np.asarray(nums[:16], dtype=np.float32)
                    if elem.text:
                        nums = [float(x) for x in elem.text.replace(",", " ").split()]
                        if len(nums) >= 16:
                            return np.asarray(nums[:16], dtype=np.float32)
        except Exception:
            pass

    return na_mat4x4_flat()


# -------------------------
# Conversion for one episode
# -------------------------
def convert_one_episode(
    episode_dir: Path,
    dataset: LeRobotDataset,
    expected_keys: set,
    static_meta: dict,
    offset_sec: float,
    max_frames=None,
    action_hz: float = 15.0,
):
    # NOTE: offset_sec is kept for debug, but video alignment is now purely time-based from video_info.json.

    folder_room  = "vega_vt__image_raw__compressed"
    folder_us    = "image_raw__compressed"
    folder_wrgb  = "zed__zed_node__rgb__color__rect__image__compressed"
    folder_wdepth= "zed__zed_node__depth__depth_registered__compressedDepth"

    room_info  = load_video_info(episode_dir / folder_room   / "video_info.json")
    us_info    = load_video_info(episode_dir / folder_us     / "video_info.json")
    wrgb_info  = load_video_info(episode_dir / folder_wrgb   / "video_info.json")
    wdepth_info= load_video_info(episode_dir / folder_wdepth / "video_info.json")

    room_fps  = room_info["fps"] if room_info["fps"] > 1e-3 else 30.0
    us_fps    = us_info["fps"] if us_info["fps"] > 1e-3 else room_fps
    wrgb_fps  = wrgb_info["fps"] if wrgb_info["fps"] > 1e-3 else room_fps
    wdepth_fps= wdepth_info["fps"] if wdepth_info["fps"] > 1e-3 else room_fps

    # ---- decode videos (full) ----
    room_frames  = read_video_frames_rgb(episode_dir / folder_room   / "video.mp4", max_frames=max_frames)
    us_frames    = read_video_frames_rgb(episode_dir / folder_us     / "video.mp4", max_frames=max_frames)
    wrgb_frames  = read_video_frames_rgb(episode_dir / folder_wrgb   / "video.mp4", max_frames=max_frames)
    wdepth_frames= read_video_frames_rgb(episode_dir / folder_wdepth / "video.mp4", max_frames=max_frames)

    # ---- build per-stream timestamps from video_info ----
    room_times_ns   = frame_times_from_info(room_info,  len(room_frames))
    us_times_ns     = frame_times_from_info(us_info,    len(us_frames))
    wrgb_times_ns   = frame_times_from_info(wrgb_info,  len(wrgb_frames))
    wdepth_times_ns = frame_times_from_info(wdepth_info,len(wdepth_frames))

    # ---- choose common overlap window [t0, t1] ----
    t0 = max(room_times_ns[0], us_times_ns[0], wrgb_times_ns[0], wdepth_times_ns[0])
    t1 = min(room_times_ns[-1], us_times_ns[-1], wrgb_times_ns[-1], wdepth_times_ns[-1])
    if t1 <= t0:
        raise RuntimeError(
            f"{episode_dir.name}: no temporal overlap across streams. "
            f"room=[{room_times_ns[0]},{room_times_ns[-1]}], "
            f"us=[{us_times_ns[0]},{us_times_ns[-1]}], "
            f"wrgb=[{wrgb_times_ns[0]},{wrgb_times_ns[-1]}], "
            f"wdepth=[{wdepth_times_ns[0]},{wdepth_times_ns[-1]}]"
        )

    # ---- output timeline at dataset fps ----
    out_fps = float(dataset.fps) if hasattr(dataset, "fps") else 30.0
    out_dt_ns = int(round(1e9 / out_fps))
    T = int((t1 - t0) // out_dt_ns) + 1
    if T < 2:
        raise RuntimeError(f"{episode_dir.name}: too short after time alignment (T={T})")

    frame_times_ns = (t0 + np.arange(T, dtype=np.int64) * out_dt_ns).astype(np.int64)

    # --- action stride for target action_hz on the OUTPUT timeline ---
    if action_hz is None or action_hz <= 1e-6:
        action_stride = 1
    else:
        action_stride = int(round(out_fps / float(action_hz)))
        action_stride = max(action_stride, 1)

    print(
        f"[INFO] {episode_dir.name}: time-align streams. "
        f"room_fps={room_fps:.3f}, us_fps={us_fps:.3f}, wrgb_fps={wrgb_fps:.3f}, wdepth_fps={wdepth_fps:.3f}, "
        f"out_fps={out_fps:.3f}, action_hz={action_hz:.3f} => action_stride={action_stride}, "
        f"overlap_sec={(t1 - t0)/1e9:.3f}, T={T}"
    )

    # keep these for dbg compatibility (now no trimming)
    us_drop_front = room_drop_back = wrgb_drop_back = wdepth_drop_back = 0

    # ndjson streams
    t_joint,  joint_payloads  = load_ndjson_times_and_payloads(episode_dir / "joint_states" / "messages.ndjson")
    t_tcp,    tcp_payloads    = load_ndjson_times_and_payloads(episode_dir / "tcp_pose_broadcaster__pose" / "messages.ndjson")
    t_probe,  probe_payloads  = load_ndjson_times_and_payloads(episode_dir / "ndi__us_probe_pose" / "messages.ndjson")
    t_needle, needle_payloads = load_ndjson_times_and_payloads(episode_dir / "ndi__needle_pose" / "messages.ndjson")
    t_wrench, wrench_payloads = load_ndjson_times_and_payloads(episode_dir / "ati_ft_broadcaster__wrench" / "messages.ndjson")

    # precompute pose seqs aligned to master timeline
    tcp_pose_seq    = np.zeros((T, 7), dtype=np.float32)
    probe_pose_seq  = np.zeros((T, 7), dtype=np.float32)
    needle_pose_seq = np.zeros((T, 7), dtype=np.float32)
    for i in range(T):
        t_ns = int(frame_times_ns[i])
        tcp_msg    = nearest_payload(t_tcp, tcp_payloads, t_ns)
        probe_msg  = nearest_payload(t_probe, probe_payloads, t_ns)
        needle_msg = nearest_payload(t_needle, needle_payloads, t_ns)
        tcp_pose_seq[i]    = pose_to_vec7(tcp_msg["pose"])
        probe_pose_seq[i]  = pose_to_vec7(probe_msg["pose"])
        needle_pose_seq[i] = pose_to_vec7(needle_msg["pose"])

    # write frames
    for i in range(T):
        t_ns = int(frame_times_ns[i])

        # pick images by time (nearest)
        img_room   = pick_frame_by_time(room_frames,  room_times_ns,   t_ns)
        img_us     = pick_frame_by_time(us_frames,    us_times_ns,     t_ns)
        img_wrgb   = pick_frame_by_time(wrgb_frames,  wrgb_times_ns,   t_ns)
        img_wdepth = pick_frame_by_time(wdepth_frames,wdepth_times_ns, t_ns)

        check_frame_shape("ultrasound",         img_us,     (1080, 1920, 3))
        check_frame_shape("room_rgb_camera",    img_room,   (768,  1024, 3))
        check_frame_shape("wrist_camera_depth", img_wdepth, (360,  640,  3))
        check_frame_shape("wrist_camera_rgb",   img_wrgb,   (360,  640,  3))

        wmsg = nearest_payload(t_wrench, wrench_payloads, t_ns)
        w = wmsg["wrench"]
        ft = np.array(
            [w["force"]["x"], w["force"]["y"], w["force"]["z"],
             w["torque"]["x"], w["torque"]["y"], w["torque"]["z"]],
            dtype=np.float32,
        )

        jmsg = nearest_payload(t_joint, joint_payloads, t_ns)
        qpos = np.asarray(jmsg["position"], dtype=np.float32)
        if qpos.shape[0] != 6:
            raise ValueError(f"{episode_dir.name}: expected 6 joints, got {qpos.shape[0]}")

        tcp_pose7    = tcp_pose_seq[i]
        probe_pose7  = probe_pose_seq[i]
        needle_pose7 = needle_pose_seq[i]

        state = np.concatenate([qpos, tcp_pose7, probe_pose7, needle_pose7], axis=0).astype(np.float32)
        if state.shape != (27,):
            raise ValueError(f"{episode_dir.name}: state shape mismatch {state.shape}")

        # ACTION: delta at target Hz using stride on output timeline
        j = i + action_stride
        if j < T:
            probe_delta  = (tcp_pose_seq[j] - tcp_pose_seq[i]).astype(np.float32)
            needle_delta = (needle_pose_seq[j] - needle_pose_seq[i]).astype(np.float32)
        else:
            probe_delta  = np.zeros((7,), dtype=np.float32)
            needle_delta = np.zeros((7,), dtype=np.float32)

        action = np.concatenate([probe_delta, needle_delta], axis=0).astype(np.float32)

        frame = {
            "observation.images.ultrasound": img_us,
            "observation.images.room_rgb_camera": img_room,
            "observation.images.wrist_camera_depth": img_wdepth,
            "observation.images.wrist_camera_rgb": img_wrgb,
            "observation.meta.force_torque": ft,
            "observation.state": state,
            "action": action,
        }
        frame.update(static_meta)

        frame["task"] = "Probe Placement"

        keep = set(expected_keys) | {"task"}
        frame = {k: v for k, v in frame.items() if k in keep}

        dataset.add_frame(frame)

    try:
        dataset.save_episode()
    except Exception as e:
        print(f"[ERROR] save_episode failed at {episode_dir.name}: {e}")
        raise

    return T, {
        "offset_sec": float(offset_sec),
        "action_hz": float(action_hz),
        "action_stride": int(action_stride),
        "us_drop_front": int(us_drop_front),
        "room_drop_back": int(room_drop_back),
        "wrgb_drop_back": int(wrgb_drop_back),
        "wdepth_drop_back": int(wdepth_drop_back),
        "room_fps": float(room_fps),
        "us_fps": float(us_fps),
        "wrgb_fps": float(wrgb_fps),
        "wdepth_fps": float(wdepth_fps),
        "out_fps": float(out_fps),
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_root", type=str, required=True,
                        help="Either a single rosbag2_* episode dir, or a directory containing rosbag2_* dirs.")
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="local/task1_probe_placement_v2")
    parser.add_argument("--robot_type", type=str, default="panda")
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--image_writer_processes", type=int, default=16)
    parser.add_argument("--image_writer_threads", type=int, default=20)
    parser.add_argument("--tolerance_s", type=float, default=0.1)

    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=None)

    # action delta target rate
    parser.add_argument("--action_hz", type=float, default=15.0,
                        help="Compute action delta at this rate (Hz) using frame stride based on OUTPUT fps. Default=15.")

    # Workspace root (for calibration files)
    parser.add_argument("--workspace_root", type=str, required=True,
                        help="Path to your workspace that contains calibration/, e.g. workspace/")
    # XML offset (kept for compatibility; video sync no longer trims by this)
    parser.add_argument("--calib_xml", type=str, required=True,
                        help="Path to PlusDeviceSet*.xml (for LocalTimeOffsetSec)")
    parser.add_argument("--offset_device_id", type=str, default="TrackerDevice",
                        help="Which <Device Id='...'> to read LocalTimeOffsetSec from. Default: TrackerDevice. Use '' to disable.")
    parser.add_argument("--override_offset_sec", type=float, default=None,
                        help="If provided, override LocalTimeOffsetSec read from XML.")

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    workspace_root = Path(args.workspace_root)

    out_root.parent.mkdir(parents=True, exist_ok=True)
    if out_root.exists():
        raise RuntimeError(f"Output root already exists: {out_root} (must NOT exist)")

    xml_path = Path(args.calib_xml)
    if args.override_offset_sec is not None:
        offset_sec = float(args.override_offset_sec)
        print(f"[INFO] override_offset_sec = {offset_sec:.6f}")
    else:
        dev = args.offset_device_id.strip()
        device_id = dev if dev != "" else None
        offset_sec = read_local_time_offset_sec(xml_path, device_id=device_id)
        print(f"[INFO] LocalTimeOffsetSec = {offset_sec:.6f} sec (from {xml_path}, device_id={device_id})")

    # Resolve episode dirs
    if raw_root.is_dir() and raw_root.name.startswith("rosbag2_"):
        episode_dirs = [raw_root]
    else:
        episode_dirs = sorted([p for p in raw_root.glob("rosbag2_*") if p.is_dir()])

    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: args.max_episodes]
    if len(episode_dirs) == 0:
        raise RuntimeError(f"No episode folders found under: {raw_root}")

    # ---- Read file-based static meta (global for this conversion) ----
    spec_path = workspace_root / "calibration" / "Spec_probe_c51.json"
    spec_json = read_json_if_exists(spec_path)
    probe_acq_param = extract_probe_acq_param_from_spec(spec_json)

    probe_type = na_str()
    if isinstance(spec_json, dict):
        v = spec_json.get("probe_type", None)
        if isinstance(v, str) and v.strip():
            probe_type = v.strip()

    probe_handeye, wrist_handeye = load_handeye_matrices_explicit(workspace_root)
    probe_from_image = load_probe_from_image_matrix(workspace_root)

    roomcam_tracker_to_color = load_roomcam_tracker_to_color_7_hardcoded()
    tip_offset_placeholder = na_floats(3)

    features = lab_features_schema_task1_fixed_hw()
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(out_root),
        robot_type=args.robot_type,
        fps=args.fps,
        use_videos=True,
        features=features,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        tolerance_s=args.tolerance_s,
    )

    expected_keys = set(dataset.features.keys())
    print("[INFO] dataset.features keys:", sorted(expected_keys))

    static_meta_global = {
        "observation.meta.probe_type": str(probe_type),
        "observation.meta.probe_acquisition_param": probe_acq_param.astype(np.float32),
        "observation.meta.probe_handeye_cali_mtx": probe_handeye.astype(np.float32),
        "observation.meta.wristcam_handeye_cali_mtx": wrist_handeye.astype(np.float32),
        "observation.meta.probe_from_image_cali_mtx": probe_from_image.astype(np.float32),
        "observation.meta.roomcam_cali_mtx_tracker_to_color": roomcam_tracker_to_color.astype(np.float32),
        "observation.meta.tip_offset_mm": tip_offset_placeholder.astype(np.float32),
        "observation.description": na_str(),
    }

    for ep in tqdm(episode_dirs, desc="Converting episodes"):
        try:
            desc = load_episode_task_description(ep)
            tip_offset = load_tip_offset_mm(ep, workspace_root)

            static_meta = dict(static_meta_global)
            static_meta["observation.description"] = desc
            static_meta["observation.meta.tip_offset_mm"] = tip_offset.astype(np.float32)

            written_T, dbg = convert_one_episode(
                episode_dir=ep,
                dataset=dataset,
                expected_keys=expected_keys,
                static_meta=static_meta,
                offset_sec=offset_sec,
                max_frames=args.max_frames,
                action_hz=args.action_hz,
            )
            print(f"[INFO] {ep.name}: wrote {written_T} frames. dbg={dbg}")
        except Exception as e:
            print(f"[WARN] Failed episode {ep.name}: {e}")
            traceback.print_exc()
            continue

    dataset.finalize()
    print(f"✅ Done. LeRobot dataset saved at: {out_root}")


if __name__ == "__main__":
    main()
