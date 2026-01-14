#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from bisect import bisect_left
import xml.etree.ElementTree as ET

import numpy as np
import cv2
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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

    # 1) Prefer a specific Device Id if provided
    if device_id is not None:
        for elem in root.iter():
            if elem.tag.endswith("Device") and elem.attrib.get("Id", "") == device_id:
                if "LocalTimeOffsetSec" in elem.attrib:
                    return float(elem.attrib["LocalTimeOffsetSec"])
        raise ValueError(f"LocalTimeOffsetSec not found on Device Id='{device_id}' in {xml_path}")

    # 2) Otherwise return the first occurrence
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


# -------------------------
# Pose helpers
# -------------------------
def pose_to_vec7(pose_dict):
    p = pose_dict["position"]
    q = pose_dict["orientation"]
    return np.array([p["x"], p["y"], p["z"], q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)


# -------------------------
# Features schema (colleague style)
# -------------------------
def lab_features_schema_fixed_hw():
    return {
        "observation.description": {"dtype": "string", "shape": (1,), "names": ["task_description"]},  # TODO
        "observation.images.ultrasound": {"dtype": "video", "shape": (1080, 1920, 3), "names": ["height", "width", "channels"]},
        "observation.images.room_rgb_camera": {"dtype": "video", "shape": (768, 1024, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_camera_depth": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist_camera_rgb": {"dtype": "video", "shape": (720, 1280, 3), "names": ["height", "width", "channels"]},
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
                # 0~5 joints
                "probe_ur_x", "probe_ur_y", "probe_ur_z", "probe_ur_ux", "probe_ur_uy", "probe_ur_uz", "probe_ur_w",
                # 6~12 tcp pose
                "probe_ndi_x", "probe_ndi_y", "probe_ndi_z", "probe_ndi_ux", "probe_ndi_uy", "probe_ndi_uz", "probe_ndi_w",
                # 13~19 probe ndi
                "needle_ndi_x", "needle_ndi_y", "needle_ndi_z", "needle_ndi_ux", "needle_ndi_uy", "needle_ndi_uz", "needle_ndi_w",
                # 20~26 needle ndi
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
        "observation.meta.roomcam_cali_mtx_tracker_to_color": {"dtype": "float32", "shape": (7,), "names": ["tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"]},
        # roomcam_cali_mtx_tracker_to_color: [-14.8879, 34.6886, -65.7274, 0.709725, 0.704474, -0.001833, 0.002027]
        "observation.meta.wristcam_cali_mtx": {"dtype": "float32", "shape": (7,), "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"]},
        # wristcam_cali_mtx: see /task_info 'T_c2g'
        # OR: read from hand_eye_xxxxx.json 'T_c2g'
        "observation.meta.prob_cali_mtx": {"dtype": "float32", "shape": (7,), "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"]},
        # prob_cali_mtx: see /task_info 'T_probe_from_image'
        # OR: read from PlusDeviceSet_fCal_xxxxx.xml 'Transform From="Image" To="Probe"'
        "observation.meta.tip_offset_mm": {"dtype": "float32", "shape": (3,), "names": ['x', 'y', 'z']},
        # tip_offset_mm: see /task_info 'tip_offset_mm'
        # OR: read from needle_1_tip_offset.json 'tip_offset_mm'
        # REMOVED: "observation.meta.wristcam_cali_mtx_depth_to_color": {"dtype": "float32", "shape": (7,), "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"]},
    }


def check_frame_shape(name: str, arr: np.ndarray, expected_shape):
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {expected_shape}")


# -------------------------
# Conversion for one episode
# -------------------------
def convert_one_episode(
    episode_dir: Path,
    dataset: LeRobotDataset,
    expected_keys: set,
    static_meta: dict,
    task_name: str,
    task_index: int,
    episode_index: int,
    global_index_start: int,
    offset_sec: float,
    max_frames=None,
):
    # folders
    folder_room = "vega_vt__image_raw"
    folder_us = "image_raw__compressed"
    folder_wrgb = "camera__camera__color__image_raw__compressed"
    folder_wdepth = "camera__camera__depth__image_rect_raw__compressedDepth"

    # load video_info for fps
    room_info = load_video_info(episode_dir / folder_room / "video_info.json")
    us_info = load_video_info(episode_dir / folder_us / "video_info.json")
    wrgb_info = load_video_info(episode_dir / folder_wrgb / "video_info.json")
    wdepth_info = load_video_info(episode_dir / folder_wdepth / "video_info.json")

    room_fps = room_info["fps"] if room_info["fps"] > 1e-3 else 30.0
    us_fps = us_info["fps"] if us_info["fps"] > 1e-3 else room_fps
    wrgb_fps = wrgb_info["fps"] if wrgb_info["fps"] > 1e-3 else room_fps
    wdepth_fps = wdepth_info["fps"] if wdepth_info["fps"] > 1e-3 else room_fps

    # compute per-stream offset frames (ceil for safety)
    us_drop_front = int(math.ceil(max(offset_sec, 0.0) * us_fps))
    room_drop_back = int(math.ceil(max(offset_sec, 0.0) * room_fps))
    wrgb_drop_back = int(math.ceil(max(offset_sec, 0.0) * wrgb_fps))
    wdepth_drop_back = int(math.ceil(max(offset_sec, 0.0) * wdepth_fps))

    # decode videos
    room_frames = read_video_frames_rgb(episode_dir / folder_room / "video.mp4", max_frames=max_frames)
    us_frames = read_video_frames_rgb(episode_dir / folder_us / "video.mp4", max_frames=max_frames)
    wrgb_frames = read_video_frames_rgb(episode_dir / folder_wrgb / "video.mp4", max_frames=max_frames)
    wdepth_frames = read_video_frames_rgb(episode_dir / folder_wdepth / "video.mp4", max_frames=max_frames)

    # apply trimming rule
    if us_drop_front > 0:
        if len(us_frames) <= us_drop_front + 1:
            raise RuntimeError(f"{episode_dir.name}: ultrasound too short for us_drop_front={us_drop_front}")
        us_frames = us_frames[us_drop_front:]

    if room_drop_back > 0:
        if len(room_frames) <= room_drop_back + 1:
            raise RuntimeError(f"{episode_dir.name}: room video too short for room_drop_back={room_drop_back}")
        room_frames = room_frames[:-room_drop_back]

    if wrgb_drop_back > 0:
        if len(wrgb_frames) <= wrgb_drop_back + 1:
            raise RuntimeError(f"{episode_dir.name}: wrist_rgb too short for wrgb_drop_back={wrgb_drop_back}")
        wrgb_frames = wrgb_frames[:-wrgb_drop_back]

    if wdepth_drop_back > 0:
        if len(wdepth_frames) <= wdepth_drop_back + 1:
            raise RuntimeError(f"{episode_dir.name}: wrist_depth too short for wdepth_drop_back={wdepth_drop_back}")
        wdepth_frames = wdepth_frames[:-wdepth_drop_back]

    # after trimming, align by shortest
    T = min(len(room_frames), len(us_frames), len(wrgb_frames), len(wdepth_frames))
    if T < 2:
        raise RuntimeError(f"{episode_dir.name}: too short after trimming (T={T})")

    room_frames = room_frames[:T]
    us_frames = us_frames[:T]
    wrgb_frames = wrgb_frames[:T]
    wdepth_frames = wdepth_frames[:T]

    # master timeline: room start_time_ns, dt based on room_fps
    dt_ns = int(round(1e9 / room_fps))
    start_time_ns = int(room_info["start_time_ns"])
    frame_times_ns = np.asarray([start_time_ns + i * dt_ns for i in range(T)], dtype=np.int64)

    # ndjson streams
    t_joint, joint_payloads = load_ndjson_times_and_payloads(episode_dir / "joint_states" / "messages.ndjson")
    t_tcp, tcp_payloads = load_ndjson_times_and_payloads(episode_dir / "tcp_pose_broadcaster__pose" / "messages.ndjson")
    t_probe, probe_payloads = load_ndjson_times_and_payloads(episode_dir / "ndi__us_probe_pose" / "messages.ndjson")
    t_needle, needle_payloads = load_ndjson_times_and_payloads(episode_dir / "ndi__needle_pose" / "messages.ndjson")
    t_wrench, wrench_payloads = load_ndjson_times_and_payloads(episode_dir / "ati_ft_broadcaster__wrench" / "messages.ndjson")

    # precompute pose seqs (aligned to master timeline)
    # - tcp_pose_seq   -> goes into state as probe_ur_* and is used for probe_delta_* in action
    # - probe_pose_seq -> goes into state as probe_ndi_* (NDI probe marker)
    # - needle_pose_seq-> goes into state as needle_ndi_* and is used for needle_delta_* in action
    tcp_pose_seq = np.zeros((T, 7), dtype=np.float32)
    probe_pose_seq = np.zeros((T, 7), dtype=np.float32)
    needle_pose_seq = np.zeros((T, 7), dtype=np.float32)
    for i in range(T):
        t_ns = int(frame_times_ns[i])
        tcp_msg = nearest_payload(t_tcp, tcp_payloads, t_ns)
        probe_msg = nearest_payload(t_probe, probe_payloads, t_ns)
        needle_msg = nearest_payload(t_needle, needle_payloads, t_ns)
        tcp_pose_seq[i] = pose_to_vec7(tcp_msg["pose"])
        probe_pose_seq[i] = pose_to_vec7(probe_msg["pose"])
        needle_pose_seq[i] = pose_to_vec7(needle_msg["pose"])

    # write frames
    for i in range(T):
        t_ns = int(frame_times_ns[i])

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

        # pose vectors (already aligned)
        tcp_pose7 = tcp_pose_seq[i]          # probe_ur_* in state
        probe_pose7 = probe_pose_seq[i]      # probe_ndi_* in state
        needle_pose7 = needle_pose_seq[i]    # needle_ndi_* in state

        state = np.concatenate([qpos, tcp_pose7, probe_pose7, needle_pose7], axis=0).astype(np.float32)
        if state.shape != (27,):
            raise ValueError(f"{episode_dir.name}: state shape mismatch {state.shape}")

        # -------------------------------------------------
        # ACTION (NEW): computed as simple difference of *state* components
        # Mapping requested:
        #   probe_delta_*   = next(probe_ur_*)   - prev(probe_ur_*)    -> tcp_pose7 delta
        #   needle_delta_*  = next(needle_ndi_*)- prev(needle_ndi_*)  -> needle_pose7 delta
        # NOTE: raw component-wise delta, including quaternion components.
        # -------------------------------------------------
        if i < T - 1:
            probe_delta = (tcp_pose_seq[i + 1] - tcp_pose_seq[i]).astype(np.float32)
            needle_delta = (needle_pose_seq[i + 1] - needle_pose_seq[i]).astype(np.float32)
        else:
            probe_delta = np.zeros((7,), dtype=np.float32)
            needle_delta = np.zeros((7,), dtype=np.float32)
        action = np.concatenate([probe_delta, needle_delta], axis=0).astype(np.float32)

        img_us = us_frames[i]
        img_room = room_frames[i]
        img_wdepth = wdepth_frames[i]
        img_wrgb = wrgb_frames[i]

        check_frame_shape("ultrasound", img_us, (1080, 1920, 3))
        check_frame_shape("room_rgb_camera", img_room, (768, 1024, 3))
        check_frame_shape("wrist_camera_depth", img_wdepth, (480, 848, 3))
        check_frame_shape("wrist_camera_rgb", img_wrgb, (720, 1280, 3))

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

        # IMPORTANT: always keep task for your LeRobot version
        frame["task"] = str(task_name)

        keep = set(expected_keys) | {"task"}
        frame = {k: v for k, v in frame.items() if k in keep}

        dataset.add_frame(frame)

    dataset.save_episode()
    return T, {
        "offset_sec": offset_sec,
        "us_drop_front": us_drop_front,
        "room_drop_back": room_drop_back,
        "wrgb_drop_back": wrgb_drop_back,
        "wdepth_drop_back": wdepth_drop_back,
        "room_fps": room_fps,
        "us_fps": us_fps,
        "wrgb_fps": wrgb_fps,
        "wdepth_fps": wdepth_fps,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="local/autonomous_needle_insertion")
    parser.add_argument("--robot_type", type=str, default="panda")
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--image_writer_processes", type=int, default=16)
    parser.add_argument("--image_writer_threads", type=int, default=20)
    parser.add_argument("--tolerance_s", type=float, default=0.1)

    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--max_frames", type=int, default=None)

    # XML offset
    parser.add_argument("--calib_xml", type=str, required=True)
    parser.add_argument("--offset_device_id", type=str, default="TrackerDevice",
                        help="Which <Device Id='...'> to read LocalTimeOffsetSec from. Default: TrackerDevice. Use '' to disable.")
    parser.add_argument("--override_offset_sec", type=float, default=None,
                        help="If provided, override LocalTimeOffsetSec read from XML.")

    # Static meta placeholders
    parser.add_argument("--probe_type", type=str, default="UNKNOWN")
    parser.add_argument("--probe_acq", type=float, nargs=6, default=[0, 0, 0, 0, 0, 0])
    parser.add_argument("--roomcam_cali", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1])
    parser.add_argument("--wristcam_cali", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1])
    parser.add_argument("--wristcam_depth2color", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1])
    parser.add_argument("--probe_cali", type=float, nargs=7, default=[0, 0, 0, 0, 0, 0, 1])

    # Task
    parser.add_argument("--task_name", type=str, default="autonomous_needle_insertion")
    parser.add_argument("--task_index", type=int, default=0)

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

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

    # resolve episodes
    if raw_root.is_dir() and raw_root.name.startswith("rosbag2_"):
        episode_dirs = [raw_root]
    else:
        episode_dirs = sorted([p for p in raw_root.glob("rosbag2_*") if p.is_dir()])

    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: args.max_episodes]

    if len(episode_dirs) == 0:
        raise RuntimeError(f"No episode folders found under: {raw_root}")

    # create dataset
    features = lab_features_schema_fixed_hw()
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

    static_meta = {
        "observation.meta.probe_type": str(args.probe_type),
        "observation.meta.probe_acquisition_param": np.asarray(args.probe_acq, dtype=np.float32),
        "observation.meta.roomcam_cali_mtx_tracker_to_color": np.asarray(args.roomcam_cali, dtype=np.float32),
        "observation.meta.wristcam_cali_mtx": np.asarray(args.wristcam_cali, dtype=np.float32),
        "observation.meta.wristcam_cali_mtx_depth_to_color": np.asarray(args.wristcam_depth2color, dtype=np.float32),
        "observation.meta.prob_cali_mtx": np.asarray(args.probe_cali, dtype=np.float32),
    }

    global_index = 0
    for ep_i, ep in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        try:
            written_T, dbg = convert_one_episode(
                episode_dir=ep,
                dataset=dataset,
                expected_keys=expected_keys,
                static_meta=static_meta,
                task_name=args.task_name,
                task_index=args.task_index,
                episode_index=ep_i,
                global_index_start=global_index,
                offset_sec=offset_sec,
                max_frames=args.max_frames,
            )
            global_index += int(written_T)
            print(f"[INFO] {ep.name}: wrote {written_T} frames. trim={dbg}")
        except Exception as e:
            print(f"[WARN] Failed episode {ep.name}: {e}")
            continue

    dataset.finalize()
    print(f"✅ Done. LeRobot dataset saved at: {out_root}")


if __name__ == "__main__":
    main()
