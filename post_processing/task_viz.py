#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


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

def load_ndjson_optional(path: Path):
    try:
        if not path.exists():
            return None, None
        return load_ndjson_times_and_payloads(path)
    except Exception:
        return None, None

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

def frame_times_from_info(info: dict, n_frames: int) -> np.ndarray:
    fps = info["fps"] if info["fps"] > 1e-6 else 30.0
    dt_ns = 1e9 / fps
    start = int(info["start_time_ns"])
    return (start + np.arange(n_frames, dtype=np.float64) * dt_ns).round().astype(np.int64)

def pick_frame_by_time(frames: list, times_ns: np.ndarray, t_ns: int) -> np.ndarray:
    idx = bisect_left(times_ns, t_ns)
    if idx <= 0:
        return frames[0]
    if idx >= len(times_ns):
        return frames[-1]
    before = times_ns[idx - 1]
    after = times_ns[idx]
    j = idx - 1 if (t_ns - before) <= (after - t_ns) else idx
    return frames[j]

def load_video_stream_optional(episode_dir: Path, folder: str, max_frames=None):
    d = episode_dir / folder
    info_p = d / "video_info.json"
    vid_p = d / "video.mp4"
    if not info_p.exists() or not vid_p.exists():
        return {"present": False, "frames": None, "times_ns": None, "info": None, "folder": folder}

    try:
        info = load_video_info(info_p)
        frames = read_video_frames_rgb(vid_p, max_frames=max_frames)
        times_ns = frame_times_from_info(info, len(frames))
        return {"present": True, "frames": frames, "times_ns": times_ns, "info": info, "folder": folder}
    except Exception:
        return {"present": False, "frames": None, "times_ns": None, "info": None, "folder": folder}


# -------------------------
# Pose / wrench extraction
# -------------------------
def pose_to_vec7(pose_dict):
    p = pose_dict["position"]
    q = pose_dict["orientation"]
    return np.array([p["x"], p["y"], p["z"], q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)

def wrench_to_vec6(wmsg) -> np.ndarray:
    w = wmsg["wrench"]
    return np.array(
        [w["force"]["x"], w["force"]["y"], w["force"]["z"],
         w["torque"]["x"], w["torque"]["y"], w["torque"]["z"]],
        dtype=np.float32,
    )

# -------------------------
# Quaternion -> Euler (roll, pitch, yaw)
# Using standard aerospace (XYZ intrinsic == roll-pitch-yaw)
# Returns radians.
# -------------------------
def quat_xyzw_to_euler_rpy(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    # guard
    if not (np.isfinite(qx) and np.isfinite(qy) and np.isfinite(qz) and np.isfinite(qw)):
        return (np.nan, np.nan, np.nan)

    # normalize to reduce drift
    n = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
    if n <= 1e-12 or not np.isfinite(n):
        return (np.nan, np.nan, np.nan)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw*qx + qy*qz)
    cosr_cosp = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw*qy - qz*qx)
    if abs(sinp) >= 1.0:
        pitch = float(np.sign(sinp) * (np.pi / 2.0))  # clamp
    else:
        pitch = float(np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))

    return (roll, pitch, yaw)

def quat_batch_to_rpy(q_xyzw: np.ndarray) -> np.ndarray:
    """
    q_xyzw: (T,4) in [x,y,z,w]
    return: (T,3) in [roll,pitch,yaw] radians
    """
    if q_xyzw.ndim != 2 or q_xyzw.shape[1] != 4:
        raise ValueError("quat_batch_to_rpy expects shape (T,4)")

    T = q_xyzw.shape[0]
    out = np.full((T, 3), np.nan, dtype=np.float32)
    for i in range(T):
        x, y, z, w = map(float, q_xyzw[i])
        r, p, yv = quat_xyzw_to_euler_rpy(x, y, z, w)
        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = yv
    return out


# -------------------------
# Mosaic utils
# -------------------------
def resize_keep_aspect_rgb(img_rgb: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if w <= 0:
        raise ValueError("Invalid image width")
    if w == target_w:
        return img_rgb
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    new_h = max(new_h, 1)
    return cv2.resize(img_rgb, (target_w, new_h), interpolation=cv2.INTER_AREA)

def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    if h > target_h:
        y0 = (h - target_h) // 2
        return img[y0:y0 + target_h, :, :]
    pad_top = (target_h - h) // 2
    pad_bot = target_h - h - pad_top
    return cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    if w > target_w:
        x0 = (w - target_w) // 2
        return img[:, x0:x0 + target_w, :]
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def stack_grid(images: List[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to stack")
    cols = max(int(cols), 1)
    rows = int(np.ceil(len(images) / cols))

    grid_rows = []
    for r in range(rows):
        row_imgs = images[r * cols:(r + 1) * cols]
        if len(row_imgs) < cols:
            h0, w0 = row_imgs[0].shape[:2]
            for _ in range(cols - len(row_imgs)):
                row_imgs.append(np.zeros((h0, w0, 3), dtype=np.uint8))

        h_row = max(im.shape[0] for im in row_imgs)
        row_imgs = [pad_to_height(im, h_row) for im in row_imgs]
        row = np.hstack(row_imgs)
        grid_rows.append(row)

    w_grid = max(rimg.shape[1] for rimg in grid_rows)
    grid_rows = [pad_to_width(rimg, w_grid) for rimg in grid_rows]
    return np.vstack(grid_rows)

def rgb_to_tk(img_rgb: np.ndarray) -> tk.PhotoImage:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode PNG for Tk display")
    return tk.PhotoImage(data=buf.tobytes())

def draw_stream_label(img_rgb: np.ndarray, text: str) -> np.ndarray:
    out = img_rgb
    x, y = 10, 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


# -------------------------
# Episode cache
# -------------------------
@dataclass
class EpisodeCache:
    name: str
    T: int
    frame_times_ns: np.ndarray
    streams: Dict[str, dict]
    ft6: Optional[np.ndarray]
    tcp7: np.ndarray
    probe7: np.ndarray
    needle7: Optional[np.ndarray]
    # NEW: euler angles (radians)
    tcp_rpy: np.ndarray        # (T,3)
    probe_rpy: np.ndarray      # (T,3)
    needle_rpy: Optional[np.ndarray]  # (T,3) or None


# -------------------------
# Viewer
# -------------------------
class RawDecodeViewer:
    def __init__(
        self,
        episode_dirs: List[Path],
        task: str,
        out_fps: float,
        max_frames: Optional[int],
        mosaic_width: int,
        folder_room: str,
        folder_us: str,
        folder_wrgb: str,
        folder_wdepth: str,
        folder_us_sync: str,
        topic_tcp: str,
        topic_probe: str,
        topic_needle: str,
        topic_wrench: str,
    ):
        self.episode_dirs = episode_dirs
        self.task = task.lower().strip()
        if self.task not in ["task1", "task4"]:
            raise ValueError("--task must be task1 or task4")

        self.out_fps = float(out_fps)
        self.max_frames = max_frames
        self.mosaic_width = int(mosaic_width)

        self.folder_room = folder_room
        self.folder_us = folder_us
        self.folder_wrgb = folder_wrgb
        self.folder_wdepth = folder_wdepth
        self.folder_us_sync = folder_us_sync

        self.topic_tcp = topic_tcp
        self.topic_probe = topic_probe
        self.topic_needle = topic_needle
        self.topic_wrench = topic_wrench

        self.cache: Dict[int, EpisodeCache] = {}
        self.cur_ep = 0
        self.cur_i = 0
        self.playing = False
        self.play_fps = 30.0

        # plot defs
        # NOTE: orientation_x -> euler roll (radians)
        if self.task == "task1":
            self.plot_defs = [
                ("ati_ft_wrench", "ft", "FT", "multi6"),
                ("tcp_pose (x)", "tcp_x", "tcp_x", "single"),
                ("tcp_pose (roll)", "tcp_roll", "tcp_roll(rad)", "single"),
                ("ndi_us_probe (x)", "probe_x", "probe_x", "single"),
                ("ndi_us_probe (roll)", "probe_roll", "probe_roll(rad)", "single"),
            ]
        else:
            self.plot_defs = [
                ("tcp_pose (x)", "tcp_x", "tcp_x", "single"),
                ("tcp_pose (roll)", "tcp_roll", "tcp_roll(rad)", "single"),
                ("ndi_us_probe (x)", "probe_x", "probe_x", "single"),
                ("ndi_us_probe (roll)", "probe_roll", "probe_roll(rad)", "single"),
                ("ndi_needle (x)", "needle_x", "needle_x", "single"),
                ("ndi_needle (roll)", "needle_roll", "needle_roll(rad)", "single"),
            ]

        # ----- Tk UI -----
        self.win = tk.Tk()
        self.win.title(f"Raw Decode Viewer - {self.task}")

        ctrl = ttk.Frame(self.win, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Episodes:").pack(side=tk.LEFT)
        ttk.Label(ctrl, text=str(len(self.episode_dirs))).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(ctrl, text="Episode:").pack(side=tk.LEFT)
        self.ep_var = tk.StringVar(value="0")
        self.ep_box = ttk.Combobox(
            ctrl,
            textvariable=self.ep_var,
            values=[str(i) for i in range(len(self.episode_dirs))],
            width=8,
            state="readonly",
        )
        self.ep_box.pack(side=tk.LEFT, padx=(6, 12))
        self.ep_box.bind("<<ComboboxSelected>>", self.on_episode_change)

        ttk.Label(ctrl, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(ctrl, textvariable=self.fps_var, width=6).pack(side=tk.LEFT, padx=(6, 12))

        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)

        self.info_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.info_var).pack(side=tk.RIGHT)

        # info bar
        infobar = ttk.Frame(self.win, padding=(8, 0, 8, 8))
        infobar.pack(side=tk.TOP, fill=tk.X)

        self.bar1 = tk.StringVar(value="")
        self.bar2 = tk.StringVar(value="")
        self.bar3 = tk.StringVar(value="")

        ttk.Label(infobar, textvariable=self.bar1).pack(side=tk.TOP, anchor="w")
        ttk.Label(infobar, textvariable=self.bar2).pack(side=tk.TOP, anchor="w")
        ttk.Label(infobar, textvariable=self.bar3).pack(side=tk.TOP, anchor="w")

        # ---- draggable splitter between video and plots ----
        self.paned = ttk.Panedwindow(self.win, orient=tk.HORIZONTAL)
        self.paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.left = ttk.Frame(self.paned)
        self.right = ttk.Frame(self.paned)
        self.paned.add(self.left, weight=3)
        self.paned.add(self.right, weight=1)

        # self.img_label = ttk.Label(self.left)
        # self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # ---- Scrollable video area (Canvas + vertical scrollbar) ----
        self.video_frame = ttk.Frame(self.left)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.vscroll = ttk.Scrollbar(self.video_frame, orient=tk.VERTICAL)
        self.vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas_vid = tk.Canvas(self.video_frame, yscrollcommand=self.vscroll.set, highlightthickness=0)
        self.canvas_vid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.vscroll.config(command=self.canvas_vid.yview)

        # store image id on canvas
        self._canvas_img_id = None
        self._tk_img = None

        def _on_canvas_configure(event):
            # keep scrollregion updated when canvas size changes
            self.canvas_vid.configure(scrollregion=self.canvas_vid.bbox("all"))

        self.canvas_vid.bind("<Configure>", _on_canvas_configure)

        # mouse wheel scroll (Windows/Linux)
        self.canvas_vid.bind_all("<MouseWheel>", lambda e: self.canvas_vid.yview_scroll(int(-1*(e.delta/120)), "units"))
        # macOS wheel (may differ)
        self.canvas_vid.bind_all("<Button-4>", lambda e: self.canvas_vid.yview_scroll(-3, "units"))
        self.canvas_vid.bind_all("<Button-5>", lambda e: self.canvas_vid.yview_scroll(3, "units"))


        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            self.left, from_=0.0, to=1.0,
            orient=tk.HORIZONTAL, variable=self.slider_var,
            command=self.on_slider
        )
        self.slider.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        # Matplotlib
        n_plots = len(self.plot_defs)
        fig_h = max(3.0, 1.25 * n_plots)
        self.fig = plt.Figure(figsize=(7.6, fig_h), dpi=100)

        self.axes = []
        for i in range(n_plots):
            if i == 0:
                ax = self.fig.add_subplot(n_plots, 1, i + 1)
            else:
                ax = self.fig.add_subplot(n_plots, 1, i + 1, sharex=self.axes[0])
            self.axes.append(ax)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # plot markers/labels
        self.vlines = []
        self.markers = []
        self.value_texts = []

        # key bindings
        self.win.bind("<space>", lambda e: self.toggle_play())
        self.win.bind("<Left>", lambda e: self.step_frame(-1))
        self.win.bind("<Right>", lambda e: self.step_frame(+1))
        self.win.bind("<Shift-Left>", lambda e: self.step_frame(-10))
        self.win.bind("<Shift-Right>", lambda e: self.step_frame(+10))

        # init
        self.load_episode(0)
        self.update_all()
        self.win.after(50, self.play_loop)

    def run(self):
        self.win.mainloop()

    def step_frame(self, delta: int):
        c = self.cache[self.cur_ep]
        self.cur_i = int(np.clip(self.cur_i + int(delta), 0, c.T - 1))
        self.slider_var.set(float(self.cur_i))
        self.update_all()

    # -------------------------
    # Data loading / alignment
    # -------------------------
    def _load_episode_cache(self, ep_dir: Path) -> EpisodeCache:
        room = load_video_stream_optional(ep_dir, self.folder_room, max_frames=self.max_frames)
        us = load_video_stream_optional(ep_dir, self.folder_us, max_frames=self.max_frames)
        wrgb = load_video_stream_optional(ep_dir, self.folder_wrgb, max_frames=self.max_frames)
        wdepth = load_video_stream_optional(ep_dir, self.folder_wdepth, max_frames=self.max_frames)

        if not us["present"]:
            raise RuntimeError(f"{ep_dir.name}: missing ultrasound: {self.folder_us}")
        if not wrgb["present"]:
            raise RuntimeError(f"{ep_dir.name}: missing wrist rgb: {self.folder_wrgb}")
        if not wdepth["present"]:
            raise RuntimeError(f"{ep_dir.name}: missing wrist depth: {self.folder_wdepth}")

        us_sync = None
        if self.task == "task4":
            us_sync = load_video_stream_optional(ep_dir, self.folder_us_sync, max_frames=self.max_frames)
            if not us_sync["present"]:
                raise RuntimeError(f"{ep_dir.name}: task4 requires us_sync but missing: {self.folder_us_sync}")

        present_times = [us["times_ns"], wrgb["times_ns"], wdepth["times_ns"]]
        if room["present"]:
            present_times.append(room["times_ns"])
        if self.task == "task4":
            present_times.append(us_sync["times_ns"])

        t0 = max(ts[0] for ts in present_times)
        t1 = min(ts[-1] for ts in present_times)
        if t1 <= t0:
            raise RuntimeError(f"{ep_dir.name}: no temporal overlap across PRESENT streams.")

        out_dt_ns = int(round(1e9 / self.out_fps))
        T = int((t1 - t0) // out_dt_ns) + 1
        if T < 2:
            raise RuntimeError(f"{ep_dir.name}: too short after time alignment (T={T})")

        frame_times_ns = (t0 + np.arange(T, dtype=np.int64) * out_dt_ns).astype(np.int64)

        t_tcp, tcp_payloads = load_ndjson_times_and_payloads(ep_dir / self.topic_tcp / "messages.ndjson")
        t_probe, probe_payloads = load_ndjson_times_and_payloads(ep_dir / self.topic_probe / "messages.ndjson")

        t_needle, needle_payloads = (None, None)
        if self.task == "task4":
            t_needle, needle_payloads = load_ndjson_times_and_payloads(ep_dir / self.topic_needle / "messages.ndjson")

        t_wrench, wrench_payloads = load_ndjson_optional(ep_dir / self.topic_wrench / "messages.ndjson")

        tcp7 = np.zeros((T, 7), dtype=np.float32)
        probe7 = np.zeros((T, 7), dtype=np.float32)
        needle7 = None if self.task != "task4" else np.zeros((T, 7), dtype=np.float32)

        ft6 = None
        if self.task == "task1":
            ft6 = np.full((T, 6), np.nan, dtype=np.float32)
        else:
            if t_wrench is not None and wrench_payloads is not None:
                ft6 = np.full((T, 6), np.nan, dtype=np.float32)

        for i in range(T):
            t_ns = int(frame_times_ns[i])

            tcp_msg = nearest_payload(t_tcp, tcp_payloads, t_ns)
            probe_msg = nearest_payload(t_probe, probe_payloads, t_ns)
            tcp7[i] = pose_to_vec7(tcp_msg["pose"])
            probe7[i] = pose_to_vec7(probe_msg["pose"])

            if self.task == "task4":
                needle_msg = nearest_payload(t_needle, needle_payloads, t_ns)
                needle7[i] = pose_to_vec7(needle_msg["pose"])

            if ft6 is not None and (t_wrench is not None and wrench_payloads is not None):
                wmsg = nearest_payload(t_wrench, wrench_payloads, t_ns)
                ft6[i] = wrench_to_vec6(wmsg)

        # NEW: precompute euler (rpy) from quaternion part
        tcp_rpy = quat_batch_to_rpy(tcp7[:, 3:7])
        probe_rpy = quat_batch_to_rpy(probe7[:, 3:7])
        needle_rpy = None
        if needle7 is not None:
            needle_rpy = quat_batch_to_rpy(needle7[:, 3:7])

        streams = {
            "room": room,
            "ultrasound": us,
            "wrist_rgb": wrgb,
            "wrist_depth": wdepth,
        }
        if self.task == "task4":
            streams["us_sync"] = us_sync

        return EpisodeCache(
            name=ep_dir.name,
            T=T,
            frame_times_ns=frame_times_ns,
            streams=streams,
            ft6=ft6,
            tcp7=tcp7,
            probe7=probe7,
            needle7=needle7,
            tcp_rpy=tcp_rpy,
            probe_rpy=probe_rpy,
            needle_rpy=needle_rpy,
        )

    # -------------------------
    # UI callbacks
    # -------------------------
    def load_episode(self, ep_index: int):
        ep_index = int(ep_index)
        self.cur_ep = ep_index
        self.cur_i = 0

        if ep_index not in self.cache:
            ep_dir = self.episode_dirs[ep_index]
            self.info_var.set(f"Loading {ep_dir.name} ...")
            self.win.update_idletasks()
            self.cache[ep_index] = self._load_episode_cache(ep_dir)

        c = self.cache[ep_index]
        self.slider.configure(from_=0.0, to=float(max(c.T - 1, 0)))
        self.slider_var.set(0.0)
        self._build_plots(c)

    def on_episode_change(self, _evt=None):
        ep_idx = int(self.ep_var.get())
        self.playing = False
        self.play_btn.configure(text="Play")
        self.load_episode(ep_idx)
        self.update_all()

    def on_slider(self, _val):
        self.cur_i = int(round(float(self.slider_var.get())))
        self.update_all()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        try:
            self.play_fps = float(self.fps_var.get())
            if not np.isfinite(self.play_fps) or self.play_fps <= 0:
                self.play_fps = 30.0
        except Exception:
            self.play_fps = 30.0

    def play_loop(self):
        if self.playing:
            c = self.cache[self.cur_ep]
            nxt = self.cur_i + 1
            if nxt >= c.T:
                nxt = 0
            self.cur_i = nxt
            self.slider_var.set(float(self.cur_i))
            self.update_all()
            delay = int(round(1000.0 / max(self.play_fps, 1e-6)))
            self.win.after(max(delay, 1), self.play_loop)
        else:
            self.win.after(50, self.play_loop)

    # -------------------------
    # Plots
    # -------------------------
    def _build_plots(self, c: EpisodeCache):
        for ax in self.axes:
            ax.clear()

        t = np.arange(c.T)

        tcp_x = c.tcp7[:, 0]
        probe_x = c.probe7[:, 0]
        needle_x = c.needle7[:, 0] if c.needle7 is not None else None

        # euler roll
        tcp_roll = c.tcp_rpy[:, 0]
        probe_roll = c.probe_rpy[:, 0]
        needle_roll = c.needle_rpy[:, 0] if c.needle_rpy is not None else None

        series = {
            "tcp_x": tcp_x,
            "tcp_roll": tcp_roll,
            "probe_x": probe_x,
            "probe_roll": probe_roll,
            "needle_x": needle_x,
            "needle_roll": needle_roll,
        }

        self.vlines = []
        self.markers = []
        self.value_texts = []

        for ax, (title, key, ylabel, style) in zip(self.axes, self.plot_defs):
            if style == "multi6":
                ft = c.ft6
                if ft is None:
                    ft = np.full((c.T, 6), np.nan, dtype=np.float32)
                ax.plot(t, ft[:, 0], label="Fx")
                ax.plot(t, ft[:, 1], label="Fy")
                ax.plot(t, ft[:, 2], label="Fz")
                ax.plot(t, ft[:, 3], label="Tx")
                ax.plot(t, ft[:, 4], label="Ty")
                ax.plot(t, ft[:, 5], label="Tz")
                ax.legend(loc="upper right", fontsize=7)

                y0 = ft[:, 0]
                mk, = ax.plot([self.cur_i], [y0[self.cur_i]], marker="o", markersize=5, linestyle="None")
                txt = ax.text(
                    0.99, 0.95, "", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none")
                )
            else:
                y = series.get(key, None)
                if y is None:
                    y = np.full((c.T,), np.nan, dtype=np.float32)
                ax.plot(t, y, label=title)
                ax.legend(loc="upper right", fontsize=7)

                mk, = ax.plot([self.cur_i], [y[self.cur_i]], marker="o", markersize=5, linestyle="None")
                txt = ax.text(
                    0.99, 0.95, "", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none")
                )

            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9, labelpad=12)

            v = ax.axvline(self.cur_i)
            self.vlines.append(v)
            self.markers.append(mk)
            self.value_texts.append(txt)

        self.axes[-1].set_xlabel("frame")

        self.fig.subplots_adjust(left=0.33, right=0.98, top=0.97, bottom=0.08, hspace=0.65)
        self._update_plot_markers_and_text()
        self.canvas.draw_idle()

    def _update_vlines(self):
        for v in self.vlines:
            v.set_xdata([self.cur_i, self.cur_i])

    def _update_plot_markers_and_text(self):
        c = self.cache[self.cur_ep]
        i = int(np.clip(self.cur_i, 0, c.T - 1))

        tcp_x = c.tcp7[:, 0]
        probe_x = c.probe7[:, 0]
        needle_x = c.needle7[:, 0] if c.needle7 is not None else None

        tcp_roll = c.tcp_rpy[:, 0]
        probe_roll = c.probe_rpy[:, 0]
        needle_roll = c.needle_rpy[:, 0] if c.needle_rpy is not None else None

        series = {
            "tcp_x": tcp_x,
            "tcp_roll": tcp_roll,
            "probe_x": probe_x,
            "probe_roll": probe_roll,
            "needle_x": needle_x,
            "needle_roll": needle_roll,
        }

        for idx, (_ax, (title, key, _ylabel, style)) in enumerate(zip(self.axes, self.plot_defs)):
            mk = self.markers[idx]
            txt = self.value_texts[idx]

            if style == "multi6":
                ft = c.ft6
                if ft is None:
                    ft = np.full((c.T, 6), np.nan, dtype=np.float32)
                mk.set_data([i], [float(ft[i, 0]) if np.isfinite(ft[i, 0]) else np.nan])
                txt.set_text(
                    f"Fx={ft[i,0]:.2f} Fy={ft[i,1]:.2f} Fz={ft[i,2]:.2f}\n"
                    f"Tx={ft[i,3]:.2f} Ty={ft[i,4]:.2f} Tz={ft[i,5]:.2f}"
                )
            else:
                y = series.get(key, None)
                if y is None:
                    val = np.nan
                else:
                    val = float(y[i]) if np.isfinite(y[i]) else np.nan
                mk.set_data([i], [val])
                txt.set_text(f"value={val:.5f}" if np.isfinite(val) else "value=NaN")

    # -------------------------
    # Image display
    # -------------------------
    def _get_frame_rgb(self, stream: dict, t_ns: int, fallback_shape: Tuple[int, int, int]) -> np.ndarray:
        if stream is None or not stream.get("present", False):
            return np.zeros(fallback_shape, dtype=np.uint8)
        return pick_frame_by_time(stream["frames"], stream["times_ns"], t_ns)

    # def _update_image(self):
    #     c = self.cache[self.cur_ep]
    #     i = int(np.clip(self.cur_i, 0, c.T - 1))
    #     t_ns = int(c.frame_times_ns[i])

    #     img_us = self._get_frame_rgb(c.streams["ultrasound"], t_ns, (1080, 1920, 3))
    #     img_room = self._get_frame_rgb(c.streams["room"], t_ns, (768, 1024, 3))
    #     img_wrgb = self._get_frame_rgb(c.streams["wrist_rgb"], t_ns, (360, 640, 3))
    #     img_wdepth = self._get_frame_rgb(c.streams["wrist_depth"], t_ns, (360, 640, 3))

    #     room_present = bool(c.streams["room"]["present"])

    #     if self.task == "task1":
    #         names = ["ultrasound", "room", "wrist_rgb", "wrist_depth"]
    #         cols = 2
    #     else:
    #         img_sync = self._get_frame_rgb(c.streams["us_sync"], t_ns, img_us.shape)
    #         if not room_present:
    #             names = ["ultrasound", "us_sync", "wrist_rgb", "wrist_depth"]
    #             cols = 2
    #         else:
    #             names = ["ultrasound", "room", "wrist_rgb", "wrist_depth", "us_sync", "blank"]
    #             cols = 2

    #     name_to_img = {
    #         "ultrasound": img_us,
    #         "room": img_room,
    #         "wrist_rgb": img_wrgb,
    #         "wrist_depth": img_wdepth,
    #     }
    #     if self.task == "task4":
    #         name_to_img["us_sync"] = img_sync
    #         name_to_img["blank"] = np.zeros_like(img_sync)

    #     tile_w = max(int(self.mosaic_width // cols), 360)

    #     tiles = []
    #     for nm in names:
    #         im = name_to_img[nm]
    #         im2 = resize_keep_aspect_rgb(im, tile_w)
    #         if nm != "blank":
    #             im2 = draw_stream_label(im2, nm)
    #         tiles.append(im2)

    #     mosaic = stack_grid(tiles, cols=cols)

    #     self._tk_img = rgb_to_tk(mosaic)
    #     self.img_label.configure(image=self._tk_img)
    def _make_current_mosaic_rgb(self) -> np.ndarray:
        c = self.cache[self.cur_ep]
        i = int(np.clip(self.cur_i, 0, c.T - 1))
        t_ns = int(c.frame_times_ns[i])

        img_us = self._get_frame_rgb(c.streams["ultrasound"], t_ns, (1080, 1920, 3))
        img_room = self._get_frame_rgb(c.streams["room"], t_ns, (768, 1024, 3))
        img_wrgb = self._get_frame_rgb(c.streams["wrist_rgb"], t_ns, (360, 640, 3))
        img_wdepth = self._get_frame_rgb(c.streams["wrist_depth"], t_ns, (360, 640, 3))

        room_present = bool(c.streams["room"]["present"])

        if self.task == "task1":
            names = ["ultrasound", "room", "wrist_rgb", "wrist_depth"]
            cols = 2
        else:
            img_sync = self._get_frame_rgb(c.streams["us_sync"], t_ns, img_us.shape)
            if not room_present:
                names = ["ultrasound", "us_sync", "wrist_rgb", "wrist_depth"]
                cols = 2
            else:
                names = ["ultrasound", "room", "wrist_rgb", "wrist_depth", "us_sync", "blank"]
                cols = 2

        name_to_img = {
            "ultrasound": img_us,
            "room": img_room,
            "wrist_rgb": img_wrgb,
            "wrist_depth": img_wdepth,
        }
        if self.task == "task4":
            name_to_img["us_sync"] = img_sync
            name_to_img["blank"] = np.zeros_like(img_sync)

        tile_w = max(int(self.mosaic_width // cols), 360)

        tiles = []
        for nm in names:
            im = name_to_img[nm]
            im2 = resize_keep_aspect_rgb(im, tile_w)
            if nm != "blank":
                im2 = draw_stream_label(im2, nm)
            tiles.append(im2)

        mosaic = stack_grid(tiles, cols=cols)
        return mosaic

    def _update_image(self):
        mosaic = self._make_current_mosaic_rgb()

        self._tk_img = rgb_to_tk(mosaic)

        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas_vid.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            self.canvas_vid.itemconfig(self._canvas_img_id, image=self._tk_img)

        # update scroll region to allow scrolling
        self.canvas_vid.config(scrollregion=(0, 0, mosaic.shape[1], mosaic.shape[0]))


    def update_all(self):
        c = self.cache[self.cur_ep]
        i = int(np.clip(self.cur_i, 0, c.T - 1))
        t_ns = int(c.frame_times_ns[i])

        room_present = bool(c.streams["room"]["present"])
        wrench_present = (c.ft6 is not None) and np.isfinite(c.ft6).any()

        self.info_var.set(
            f"{c.name} | frame={i}/{c.T-1} | room_present={room_present} | wrench_present={wrench_present}"
        )

        tcp_x = float(c.tcp7[i, 0])
        probe_x = float(c.probe7[i, 0])

        tcp_roll = float(c.tcp_rpy[i, 0])
        probe_roll = float(c.probe_rpy[i, 0])
        # 你想显示全 rpy 的话，可以用：
        # tcp_r, tcp_p, tcp_y = map(float, c.tcp_rpy[i])
        # probe_r, probe_p, probe_y = map(float, c.probe_rpy[i])

        self.bar1.set(f"{c.name} | ep={self.cur_ep}/{len(self.episode_dirs)-1} | frame={i}/{c.T-1} | t_ns={t_ns}")
        if self.task == "task4":
            needle_x = float(c.needle7[i, 0])
            needle_roll = float(c.needle_rpy[i, 0]) if c.needle_rpy is not None else float("nan")
            self.bar2.set(
                f"tcp_x={tcp_x:.3f} tcp_roll={tcp_roll:.3f} | probe_x={probe_x:.3f} probe_roll={probe_roll:.3f} | "
                f"needle_x={needle_x:.3f} needle_roll={needle_roll:.3f}"
            )
        else:
            self.bar2.set(
                f"tcp_x={tcp_x:.3f} tcp_roll={tcp_roll:.3f} | probe_x={probe_x:.3f} probe_roll={probe_roll:.3f}"
            )

        if c.ft6 is not None and np.isfinite(c.ft6[i]).any():
            ft = c.ft6[i]
            self.bar3.set("FT: " + ", ".join([f"{x:.2f}" for x in ft.tolist()]))
        else:
            self.bar3.set("")

        self._update_image()
        self._update_vlines()
        self._update_plot_markers_and_text()
        self.canvas.draw_idle()


# -------------------------
# CLI
# -------------------------
def resolve_episode_dirs(raw_root: Path, max_episodes: int) -> List[Path]:
    if raw_root.is_dir() and raw_root.name.startswith("rosbag2_"):
        eps = [raw_root]
    else:
        eps = sorted([p for p in raw_root.glob("rosbag2_*") if p.is_dir()])
    if max_episodes > 0:
        eps = eps[:max_episodes]
    if not eps:
        raise RuntimeError(f"No episode folders found under: {raw_root}")
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, help="A rosbag2_* dir, or a dir containing rosbag2_* dirs")
    ap.add_argument("--task", required=True, choices=["task1", "task4"], help="task1 or task4(=task4.1/4.2)")

    ap.add_argument("--out_fps", type=float, default=30.0, help="Master timeline fps")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit decoded frames per video (debug)")
    ap.add_argument("--max_episodes", type=int, default=-1)
    ap.add_argument("--mosaic_width", type=int, default=1400)

    ap.add_argument("--folder_room", type=str, default="vega_vt__image_raw__compressed")
    ap.add_argument("--folder_us", type=str, default="image_raw__compressed")
    ap.add_argument("--folder_wrgb", type=str, default="zed__zed_node__rgb__color__rect__image__compressed")
    ap.add_argument("--folder_wdepth", type=str, default="zed__zed_node__depth__depth_registered__compressedDepth")
    ap.add_argument("--folder_us_sync", type=str, default="visualize__us_imaging_sync__compressed")

    ap.add_argument("--topic_tcp", type=str, default="tcp_pose_broadcaster__pose")
    ap.add_argument("--topic_probe", type=str, default="ndi__us_probe_pose")
    ap.add_argument("--topic_needle", type=str, default="ndi__needle_pose")
    ap.add_argument("--topic_wrench", type=str, default="ati_ft_broadcaster__wrench")

    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    eps = resolve_episode_dirs(raw_root, args.max_episodes)

    try:
        app = RawDecodeViewer(
            episode_dirs=eps,
            task=args.task,
            out_fps=args.out_fps,
            max_frames=args.max_frames,
            mosaic_width=args.mosaic_width,
            folder_room=args.folder_room,
            folder_us=args.folder_us,
            folder_wrgb=args.folder_wrgb,
            folder_wdepth=args.folder_wdepth,
            folder_us_sync=args.folder_us_sync,
            topic_tcp=args.topic_tcp,
            topic_probe=args.topic_probe,
            topic_needle=args.topic_needle,
            topic_wrench=args.topic_wrench,
        )
        app.run()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise


if __name__ == "__main__":
    main()
