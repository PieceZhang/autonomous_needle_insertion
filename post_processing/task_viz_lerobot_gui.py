#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# -------------------------
# Utilities: resize + pad to make mosaic robust
# -------------------------
def resize_keep_aspect(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= 0:
        raise ValueError("Invalid image width")
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    new_h = max(new_h, 1)
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)

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

def stack_2x2(img00, img01, img10, img11) -> np.ndarray:
    h_top = max(img00.shape[0], img01.shape[0])
    img00_ = pad_to_height(img00, h_top)
    img01_ = pad_to_height(img01, h_top)
    top = np.hstack([img00_, img01_])

    h_bot = max(img10.shape[0], img11.shape[0])
    img10_ = pad_to_height(img10, h_bot)
    img11_ = pad_to_height(img11, h_bot)
    bot = np.hstack([img10_, img11_])

    w = max(top.shape[1], bot.shape[1])
    top = pad_to_width(top, w)
    bot = pad_to_width(bot, w)
    return np.vstack([top, bot])

def bgr_to_tk(img_bgr: np.ndarray) -> tk.PhotoImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", rgb)
    if not ok:
        raise RuntimeError("Failed to encode PNG for Tk display")
    return tk.PhotoImage(data=buf.tobytes())

def swap_rb(img: np.ndarray) -> np.ndarray:
    # swap channel 0 and 2
    return img[:, :, ::-1].copy()


# -------------------------
# Backends: ImageBank (preferred) and VideoBank (fallback)
# -------------------------
class ImageBank:
    def __init__(self, root: Path, feature_key: str):
        self.root = root
        self.feature_key = feature_key
        base = self.root / "images" / self.feature_key
        if not base.exists():
            raise FileNotFoundError(f"Image folder not found: {base}")

        exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        files = []
        for e in exts:
            files.extend(base.glob(f"chunk-*/**/{e}"))
            files.extend(base.glob(f"chunk-*/{e}"))
            files.extend(base.glob(f"**/{e}"))
        self.files = sorted(set(files))
        if not self.files:
            raise FileNotFoundError(f"No image files found under: {base}")

    def read_bgr(self, global_index: int) -> np.ndarray:
        if global_index < 0 or global_index >= len(self.files):
            raise IndexError(f"[ImageBank] global_index={global_index} out of range (N={len(self.files)})")
        p = self.files[global_index]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # OpenCV returns BGR
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return img


class VideoBank:
    def __init__(self, root: Path, feature_key: str):
        self.root = root
        self.feature_key = feature_key
        base = self.root / "videos" / self.feature_key
        if not base.exists():
            raise FileNotFoundError(f"Video folder not found: {base}")

        self.mp4s = sorted(base.glob("chunk-*/*.mp4"), key=lambda p: (p.parent.name, p.name))
        if not self.mp4s:
            raise FileNotFoundError(f"No mp4 found under: {base}")

        self.paths: List[Path] = []
        self.starts: List[int] = []
        self.ends: List[int] = []
        self.caps: List[cv2.VideoCapture] = []
        start = 0

        for p in self.mp4s:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {p}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n <= 0:
                n = self._count_frames_slow(cap)
                cap.release()
                cap = cv2.VideoCapture(str(p))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to reopen video: {p}")

            self.paths.append(p)
            self.starts.append(start)
            self.ends.append(start + n)
            self.caps.append(cap)
            start += n

    @staticmethod
    def _count_frames_slow(cap: cv2.VideoCapture) -> int:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        return n

    def read_bgr(self, global_index: int) -> np.ndarray:
        if global_index < 0:
            raise IndexError("[VideoBank] global_index < 0")

        seg = None
        for i in range(len(self.starts)):
            if self.starts[i] <= global_index < self.ends[i]:
                seg = i
                break
        if seg is None:
            raise IndexError(f"[VideoBank] global_index={global_index} out of all segments (max={self.ends[-1]-1})")

        local = global_index - self.starts[seg]
        cap = self.caps[seg]

        cap.set(cv2.CAP_PROP_POS_FRAMES, float(local))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {local} from {self.paths[seg]}")
        return frame_bgr

# class VideoBank:
#     def __init__(self, root: Path, feature_key: str):
#         self.root = root
#         self.feature_key = feature_key
#         base = self.root / "videos" / self.feature_key
#         if not base.exists():
#             raise FileNotFoundError(f"Video folder not found: {base}")

#         self.mp4s = sorted(base.glob("chunk-*/*.mp4"), key=lambda p: (p.parent.name, p.name))
#         if not self.mp4s:
#             raise FileNotFoundError(f"No mp4 found under: {base}")

#         self.paths: List[Path] = []
#         self.starts: List[int] = []
#         self.ends: List[int] = []
#         self.frame_counts: List[int] = []

#         start = 0
#         for p in self.mp4s:
#             cap = cv2.VideoCapture(str(p))
#             if not cap.isOpened():
#                 raise RuntimeError(f"Failed to open video: {p}")
#             n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             if n <= 0:
#                 # fallback: assume unknown; we will handle by reading until fail
#                 n = 10**9

#             self.paths.append(p)
#             self.starts.append(start)
#             self.ends.append(start + n)
#             self.frame_counts.append(n)
#             start += n if n < 10**9 else 0

#         # streaming state
#         self._cur_seg = None
#         self._cap = None
#         self._cur_pos = -1  # last returned local frame index within current seg

#     def _open_seg(self, seg: int):
#         if self._cap is not None:
#             self._cap.release()
#         self._cap = cv2.VideoCapture(str(self.paths[seg]))
#         if not self._cap.isOpened():
#             raise RuntimeError(f"Failed to open video: {self.paths[seg]}")
#         self._cur_seg = seg
#         self._cur_pos = -1

#     def _seek_by_skip(self, target_local: int):
#         # reopen and sequentially skip to target_local (robust for AV1 etc.)
#         assert self._cap is not None
#         for _ in range(target_local):
#             ok = self._cap.grab()
#             if not ok:
#                 break
#         self._cur_pos = target_local - 1

#     def read_bgr(self, global_index: int) -> np.ndarray:
#         if global_index < 0:
#             raise IndexError("[VideoBank] global_index < 0")

#         seg = None
#         for i in range(len(self.starts)):
#             if self.starts[i] <= global_index < self.ends[i]:
#                 seg = i
#                 break
#         if seg is None:
#             # if frame_count was unknown, ends might be nonsense; fall back to first seg
#             seg = 0

#         local = global_index - self.starts[seg]

#         # open correct segment
#         if self._cur_seg != seg or self._cap is None:
#             self._open_seg(seg)

#         # fast path: play mode sequential read
#         if local == self._cur_pos + 1:
#             ok, frame = self._cap.read()
#             if not ok or frame is None:
#                 # maybe hit end; reopen and try again
#                 self._open_seg(seg)
#                 self._seek_by_skip(local)
#                 ok, frame = self._cap.read()
#             if not ok or frame is None:
#                 raise RuntimeError(f"Failed to read frame local={local} from {self.paths[seg]}")
#             self._cur_pos = local
#             return frame

#         # slow path: jump (slider / episode change)
#         self._open_seg(seg)
#         if local > 0:
#             self._seek_by_skip(local)
#         ok, frame = self._cap.read()
#         if not ok or frame is None:
#             raise RuntimeError(f"Failed to read frame local={local} from {self.paths[seg]}")
#         self._cur_pos = local
#         return frame



class FrameBank:
    def __init__(self, root: Path, feature_key: str):
        self.root = root
        self.feature_key = feature_key
        try:
            self.backend = ImageBank(root, feature_key)
            self.kind = "images"
        except Exception:
            self.backend = VideoBank(root, feature_key)
            self.kind = "videos"

    def read_bgr(self, global_index: int) -> np.ndarray:
        return self.backend.read_bgr(global_index)


# -------------------------
# Episode cache
# -------------------------
@dataclass
class EpisodeCache:
    row_indices: List[int]
    global_indices: np.ndarray
    T: int
    tcp_xyz: np.ndarray
    probe_xyz: np.ndarray
    needle_xyz: np.ndarray
    act_probe_xyz: np.ndarray
    act_needle_xyz: np.ndarray


# -------------------------
# Viewer
# -------------------------
class LeRobotViewer042:
    def __init__(self, dataset_root: str, mosaic_width: int = 1200, swap_rb_streams: Set[str] | None = None):
        self.root = Path(dataset_root)
        self.ds = LeRobotDataset(str(self.root))  # lerobot 0.4.2
        self.hfd = self.ds.hf_dataset
        if self.hfd is None:
            raise RuntimeError("ds.hf_dataset is None")

        self.n_eps = int(self.ds.num_episodes)
        if self.n_eps <= 0:
            raise RuntimeError("No episodes found (num_episodes <= 0)")

        cols = set(self.hfd.column_names)
        for need in ["episode_index", "frame_index", "index", "observation.state", "action"]:
            if need not in cols:
                raise RuntimeError(f"Missing column '{need}' in hf_dataset columns")

        # stream keys (match folders under images/ and videos/)
        self.k_us = "observation.images.ultrasound"
        self.k_room = "observation.images.room_rgb_camera"
        self.k_wrgb = "observation.images.wrist_camera_rgb"
        self.k_wdepth = "observation.images.wrist_camera_depth"

        self.us_bank = FrameBank(self.root, self.k_us)
        self.room_bank = FrameBank(self.root, self.k_room)
        self.wrgb_bank = FrameBank(self.root, self.k_wrgb)
        self.wdepth_bank = FrameBank(self.root, self.k_wdepth)

        print("[INFO] ultrasound backend:", self.us_bank.kind)
        print("[INFO] room backend:", self.room_bank.kind)
        print("[INFO] wrist_rgb backend:", self.wrgb_bank.kind)
        print("[INFO] wrist_depth backend:", self.wdepth_bank.kind)

        self.swap_rb_streams = set(swap_rb_streams or set())

        self.cache: Dict[int, EpisodeCache] = {}
        self.current_ep = 0
        self.current_frame = 0
        self.playing = False
        self.play_fps = 30.0
        self.mosaic_width = mosaic_width

        # ---- Tk UI ----
        self.win = tk.Tk()
        self.win.title("LeRobot Viewer (0.4.2) - stream RB swap fix")

        ctrl = ttk.Frame(self.win, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Dataset:").pack(side=tk.LEFT)
        ttk.Label(ctrl, text=str(self.root)).pack(side=tk.LEFT, padx=(6, 12))

        ttk.Label(ctrl, text="Episode:").pack(side=tk.LEFT)
        self.ep_var = tk.StringVar(value="0")
        self.ep_box = ttk.Combobox(
            ctrl, textvariable=self.ep_var,
            values=[str(i) for i in range(self.n_eps)],
            width=8, state="readonly"
        )
        self.ep_box.pack(side=tk.LEFT, padx=(6, 12))
        self.ep_box.bind("<<ComboboxSelected>>", self.on_episode_change)

        ttk.Label(ctrl, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(ctrl, textvariable=self.fps_var, width=6).pack(side=tk.LEFT, padx=(6, 12))

        self.play_btn = ttk.Button(ctrl, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)

        # RB swap toggles (room + wrgb)
        self.var_swap_room = tk.BooleanVar(value=("room" in self.swap_rb_streams))
        self.var_swap_wrgb = tk.BooleanVar(value=("wrgb" in self.swap_rb_streams))
        ttk.Checkbutton(ctrl, text="SwapRB room", variable=self.var_swap_room, command=self.on_swap_toggle).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(ctrl, text="SwapRB wrgb", variable=self.var_swap_wrgb, command=self.on_swap_toggle).pack(side=tk.LEFT, padx=(6, 0))

        self.info_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.info_var).pack(side=tk.RIGHT)

        main = ttk.Frame(self.win, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.img_label = ttk.Label(left)
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            left, from_=0.0, to=1.0,
            orient=tk.HORIZONTAL, variable=self.slider_var,
            command=self.on_slider
        )
        self.slider.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.fig = plt.Figure(figsize=(7.2, 7.2), dpi=100)
        self.ax1 = self.fig.add_subplot(4, 1, 1)
        self.ax2 = self.fig.add_subplot(4, 1, 2, sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(4, 1, 3, sharex=self.ax1)
        self.ax4 = self.fig.add_subplot(4, 1, 4, sharex=self.ax1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.vlines = []

        # init
        self.load_episode(0)
        self.update_all()
        self.win.after(50, self.play_loop)

    def run(self):
        self.win.mainloop()

    def on_swap_toggle(self):
        self.swap_rb_streams = set()
        if self.var_swap_room.get():
            self.swap_rb_streams.add("room")
        if self.var_swap_wrgb.get():
            self.swap_rb_streams.add("wrgb")
        self.update_all()

    def _episode_rows(self, ep_idx: int) -> List[int]:
        epi = self.hfd["episode_index"]
        rows = [i for i, v in enumerate(epi) if int(v) == int(ep_idx)]
        if not rows:
            return rows
        frame_ids = [int(self.hfd[i]["frame_index"]) for i in rows]
        order = np.argsort(frame_ids)
        return [rows[i] for i in order]

    def _extract_cache(self, ep_idx: int) -> EpisodeCache:
        rows = self._episode_rows(ep_idx)
        if not rows:
            raise RuntimeError(f"Episode {ep_idx} has 0 frames.")

        T = len(rows)
        tcp_xyz = np.zeros((T, 3), dtype=np.float32)
        probe_xyz = np.zeros((T, 3), dtype=np.float32)
        needle_xyz = np.zeros((T, 3), dtype=np.float32)
        act_probe_xyz = np.zeros((T, 3), dtype=np.float32)
        act_needle_xyz = np.zeros((T, 3), dtype=np.float32)
        global_idx = np.zeros((T,), dtype=np.int64)

        for j, ridx in enumerate(rows):
            fr = self.hfd[ridx]
            state = np.asarray(fr["observation.state"], dtype=np.float32)
            action = np.asarray(fr["action"], dtype=np.float32)
            global_idx[j] = int(fr["index"])

            tcp_xyz[j] = state[6:9]
            probe_xyz[j] = state[13:16]
            needle_xyz[j] = state[20:23]

            act_probe_xyz[j] = action[0:3]
            act_needle_xyz[j] = action[7:10]

        return EpisodeCache(
            row_indices=rows,
            global_indices=global_idx,
            T=T,
            tcp_xyz=tcp_xyz,
            probe_xyz=probe_xyz,
            needle_xyz=needle_xyz,
            act_probe_xyz=act_probe_xyz,
            act_needle_xyz=act_needle_xyz,
        )

    def load_episode(self, ep_idx: int):
        ep_idx = int(ep_idx)
        self.current_ep = ep_idx
        self.current_frame = 0

        if ep_idx not in self.cache:
            self.info_var.set(f"Loading episode {ep_idx}...")
            self.win.update_idletasks()
            self.cache[ep_idx] = self._extract_cache(ep_idx)

        c = self.cache[ep_idx]
        self.slider.configure(from_=0.0, to=float(max(c.T - 1, 0)))
        self.slider_var.set(0.0)
        self._build_plots(c)

    def _build_plots(self, c: EpisodeCache):
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()

        t = np.arange(c.T)

        self.ax1.plot(t, c.tcp_xyz[:, 0], label="tcp_x")
        self.ax1.plot(t, c.tcp_xyz[:, 1], label="tcp_y")
        self.ax1.plot(t, c.tcp_xyz[:, 2], label="tcp_z")
        self.ax1.set_ylabel("TCP xyz")
        self.ax1.legend(loc="upper right", fontsize=8)

        self.ax2.plot(t, c.probe_xyz[:, 0], label="probe_x")
        self.ax2.plot(t, c.probe_xyz[:, 1], label="probe_y")
        self.ax2.plot(t, c.probe_xyz[:, 2], label="probe_z")
        self.ax2.set_ylabel("Probe xyz")
        self.ax2.legend(loc="upper right", fontsize=8)

        self.ax3.plot(t, c.needle_xyz[:, 0], label="needle_x")
        self.ax3.plot(t, c.needle_xyz[:, 1], label="needle_y")
        self.ax3.plot(t, c.needle_xyz[:, 2], label="needle_z")
        self.ax3.set_ylabel("Needle xyz")
        self.ax3.legend(loc="upper right", fontsize=8)

        self.ax4.plot(t, c.act_probe_xyz[:, 0], label="d_probe_x")
        self.ax4.plot(t, c.act_probe_xyz[:, 1], label="d_probe_y")
        self.ax4.plot(t, c.act_probe_xyz[:, 2], label="d_probe_z")
        self.ax4.plot(t, c.act_needle_xyz[:, 0], label="d_needle_x")
        self.ax4.plot(t, c.act_needle_xyz[:, 1], label="d_needle_y")
        self.ax4.plot(t, c.act_needle_xyz[:, 2], label="d_needle_z")
        self.ax4.set_ylabel("Action Δxyz")
        self.ax4.set_xlabel("frame")
        self.ax4.legend(loc="upper right", fontsize=8)

        self.vlines = []
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            self.vlines.append(ax.axvline(self.current_frame))

        self.fig.subplots_adjust(left=0.20, right=0.98, top=0.98, bottom=0.06, hspace=0.30)
        self.canvas.draw_idle()

    def _update_vlines(self):
        for v in self.vlines:
            v.set_xdata([self.current_frame, self.current_frame])
        self.canvas.draw_idle()

    def _update_image(self):
        c = self.cache[self.current_ep]
        i = int(np.clip(self.current_frame, 0, c.T - 1))
        gidx = int(c.global_indices[i])

        us = self.us_bank.read_bgr(gidx)
        room = self.room_bank.read_bgr(gidx)
        wrgb = self.wrgb_bank.read_bgr(gidx)
        wdepth = self.wdepth_bank.read_bgr(gidx)

        # ---- FIX: swap RB for streams that are blue-tinted due to channel order ----
        # This corrects “blue skin” immediately.
        if "room" in self.swap_rb_streams:
            room = swap_rb(room)
        if "wrgb" in self.swap_rb_streams:
            wrgb = swap_rb(wrgb)

        tile_w = max(int(self.mosaic_width // 2), 320)
        us = resize_keep_aspect(us, tile_w)
        room = resize_keep_aspect(room, tile_w)
        wrgb = resize_keep_aspect(wrgb, tile_w)
        wdepth = resize_keep_aspect(wdepth, tile_w)

        mosaic = stack_2x2(us, room, wrgb, wdepth)

        fr = self.hfd[c.row_indices[i]]
        ft = fr.get("observation.meta.force_torque", None)
        if ft is not None:
            ft = np.asarray(ft, dtype=np.float32).reshape(-1)
            txt = f"ep={self.current_ep} frame={i}/{c.T-1} global={gidx} | FT=" + ", ".join([f"{x:.2f}" for x in ft[:6]])
        else:
            txt = f"ep={self.current_ep} frame={i}/{c.T-1} global={gidx}"

        cv2.putText(mosaic, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        self._tk_img = bgr_to_tk(mosaic)
        self.img_label.configure(image=self._tk_img)

    def update_all(self):
        c = self.cache[self.current_ep]
        i = int(np.clip(self.current_frame, 0, c.T - 1))
        self.info_var.set(f"episode={self.current_ep}/{self.n_eps-1}  frame={i}/{c.T-1}")
        self._update_image()
        self._update_vlines()

    def on_episode_change(self, _evt=None):
        ep_idx = int(self.ep_var.get())
        self.playing = False
        self.play_btn.configure(text="Play")
        self.load_episode(ep_idx)
        self.update_all()

    def on_slider(self, _val):
        self.current_frame = int(round(float(self.slider_var.get())))
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
            c = self.cache[self.current_ep]
            nxt = self.current_frame + 1
            if nxt >= c.T:
                nxt = 0
            self.current_frame = nxt
            self.slider_var.set(float(self.current_frame))
            self.update_all()

            delay = int(round(1000.0 / max(self.play_fps, 1e-6)))
            self.win.after(max(delay, 1), self.play_loop)
        else:
            self.win.after(50, self.play_loop)


def parse_swap_rb(s: str) -> Set[str]:
    s = (s or "").strip()
    if not s:
        return set()
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    # allowed: room, wrgb
    out = set()
    for p in parts:
        if p in ["room", "wrgb"]:
            out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LeRobot dataset root, e.g. task1_output_v2")
    ap.add_argument("--mosaic_width", type=int, default=1200)
    ap.add_argument("--swap_rb", type=str, default="room,wrgb",
                    help="Comma-separated streams to swap R/B for: room,wrgb. Default fixes blue skin on these.")
    args = ap.parse_args()

    try:
        app = LeRobotViewer042(
            dataset_root=args.root,
            mosaic_width=args.mosaic_width,
            swap_rb_streams=parse_swap_rb(args.swap_rb),
        )
        app.run()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise


if __name__ == "__main__":
    main()
