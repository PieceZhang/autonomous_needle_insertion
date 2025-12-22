#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def to_uint8_hwc3(x):
    """
    Convert image-like to uint8 HxWx3 (RGB) for visualization.
    Handles:
      - HxW
      - HxWxC
      - CxHxW (CHW)
      - HxWx1
    """
    if x is None:
        return None

    try:
        arr = np.array(x)
    except Exception:
        return None

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[1] > 10 and arr.shape[2] > 10:
        arr = np.transpose(arr, (1, 2, 0))

    # grayscale -> 3ch
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # HWC with 1 channel -> 3ch
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # ensure 3 channels
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None

    # drop alpha if exists
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # float -> uint8
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            # common case [0,1]
            if np.nanmax(arr) <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # now arr is RGB uint8 HWC
    return arr


def resize_keep_aspect(img_rgb, target_w):
    h, w = img_rgb.shape[:2]
    if w == target_w:
        return img_rgb
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    return cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def pad_to_size(img_rgb, H, W):
    h, w = img_rgb.shape[:2]
    out = np.zeros((H, W, 3), dtype=np.uint8)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img_rgb
    return out


def make_grid_2x2(imgs_rgb, titles=None):
    """
    imgs_rgb: list of 4 images (HWC RGB uint8) or None
    We assume they already have same size; if not, user should resize/pad before.
    """
    # replace None with black
    for i in range(4):
        if imgs_rgb[i] is None:
            imgs_rgb[i] = np.zeros((240, 320, 3), dtype=np.uint8)

    # determine cell size (use max)
    hs = [im.shape[0] for im in imgs_rgb]
    ws = [im.shape[1] for im in imgs_rgb]
    cell_h = max(hs)
    cell_w = max(ws)

    # pad each to same cell size
    cells = [pad_to_size(im, cell_h, cell_w) for im in imgs_rgb]

    top = np.concatenate([cells[0], cells[1]], axis=1)
    bot = np.concatenate([cells[2], cells[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)

    if titles is not None:
        # draw titles (in BGR for cv2)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        color = (0, 255, 0)

        # positions
        cv2.putText(grid_bgr, titles[0], (10, 30), font, scale, color, thick, cv2.LINE_AA)
        cv2.putText(grid_bgr, titles[1], (cell_w + 10, 30), font, scale, color, thick, cv2.LINE_AA)
        cv2.putText(grid_bgr, titles[2], (10, cell_h + 30), font, scale, color, thick, cv2.LINE_AA)
        cv2.putText(grid_bgr, titles[3], (cell_w + 10, cell_h + 30), font, scale, color, thick, cv2.LINE_AA)

        grid = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)

    return grid


def plot_action_delta_error(states, actions, out_png: Path):
    """
    expected_action = [state_next[6:13]-state_now[6:13], state_next[20:27]-state_now[20:27]]
    last frame expected = 0
    """
    T = states.shape[0]
    if T < 2:
        return

    idx_probe_ur = slice(6, 13)
    idx_needle_ndi = slice(20, 27)

    expected = np.zeros_like(actions, dtype=np.float32)
    expected[:-1, :7] = states[1:, idx_probe_ur] - states[:-1, idx_probe_ur]
    expected[:-1, 7:] = states[1:, idx_needle_ndi] - states[:-1, idx_needle_ndi]
    expected[-1, :] = 0.0

    err = np.max(np.abs(actions - expected), axis=1)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(err)
    plt.title("Per-frame max|action - expected_delta(state)|")
    plt.xlabel("frame index in window")
    plt.ylabel("max abs error")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="LeRobot dataset root folder")
    parser.add_argument("--start", type=int, default=0, help="Global start index")
    parser.add_argument("--length", type=int, default=300, help="How many frames to export (before stride)")
    parser.add_argument("--stride", type=int, default=1, help="Use every k-th frame")
    parser.add_argument("--fps", type=float, default=30.0, help="Output mp4 fps")
    parser.add_argument("--cell_width", type=int, default=640, help="Resize each view to this width (keep aspect).")
    parser.add_argument("--out_mp4", type=str, default="lerobot_grid.mp4", help="Output mp4 path")
    parser.add_argument("--out_err_png", type=str, default="delta_error.png", help="Output delta error curve png")
    parser.add_argument("--write_error_plot", action="store_true", help="Also write delta_error.png for the window")
    args = parser.parse_args()

    ds = LeRobotDataset(args.root)
    N = len(ds)
    if N == 0:
        raise RuntimeError("Dataset is empty")

    start = max(0, min(args.start, N - 1))
    end = min(N, start + args.length)
    indices = list(range(start, end, args.stride))
    if len(indices) < 2:
        raise RuntimeError("Not enough frames after stride.")

    keys = [
        "observation.images.ultrasound",
        "observation.images.room_rgb_camera",
        "observation.images.wrist_camera_rgb",
        "observation.images.wrist_camera_depth",
    ]
    titles = ["ultrasound", "room_rgb", "wrist_rgb", "wrist_depth"]

    # Collect numeric series for optional error plot
    states = []
    actions = []

    # Prepare writer after we know output frame size
    writer = None
    out_mp4 = str(Path(args.out_mp4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for k, i in enumerate(indices):
        x = ds[i]

        states.append(np.array(x["observation.state"], dtype=np.float32))
        actions.append(np.array(x["action"], dtype=np.float32))

        imgs = []
        for key in keys:
            img = to_uint8_hwc3(x.get(key))
            if img is None:
                imgs.append(None)
                continue
            img = resize_keep_aspect(img, args.cell_width)
            imgs.append(img)

        grid = make_grid_2x2(imgs, titles=titles)

        # Initialize writer
        if writer is None:
            H, W = grid.shape[:2]
            writer = cv2.VideoWriter(out_mp4, fourcc, float(args.fps), (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for: {out_mp4}")
            print(f"[INFO] Writing mp4: {out_mp4}  size=({W},{H}) fps={args.fps} frames={len(indices)}")

        # cv2 expects BGR
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        writer.write(grid_bgr)

        if (k % 50) == 0:
            print(f"[INFO] wrote {k}/{len(indices)} frames...")

    if writer is not None:
        writer.release()

    print(f"✅ MP4 saved to: {out_mp4}")

    if args.write_error_plot:
        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        out_err = Path(args.out_err_png)
        plot_action_delta_error(states, actions, out_err)
        print(f"✅ Delta error plot saved to: {out_err}")


if __name__ == "__main__":
    main()
