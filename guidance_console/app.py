from __future__ import annotations

import base64
import csv
import io
import json
import os
import socket
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import dash
import numpy as np
import pydicom
import trimesh
from dash import Input, Output, State, dcc, html
from PIL import Image
import plotly.graph_objects as go
from werkzeug.middleware.proxy_fix import ProxyFix

try:
    from scipy import ndimage
except Exception:
    ndimage = None

try:
    from skimage import measure
except Exception:
    measure = None

try:
    import straight_line_planner as straight_line_planner_module
    STRAIGHT_LINE_IMPORT_ERROR = ""
except Exception as exc:
    straight_line_planner_module = None
    STRAIGHT_LINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

MRI_SURFACE_PICK_TAG = "mri-surface"
TRACKER_DEVICE_ID = "TrackerDevice"

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DATA_ROOT = Path(os.environ.get("UI_DATA_ROOT", str(PROJECT_ROOT))).expanduser().resolve()
MRI_DIR = DATA_ROOT / "data" / "preop" / "MRI" / "SE16"
PROBE_STEP = MODULE_ROOT / "needle_and_probe" / "C5-1.step"
PROBE_STL_FALLBACK = MODULE_ROOT / "needle_and_probe" / "c5-1.STL"
AFFINE_NPZ = MODULE_ROOT / "affine_matrix" / "registration_result_MRI_post.npz"
CALIB_XML = DATA_ROOT / "calibration" / "PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml"
INTROOP_RECORDING_DIR = DATA_ROOT / "data" / "intraop_recording" / "rosbag2_20260319_083246_rigidregistration_unknown"
POSE_NDJSON = INTROOP_RECORDING_DIR / "ndi__us_probe_pose" / "messages.ndjson"
VIDEO_INFO = INTROOP_RECORDING_DIR / "image_raw__compressed" / "video_info.json"
VIDEO_PATH = INTROOP_RECORDING_DIR / "image_raw__compressed" / "video.mp4"
MRI_MARKERS_TSV = DATA_ROOT / "registration" / "data" / "preop" / "abdominal_MRI_paired.tsv"


@dataclass
class PoseSample:
    t_ns: int
    T_pol_probe: np.ndarray


def _quat_to_rot(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _load_transform(root: ET.Element, frm: str, to: str):
    elem = root.find(f".//Transform[@From='{frm}'][@To='{to}']")
    if elem is None:
        raise RuntimeError(f"Missing Transform From='{frm}' To='{to}' in {CALIB_XML}")
    vals = [float(x) for x in elem.get("Matrix", "").replace("\t", " ").replace("\n", " ").split()]
    if len(vals) != 16:
        raise RuntimeError(f"Transform From='{frm}' To='{to}' is not a 4x4 matrix in {CALIB_XML}")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def _read_local_time_offset_sec(xml_path: Path, device_id: str | None = None):
    root = ET.parse(str(xml_path)).getroot()
    if device_id is not None:
        for elem in root.iter():
            if elem.tag.endswith("Device") and elem.attrib.get("Id", "") == device_id:
                if "LocalTimeOffsetSec" in elem.attrib:
                    return float(elem.attrib["LocalTimeOffsetSec"])
        raise RuntimeError(f"LocalTimeOffsetSec not found on Device Id='{device_id}' in {xml_path}")

    for elem in root.iter():
        if "LocalTimeOffsetSec" in elem.attrib:
            return float(elem.attrib["LocalTimeOffsetSec"])
    raise RuntimeError(f"LocalTimeOffsetSec not found in {xml_path}")


def _load_model_to_object(root: ET.Element, object_id: str):
    elem = root.find(f".//DisplayableObject[@Id='{object_id}']")
    if elem is None:
        raise RuntimeError(f"Missing DisplayableObject Id='{object_id}' in {CALIB_XML}")
    matrix_str = elem.get("ModelToObjectTransform", "")
    vals = [float(x) for x in matrix_str.replace("\t", " ").replace("\n", " ").split()]
    if len(vals) != 16:
        raise RuntimeError(f"DisplayableObject Id='{object_id}' does not have a 4x4 ModelToObjectTransform")
    return np.array(vals, dtype=np.float64).reshape(4, 4)


def _project_to_se3(T):
    T = np.asarray(T, dtype=np.float64).copy()
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T[:3, :3] = R
    T[3, :] = [0.0, 0.0, 0.0, 1.0]
    return T


def _rot_y_90():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return T


def _translate_probe_towards_slice():
    T = np.eye(4, dtype=np.float64)
    # Local correction so the probe housing sits closer to the ultrasound plane.
    T[:3, 3] = np.array([0.0, 0.0, 0.025], dtype=np.float64)
    return T


def _read_volume():
    if not MRI_DIR.exists():
        raise FileNotFoundError(f"MRI directory not found: {MRI_DIR}")

    dcm_files = sorted(p for p in MRI_DIR.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(p for p in MRI_DIR.iterdir() if p.is_file() and not p.name.startswith("._"))
    if not dcm_files:
        raise RuntimeError(f"No DICOM files found in {MRI_DIR}")

    slices = []
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), force=True)
        if not hasattr(ds, "pixel_array"):
            continue
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        img = img * slope + intercept
        ipp = np.asarray(getattr(ds, "ImagePositionPatient", [0.0, 0.0, float(len(slices))]), dtype=np.float64)
        iop = np.asarray(getattr(ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=np.float64)
        row_dir = iop[:3]
        col_dir = iop[3:]
        normal = np.cross(row_dir, col_dir)
        n_norm = np.linalg.norm(normal)
        normal = normal / (n_norm if n_norm > 1e-12 else 1.0)
        pos_along_normal = float(np.dot(ipp, normal))
        slices.append((p, pos_along_normal, ds, img, ipp, row_dir, col_dir, normal))

    if not slices:
        raise RuntimeError(f"No readable DICOM slices with pixel data in {MRI_DIR}")

    slices.sort(key=lambda x: x[1])
    vol = np.stack([s[3] for s in slices], axis=0).astype(np.float32)

    ds0 = slices[0][2]
    spacing_xy = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    if len(slices) > 1:
        dz = float(np.median(np.diff([s[1] for s in slices])))
        dz = abs(dz) if abs(dz) > 1e-9 else float(getattr(ds0, "SliceThickness", 1.0))
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0))
    spacing = np.array([dz, spacing_xy[0], spacing_xy[1]], dtype=np.float64) * 1e-3
    # Build patient(mm) -> voxel(z,y,x) transform from DICOM geometry.
    ipp0 = slices[0][4]
    row_dir = slices[0][5]
    col_dir = slices[0][6]
    normal = slices[0][7]
    z_max = vol.shape[0] - 1
    x_max = vol.shape[2] - 1
    # Rotate the MRI by 180 degrees around the R-slice axis (voxel y),
    # which is equivalent to flipping the x and z voxel axes.
    ipp0_rotated = (
        ipp0
        + row_dir * spacing_xy[1] * float(x_max)
        + normal * dz * float(z_max)
    )
    voxel_to_patient = np.eye(4, dtype=np.float64)
    voxel_to_patient[:3, 0] = -row_dir * spacing_xy[1]  # x: column index
    voxel_to_patient[:3, 1] = col_dir * spacing_xy[0]   # y: row index
    voxel_to_patient[:3, 2] = -normal * dz              # z: slice index
    voxel_to_patient[:3, 3] = ipp0_rotated
    patient_to_voxel = np.linalg.inv(voxel_to_patient)
    return vol, spacing, patient_to_voxel


def _load_affine():
    data = np.load(AFFINE_NPZ)
    mats = [data[k] for k in data.files if data[k].shape == (4, 4)]
    if mats:
        M = mats[0].astype(np.float64)
        return {"mode": "mat4", "M": M, "expects_mm": False}

    if {"R", "t"}.issubset(set(data.files)):
        R = np.asarray(data["R"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
        s = float(np.asarray(data["s"], dtype=np.float64)) if "s" in data.files else 1.0
        if R.shape != (3, 3) or t.shape[0] != 3:
            raise RuntimeError(f"Invalid R/t shape in affine npz: R={R.shape}, t={t.shape}")
        # Registration output follows row-vector form: x_mri = s * (x_sensor @ R) + t.
        return {"mode": "rt", "R": R, "t": t[:3], "s": s, "expects_mm": True}

    raise RuntimeError("No usable affine in npz (expected a 4x4 matrix or R/t[/s])")


def _load_calibration():
    root = ET.parse(CALIB_XML).getroot()
    image_in_probe = _load_transform(root, "Image", "Probe")
    image_in_top = _load_transform(root, "Image", "TransducerOriginPixel")
    top_in_to = _load_transform(root, "TransducerOriginPixel", "TransducerOrigin")
    probe_model_to_to = _load_model_to_object(root, "ProbeModel")

    # Rebuild the chain the same way as us_probe.py:
    # to_in_probe = image_in_probe @ inv(image_in_top) @ inv(top_in_to)
    # Then image points are mapped through Image -> TOP -> TO -> Probe.
    to_in_probe = image_in_probe @ np.linalg.inv(image_in_top) @ np.linalg.inv(top_in_to)
    to_in_probe[:3, 3] *= 1e-3
    to_in_probe = _project_to_se3(to_in_probe)

    top_in_to_m = top_in_to.copy()
    top_in_to_m[:3, :] *= 1e-3
    probe_in_image = to_in_probe @ top_in_to_m @ image_in_top

    probe_model_to_to_m = probe_model_to_to.copy()
    probe_model_to_to_m[:3, 3] *= 1e-3

    return {
        "image_in_probe": probe_in_image,
        "to_in_probe": to_in_probe,
        "image_in_top": image_in_top,
        "top_in_to": top_in_to_m,
        "probe_model_to_to": probe_model_to_to_m,
    }


def _load_probe_poses():
    poses = []
    with POSE_NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pose = obj["data"]["pose"]
            p = pose["position"]
            q = pose["orientation"]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _quat_to_rot(q["x"], q["y"], q["z"], q["w"])
            T[:3, 3] = [p["x"], p["y"], p["z"]]
            poses.append(PoseSample(obj["publish_time_ns"], T))
    return poses


def _load_probe_mesh():
    candidates = [PROBE_STEP, PROBE_STL_FALLBACK]
    errors = []
    for mesh_path in candidates:
        if not mesh_path.exists():
            continue
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if mesh is None or len(mesh.vertices) == 0:
                raise RuntimeError(f"Empty mesh from: {mesh_path}")
            if mesh_path != PROBE_STEP and errors:
                print("STEP load failed; using STL fallback. " + " | ".join(errors))
            return mesh, mesh_path
        except Exception as exc:
            errors.append(f"{mesh_path.name}: {exc}")
    raise RuntimeError("Failed to load probe mesh from STEP/STL candidates. " + " | ".join(errors))


def _load_marker_table(path: Path):
    names = []
    segment_a = []
    segment_b = []
    pair_dist = []
    points_mm = []
    if not path.exists():
        return {
            "names": names,
            "segment_a": segment_a,
            "segment_b": segment_b,
            "pair_dist": np.empty((0,), dtype=np.float64),
            "points_mm": np.empty((0, 3), dtype=np.float64),
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            names.append(row.get("Marker", f"Marker_{len(names) + 1}"))
            segment_a.append(row.get("Segment_A", ""))
            segment_b.append(row.get("Segment_B", ""))
            pair_dist.append(float(row.get("Pair_Distance", 0.0)))
            points_mm.append([
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            ])

    return {
        "names": names,
        "segment_a": segment_a,
        "segment_b": segment_b,
        "pair_dist": np.asarray(pair_dist, dtype=np.float64),
        "points_mm": np.asarray(points_mm, dtype=np.float64).reshape(-1, 3),
    }


def _b64_img(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")


def _to_display_uint8(arr, lo_pct=1.0, hi_pct=99.0):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _foreground_values(arr, min_fraction=0.02):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.empty((0,), dtype=np.float32)
    vals = arr[finite]
    if vals.size == 0:
        return vals

    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        return vals

    threshold = vmin + (vmax - vmin) * min_fraction
    fg = vals[vals > threshold]
    if fg.size < max(64, vals.size // 100):
        fg = vals[vals > np.percentile(vals, 20.0)]
    return fg if fg.size else vals


def _auto_window_uint8(arr, lo_pct=0.5, hi_pct=99.5):
    arr = np.asarray(arr, dtype=np.float32)
    fg = _foreground_values(arr)
    if fg.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(fg, lo_pct))
    hi = float(np.percentile(fg, hi_pct))
    if hi <= lo:
        lo = float(fg.min())
        hi = float(fg.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _crop_foreground(img, padding=8):
    if img.ndim != 2:
        return img
    vals = _foreground_values(img, min_fraction=0.08)
    if vals.size == 0:
        return img
    threshold = float(np.percentile(vals, 15.0))
    mask = img > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(img.shape[0], y1 + padding)
    x1 = min(img.shape[1], x1 + padding)
    return img[y0:y1, x0:x1]


def _window_level_uint8(arr, level, width):
    arr = np.asarray(arr, dtype=np.float32)
    width = max(float(width), 1e-6)
    lo = float(level) - width / 2.0
    hi = float(level) + width / 2.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _resize_with_letterbox(img, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _render_slice_tile(img, title, target_w=220, target_h=220):
    del title
    del target_w, target_h
    return img


def _slice_header_style(color):
    return {
        "background": color,
        "color": "#111",
        "padding": "6px 10px",
        "fontWeight": "700",
        "fontSize": "18px",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
    }


def _hex_to_rgb(color: str):
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {color}")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _rotation_matrix_zyx_axes(rot_x_deg: float, rot_y_deg: float, rot_z_deg: float):
    # Coordinates here are ordered as (z, y, x).
    rx = np.deg2rad(float(rot_x_deg))
    ry = np.deg2rad(float(rot_y_deg))
    rz = np.deg2rad(float(rot_z_deg))

    rot_x = np.array(
        [
            [np.cos(rx), -np.sin(rx), 0.0],
            [np.sin(rx), np.cos(rx), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rz), -np.sin(rz)],
            [0.0, np.sin(rz), np.cos(rz)],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def _panel_height_from_physical_aspect(panel_width_px: int, span_u_mm: float, span_v_mm: float):
    panel_width_px = int(max(1, panel_width_px))
    span_u_mm = float(max(span_u_mm, 1e-6))
    span_v_mm = float(max(span_v_mm, 1e-6))
    return max(1, int(round(panel_width_px * span_v_mm / span_u_mm)))


def _normalize_vec(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if n <= eps:
        raise RuntimeError("Vector norm is too small.")
    return vec / n


def _plane_points_from_store(data):
    if not data or len(data) != 3:
        return None
    try:
        pts = np.asarray([[float(item["x"]), float(item["y"]), float(item["z"])] for item in data], dtype=np.float64)
    except Exception:
        return None
    if pts.shape != (3, 3) or not np.all(np.isfinite(pts)):
        return None
    return pts


def _surface_pick_from_click_data(click_data):
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    if point.get("customdata") != MRI_SURFACE_PICK_TAG:
        return None
    if any(key not in point for key in ("x", "y", "z")):
        return None
    return {
        "x": float(point["x"]),
        "y": float(point["y"]),
        "z": float(point["z"]),
    }


def _pick_status_text(point_count: int):
    point_count = int(max(0, point_count))
    if point_count <= 0:
        return "Pick 3 points on the MRI surface to define a custom slice."
    if point_count < 3:
        return f"Picked {point_count}/3 surface points."
    return "Picked 3/3 surface points. (3,2) and (3,1) now use the custom plane."


def _to_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected a 2D grayscale or 3D RGB image.")
    return np.clip(img, 0, 255).astype(np.uint8)


def _interactive_image_figure(img, marker_xy=None):
    rgb = _to_rgb_uint8(img)
    h, w = rgb.shape[:2]
    fig = go.Figure()
    step = max(4, int(round(min(h, w) / 80.0)))
    xs = np.arange(0, max(1, w), step, dtype=np.float32)
    ys = np.arange(0, max(1, h), step, dtype=np.float32)
    if xs.size == 0 or xs[-1] != float(max(0, w - 1)):
        xs = np.unique(np.r_[xs, float(max(0, w - 1))]).astype(np.float32)
    if ys.size == 0 or ys[-1] != float(max(0, h - 1)):
        ys = np.unique(np.r_[ys, float(max(0, h - 1))]).astype(np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    fig.add_trace(
        go.Scatter(
            x=grid_x.reshape(-1),
            y=grid_y.reshape(-1),
            mode="markers",
            marker=dict(
                size=max(8, step + 4),
                color="rgba(0,0,0,0.01)",
                line=dict(color="rgba(0,0,0,0.01)", width=1),
            ),
            hoverinfo="none",
            showlegend=False,
        )
    )

    if marker_xy is not None:
        x = float(np.clip(marker_xy[0], 0, max(0, w - 1)))
        y = float(np.clip(marker_xy[1], 0, max(0, h - 1)))
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=12,
                    color="#ff00ff",
                    line=dict(color="#ffffff", width=1),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_shape(
            type="line",
            x0=0,
            x1=max(0, w - 1),
            y0=y,
            y1=y,
            line=dict(color="#22d3ee", width=1),
        )
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=max(0, h - 1),
            line=dict(color="#22d3ee", width=1),
        )

    fig.update_xaxes(visible=False, range=[0, max(0, w - 1)], fixedrange=True)
    fig.update_yaxes(visible=False, range=[max(0, h - 1), 0], scaleanchor="x", scaleratio=1, fixedrange=True)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#000",
        plot_bgcolor="#000",
        clickmode="event+select",
        hovermode=False,
        uirevision="image-static",
        images=[
            dict(
                source=_b64_img(rgb),
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=max(1, w),
                sizey=max(1, h),
                sizing="stretch",
                layer="below",
                yanchor="top",
            )
        ],
    )
    return fig


def _extract_click_xy(click_data):
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    if "x" not in point or "y" not in point:
        return None
    return float(point["x"]), float(point["y"])


def _crop_patch(img, center_xy, patch_radius=90):
    img = np.asarray(img)
    if img.ndim not in {2, 3}:
        raise ValueError("Expected a 2D or 3D image for patch cropping.")
    h, w = img.shape[:2]
    cx = int(np.clip(round(center_xy[0]), 0, max(0, w - 1)))
    cy = int(np.clip(round(center_xy[1]), 0, max(0, h - 1)))
    r = int(max(8, patch_radius))

    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((2 * r, 2 * r), dtype=np.uint8) if img.ndim == 2 else np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)

    if img.ndim == 2:
        out = np.zeros((2 * r, 2 * r), dtype=img.dtype)
    else:
        out = np.zeros((2 * r, 2 * r, img.shape[2]), dtype=img.dtype)
    oy = (out.shape[0] - patch.shape[0]) // 2
    ox = (out.shape[1] - patch.shape[1]) // 2
    out[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
    return out, (cx, cy)


def _blank_patch(size=180):
    size = int(max(16, size))
    return np.zeros((size, size, 3), dtype=np.uint8)


def _draw_crosshair(img, marker_xy, color=(34, 211, 238), cross_color=(255, 0, 255)):
    rgb = _to_rgb_uint8(img).copy()
    h, w = rgb.shape[:2]
    x = int(np.clip(round(marker_xy[0]), 0, max(0, w - 1)))
    y = int(np.clip(round(marker_xy[1]), 0, max(0, h - 1)))
    cv2.line(rgb, (0, y), (max(0, w - 1), y), color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(rgb, (x, 0), (x, max(0, h - 1)), color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        rgb,
        (x, y),
        color=cross_color,
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=14,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    return rgb


def _orient_slice_image(img, axis_name: str, row_spacing: float, col_spacing: float, interpolation=cv2.INTER_LINEAR):
    img = np.asarray(img)
    if axis_name == "axial":
        img = np.flipud(img)
    elif axis_name == "coronal":
        img = np.flipud(img)
    else:
        img = np.fliplr(np.rot90(img, k=1))
        row_spacing, col_spacing = col_spacing, row_spacing

    img = np.rot90(img, k=-1)
    row_spacing, col_spacing = col_spacing, row_spacing
    if axis_name == "sagittal":
        img = np.flipud(img)

    scale_base = min(row_spacing, col_spacing)
    out_h = max(1, int(round(img.shape[0] * row_spacing / scale_base)))
    out_w = max(1, int(round(img.shape[1] * col_spacing / scale_base)))
    if out_h != img.shape[0] or out_w != img.shape[1]:
        img = cv2.resize(img, (out_w, out_h), interpolation=interpolation)
    return img


def _blend_marker_overlay(gray_img, mask, color_rgb):
    base = np.repeat(np.asarray(gray_img, dtype=np.uint8)[:, :, None], 3, axis=2).astype(np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[:2] != base.shape[:2]:
        raise ValueError("Marker overlay mask must match the grayscale image shape.")
    alpha = (mask / 255.0)[:, :, None] * 0.9
    color = np.asarray(color_rgb, dtype=np.float32)[None, None, :]
    return np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 255.0).astype(np.uint8)


def _sample_trilinear(vol, xyz):
    z, y, x = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    z0, y0, x0 = np.floor(z).astype(int), np.floor(y).astype(int), np.floor(x).astype(int)
    z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
    valid = (z0 >= 0) & (y0 >= 0) & (x0 >= 0) & (z1 < vol.shape[0]) & (y1 < vol.shape[1]) & (x1 < vol.shape[2])
    out = np.zeros_like(z, dtype=np.float32)
    if not np.any(valid):
        return out
    z, y, x = z[valid], y[valid], x[valid]
    z0, y0, x0, z1, y1, x1 = z0[valid], y0[valid], x0[valid], z1[valid], y1[valid], x1[valid]
    dz, dy, dx = z - z0, y - y0, x - x0

    c000 = vol[z0, y0, x0]
    c001 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c011 = vol[z0, y1, x1]
    c100 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c110 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    out[valid] = c0 * (1 - dz) + c1 * dz
    return out


class AppData:
    def __init__(self):
        self.volume, self.spacing_m, self.patient_mm_to_voxel = _read_volume()
        self.spacing_zyx_mm = self.spacing_m * 1e3
        self.volume_extent_zyx_mm = np.maximum(np.asarray(self.volume.shape, dtype=np.float64) - 1.0, 1.0) * self.spacing_zyx_mm
        self.voxel_xyz_to_patient_mm = np.linalg.inv(self.patient_mm_to_voxel)
        self.mri_g_axis_patient_unit = _normalize_vec(self.voxel_xyz_to_patient_mm[:3, 0])
        self.mri_g_offset_mm = 25.0
        self.affine_cfg = _load_affine()
        self.calibration = _load_calibration()
        self.tracker_local_time_offset_sec = _read_local_time_offset_sec(CALIB_XML, TRACKER_DEVICE_ID)
        self.video_time_offset_sec = 0.0
        self.T_probe_image = self.calibration["image_in_probe"]
        self.T_probe_to = self.calibration["to_in_probe"]
        self.poses = _load_probe_poses()
        self.mri_fg_values = _foreground_values(self.volume)
        if self.mri_fg_values.size == 0:
            self.mri_fg_values = self.volume.reshape(-1)
        self.mri_min = float(np.min(self.mri_fg_values))
        self.mri_max = float(np.max(self.mri_fg_values))
        self.default_wl_low = float(np.percentile(self.mri_fg_values, 0.5))
        self.default_wl_high = float(np.percentile(self.mri_fg_values, 99.5))
        self.default_level = 0.5 * (self.default_wl_low + self.default_wl_high)
        self.default_width = self.default_wl_high - self.default_wl_low
        self.panel_configs = {
            "axial": {"source_axis": 1, "title": "R", "color": "#ef4444"},
            "coronal": {"source_axis": 2, "title": "G", "color": "#84cc16"},
            "sagittal": {"source_axis": 0, "title": "Y", "color": "#facc15"},
        }
        self.slice_max = {
            panel_id: self.volume.shape[cfg["source_axis"]] - 1
            for panel_id, cfg in self.panel_configs.items()
        }
        self.slice_default = {
            panel_id: self.volume.shape[cfg["source_axis"]] // 2
            for panel_id, cfg in self.panel_configs.items()
        }
        self.marker_table = _load_marker_table(MRI_MARKERS_TSV)
        self.mri_marker_names = self.marker_table["names"]
        self.mri_marker_points_mm = self.marker_table["points_mm"]
        self.mri_marker_points_zyx = self.patient_mm_to_voxel_batch(self.mri_marker_points_mm)
        self.mri_marker_points_world_mm = self.mri_mm_to_world_mm_batch(self.mri_marker_points_mm)
        self.mri_marker_hover = [
            (
                f"{name}<br>"
                f"{seg_a} / {seg_b}<br>"
                f"MRI(mm)=({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})<br>"
                f"pair={dist:.2f} mm"
            )
            for name, seg_a, seg_b, dist, pt in zip(
                self.mri_marker_names,
                self.marker_table["segment_a"],
                self.marker_table["segment_b"],
                self.marker_table["pair_dist"],
                self.mri_marker_points_mm,
            )
        ]

        with VIDEO_INFO.open("r", encoding="utf-8") as f:
            vi = json.load(f)
        self.video_start_ns = int(vi["start_time_ns"])
        self.video_fps = float(vi["measured_fps"])
        self.frame_count = int(vi["frame_count"])
        self.video_width = int(vi.get("resolution", {}).get("width", 1920))
        self.video_height = int(vi.get("resolution", {}).get("height", 1080))
        self.cap = cv2.VideoCapture(str(VIDEO_PATH))
        self.cap_hq = cv2.VideoCapture(str(VIDEO_PATH))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
        if not self.cap_hq.isOpened():
            raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
        self.cap_lock = threading.Lock()
        self.cap_hq_lock = threading.Lock()
        self.cap_next_frame_idx = None
        self.cap_hq_next_frame_idx = None
        self.frame_cache = OrderedDict()
        self.frame_cache_max = 48
        self.frame_hq_cache = OrderedDict()
        self.frame_hq_cache_max = 24

        self.pose_times = np.array([p.t_ns for p in self.poses], dtype=np.int64)
        if len(self.pose_times) > 1:
            pose_dt_ms = float(np.median(np.diff(self.pose_times))) / 1e6
        else:
            pose_dt_ms = 100.0
        self.play_interval_ms = int(np.clip(round(pose_dt_ms), 30, 250))
        self.probe_mesh, self.probe_mesh_source = _load_probe_mesh()
        print(f"Using probe mesh: {self.probe_mesh_source}")
        self.probe_model_in_probe = self.T_probe_to @ self.calibration["probe_model_to_to"] @ _rot_y_90() @ _translate_probe_towards_slice()
        self.us_plane_width = 960
        self.us_plane_height = max(1, int(round(self.us_plane_width * self.video_height / self.video_width)))
        self.us_plane_mesh_cols = 144
        self.us_plane_mesh_rows = 81
        self.us_hq_width = 1280
        self.us_hq_height = max(1, int(round(self.us_hq_width * self.video_height / self.video_width)))
        self.resample_panel_width = 720
        self.probe_plane_u_span_mm, self.probe_plane_v_span_mm = self._probe_plane_extent_mm()
        self.resample_panel_height = _panel_height_from_physical_aspect(
            self.resample_panel_width,
            self.probe_plane_u_span_mm,
            self.probe_plane_v_span_mm,
        )
        self.auto_section_patch_radius = 90
        self.oblique_panel_width = 860
        self.oblique_half_u_mm = 0.45 * max(1.0, self.volume_extent_zyx_mm[2])
        self.oblique_half_v_mm = 0.45 * max(1.0, self.volume_extent_zyx_mm[1])
        self.oblique_panel_height = _panel_height_from_physical_aspect(
            self.oblique_panel_width,
            2.0 * self.oblique_half_u_mm,
            2.0 * self.oblique_half_v_mm,
        )
        self.oblique_center_default = {
            "z": self.volume.shape[0] // 2,
            "y": self.volume.shape[1] // 2,
            "x": self.volume.shape[2] // 2,
        }
        uu = np.linspace(0.0, self.video_width - 1.0, self.us_plane_width, dtype=np.float64)
        vv = np.linspace(0.0, self.video_height - 1.0, self.us_plane_height, dtype=np.float64)
        self.grid_u, self.grid_v = np.meshgrid(uu, vv, indexing="xy")
        try:
            self.mri_surface_base = self._build_mri_surface_mesh()
        except Exception as exc:
            print(f"MRI surface reconstruction unavailable: {exc}")
            self.mri_surface_base = None
        self.mri_marker_3d_trace = self.mri_marker_trace()

    @staticmethod
    def pose_m_to_display_mm(T):
        T_mm = np.asarray(T, dtype=np.float64).copy()
        T_mm[:3, 3] *= 1e3
        return T_mm

    def frame_to_pose_idx(self, frame_idx: int):
        t = self.video_start_ns + int(1e9 * frame_idx / self.video_fps)
        aligned_pose_times = self.pose_times + int(round(self.video_time_offset_sec * 1e9))
        i = int(np.searchsorted(aligned_pose_times, t, side="left"))
        if i <= 0:
            return 0
        if i >= len(aligned_pose_times):
            return len(aligned_pose_times) - 1
        if abs(aligned_pose_times[i] - t) < abs(t - aligned_pose_times[i - 1]):
            return i
        i = i - 1
        return i

    def pose_idx_to_frame_idx(self, pose_idx: int):
        pose_idx = int(np.clip(pose_idx, 0, len(self.poses) - 1))
        t = int(self.pose_times[pose_idx] + round(self.video_time_offset_sec * 1e9))
        frame_float = (t - self.video_start_ns) * self.video_fps / 1e9
        frame_idx = int(round(frame_float))
        return int(np.clip(frame_idx, 0, self.frame_count - 1))

    def read_us_frame(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.frame_count - 1))
        cached = self.frame_cache.get(frame_idx)
        if cached is not None:
            self.frame_cache.move_to_end(frame_idx)
            return cached

        with self.cap_lock:
            if self.cap_next_frame_idx != frame_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self.cap_next_frame_idx = frame_idx
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {VIDEO_PATH}")
            self.cap_next_frame_idx = frame_idx + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != self.us_plane_width or gray.shape[0] != self.us_plane_height:
            gray = cv2.resize(gray, (self.us_plane_width, self.us_plane_height), interpolation=cv2.INTER_AREA)
        display = _to_display_uint8(gray, lo_pct=2.0, hi_pct=99.5)
        self.frame_cache[frame_idx] = display
        self.frame_cache.move_to_end(frame_idx)
        if len(self.frame_cache) > self.frame_cache_max:
            self.frame_cache.popitem(last=False)
        return display

    def read_us_frame_hq(self, frame_idx: int):
        frame_idx = int(np.clip(frame_idx, 0, self.frame_count - 1))
        cached = self.frame_hq_cache.get(frame_idx)
        if cached is not None:
            self.frame_hq_cache.move_to_end(frame_idx)
            return cached

        with self.cap_hq_lock:
            if self.cap_hq_next_frame_idx != frame_idx:
                self.cap_hq.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                self.cap_hq_next_frame_idx = frame_idx
            ok, frame = self.cap_hq.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {VIDEO_PATH}")
            self.cap_hq_next_frame_idx = frame_idx + 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] != self.us_hq_width or gray.shape[0] != self.us_hq_height:
            gray = cv2.resize(gray, (self.us_hq_width, self.us_hq_height), interpolation=cv2.INTER_CUBIC)
        display = _to_display_uint8(gray, lo_pct=1.0, hi_pct=99.8)
        self.frame_hq_cache[frame_idx] = display
        self.frame_hq_cache.move_to_end(frame_idx)
        if len(self.frame_hq_cache) > self.frame_hq_cache_max:
            self.frame_hq_cache.popitem(last=False)
        return display

    def frame_time_ns(self, frame_idx: int):
        return self.video_start_ns + int(round(1e9 * int(frame_idx) / self.video_fps))

    def mri_g_offset_patient_mm(self):
        return self.mri_g_axis_patient_unit * float(self.mri_g_offset_mm)

    def mri_g_offset_world_mm(self):
        offset_patient_mm = self.mri_g_offset_patient_mm()
        if float(np.linalg.norm(offset_patient_mm)) <= 1e-12:
            return np.zeros(3, dtype=np.float64)
        cfg = self.affine_cfg
        if cfg["mode"] != "rt":
            return np.zeros(3, dtype=np.float64)
        scale = float(cfg.get("s", 1.0))
        if abs(scale) < 1e-12:
            raise RuntimeError("Registration scale is too small to invert.")
        return offset_patient_mm @ cfg["R"].T / scale

    def patient_mm_to_voxel_batch(self, patient_mm_batch):
        pts = np.asarray(patient_mm_batch, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        pts = pts - self.mri_g_offset_patient_mm()[None, :]
        hom = np.c_[pts, np.ones((pts.shape[0], 1), dtype=np.float64)]
        vox_xyz = (self.patient_mm_to_voxel @ hom.T).T[:, :3]
        return np.c_[vox_xyz[:, 2], vox_xyz[:, 1], vox_xyz[:, 0]]

    def voxel_to_patient_mm_batch(self, voxel_zyx_batch):
        pts = np.asarray(voxel_zyx_batch, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        vox_xyz = np.c_[pts[:, 2], pts[:, 1], pts[:, 0]]
        hom = np.c_[vox_xyz, np.ones((vox_xyz.shape[0], 1), dtype=np.float64)]
        patient_mm = (self.voxel_xyz_to_patient_mm @ hom.T).T[:, :3]
        return patient_mm + self.mri_g_offset_patient_mm()[None, :]

    def mri_mm_to_world_mm_batch(self, patient_mm_batch):
        pts = np.asarray(patient_mm_batch, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        cfg = self.affine_cfg
        if cfg["mode"] != "rt":
            return np.full((pts.shape[0], 3), np.nan, dtype=np.float64)
        scale = float(cfg.get("s", 1.0))
        if abs(scale) < 1e-12:
            raise RuntimeError("Registration scale is too small to invert.")
        sensor_mm = (pts - cfg["t"][None, :]) @ cfg["R"].T / scale
        return sensor_mm

    def world_to_voxel_batch(self, xyz_m_batch):
        pts = np.asarray(xyz_m_batch, dtype=np.float64).reshape(-1, 3)
        cfg = self.affine_cfg
        if cfg["mode"] == "rt":
            sensor_mm = pts * 1e3
            mri_mm = cfg["s"] * (sensor_mm @ cfg["R"]) + cfg["t"][None, :]
            mri_mm = mri_mm - self.mri_g_offset_patient_mm()[None, :]
            hom = np.c_[mri_mm, np.ones((mri_mm.shape[0], 1), dtype=np.float64)]
            vox_xyz = (self.patient_mm_to_voxel @ hom.T).T[:, :3]
        else:
            p = np.c_[pts, np.ones((pts.shape[0], 1), dtype=np.float64)]
            if cfg.get("expects_mm", False):
                p[:, :3] *= 1e3
            c = (cfg["M"] @ p.T).T[:, :3]
            vox_xyz = c
        return np.c_[vox_xyz[:, 2], vox_xyz[:, 1], vox_xyz[:, 0]]

    def world_to_voxel(self, xyz_m):
        return self.world_to_voxel_batch(np.asarray(xyz_m, dtype=np.float64)[None, :])[0]

    def _slice_raw(self, axis_name: str, slice_idx: int):
        axis_name = axis_name.lower()
        cfg = self.panel_configs.get(axis_name)
        if cfg is None:
            raise ValueError(f"Unknown axis name: {axis_name}")

        source_axis = cfg["source_axis"]
        if source_axis == 0:
            img = self.volume[int(np.clip(slice_idx, 0, self.volume.shape[0] - 1)), :, :]
            fallback = self.volume[self.volume.shape[0] // 2, :, :]
            row_spacing, col_spacing = self.spacing_m[1], self.spacing_m[2]
        elif source_axis == 1:
            img = self.volume[:, int(np.clip(slice_idx, 0, self.volume.shape[1] - 1)), :]
            fallback = self.volume[:, self.volume.shape[1] // 2, :]
            row_spacing, col_spacing = self.spacing_m[0], self.spacing_m[2]
        else:
            img = self.volume[:, :, int(np.clip(slice_idx, 0, self.volume.shape[2] - 1))]
            fallback = self.volume[:, :, self.volume.shape[2] // 2]
            row_spacing, col_spacing = self.spacing_m[0], self.spacing_m[1]

        if float(np.max(img)) <= 1e-6:
            img = fallback
        return img, row_spacing, col_spacing, source_axis

    def oriented_slice(self, axis_name: str, slice_idx: int):
        img, row_spacing, col_spacing, _ = self._slice_raw(axis_name, slice_idx)
        return _orient_slice_image(img, axis_name, row_spacing, col_spacing, interpolation=cv2.INTER_LINEAR)

    def slice_marker_mask(self, axis_name: str, slice_idx: int, radius_px: int = 6, tol_vox: float = 1.1):
        img, row_spacing, col_spacing, source_axis = self._slice_raw(axis_name, slice_idx)
        if self.mri_marker_points_zyx.size == 0:
            empty = np.zeros(img.shape, dtype=np.uint8)
            empty = _orient_slice_image(empty, axis_name, row_spacing, col_spacing, interpolation=cv2.INTER_NEAREST)
            return empty, 0

        if source_axis == 0:
            raw_shape = self.volume.shape[1], self.volume.shape[2]
            coords_rc = self.mri_marker_points_zyx[:, [1, 2]]
        elif source_axis == 1:
            raw_shape = self.volume.shape[0], self.volume.shape[2]
            coords_rc = self.mri_marker_points_zyx[:, [0, 2]]
        else:
            raw_shape = self.volume.shape[0], self.volume.shape[1]
            coords_rc = self.mri_marker_points_zyx[:, [0, 1]]

        dist_to_plane = np.abs(self.mri_marker_points_zyx[:, source_axis] - float(slice_idx))
        on_slice = dist_to_plane <= tol_vox
        mask = np.zeros(raw_shape, dtype=np.uint8)
        for row, col in coords_rc[on_slice]:
            cv2.circle(
                mask,
                (int(round(col)), int(round(row))),
                int(radius_px),
                255,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        mask = _orient_slice_image(mask, axis_name, row_spacing, col_spacing, interpolation=cv2.INTER_NEAREST)
        return mask, int(np.count_nonzero(on_slice))

    def render_mri_slice(self, axis_name: str, slice_idx: int, title: str):
        raw = self.oriented_slice(axis_name, slice_idx)
        disp = _auto_window_uint8(raw, lo_pct=0.5, hi_pct=99.7)
        mask, marker_count = self.slice_marker_mask(axis_name, slice_idx)
        color_rgb = _hex_to_rgb(self.panel_configs[axis_name]["color"])
        rgb = _blend_marker_overlay(disp, mask, color_rgb)
        return _render_slice_tile(rgb, title), marker_count

    def image_plane_world(self, T_pol_probe, u_coords=None, v_coords=None):
        if u_coords is None:
            u_coords = np.array([0.0, self.video_width - 1.0], dtype=np.float64)
        if v_coords is None:
            v_coords = np.array([0.0, self.video_height - 1.0], dtype=np.float64)
        uu, vv = np.meshgrid(u_coords, v_coords, indexing="xy")
        pix = np.stack([uu, vv, np.zeros_like(uu), np.ones_like(uu)], axis=-1).reshape(-1, 4)
        image_in_to = (self.calibration["top_in_to"] @ self.calibration["image_in_top"] @ pix.T).T
        pts_probe = (self.T_probe_to @ image_in_to.T).T
        pts_world = (T_pol_probe @ pts_probe.T).T[:, :3]
        return (pts_world * 1e3).reshape(len(v_coords), len(u_coords), 3)

    def pose_to_slice_indices(self, pose_idx: int):
        pose_idx = int(np.clip(pose_idx, 0, len(self.poses) - 1))
        pose = self.poses[pose_idx]
        u_center = np.array([0.5 * (self.video_width - 1.0)], dtype=np.float64)
        v_center = np.array([0.5 * (self.video_height - 1.0)], dtype=np.float64)
        plane_center_world_mm = self.image_plane_world(pose.T_pol_probe, u_coords=u_center, v_coords=v_center)[0, 0]
        plane_center_world_m = plane_center_world_mm * 1e-3
        voxel_zyx = self.world_to_voxel(plane_center_world_m)
        return {
            panel_id: int(np.clip(np.rint(voxel_zyx[cfg["source_axis"]]), 0, self.slice_max[panel_id]))
            for panel_id, cfg in self.panel_configs.items()
        }

    def mesh_trace(self, mesh: trimesh.Trimesh, T, model_to_local=None):
        T = self.pose_m_to_display_mm(T)
        if model_to_local is not None:
            model_to_local = np.asarray(model_to_local, dtype=np.float64).copy()
            model_to_local[:3, 3] *= 1e3
            T = T @ model_to_local
        v = np.c_[mesh.vertices, np.ones((len(mesh.vertices), 1))]
        vw = (T @ v.T).T[:, :3]
        tri = mesh.faces
        return go.Mesh3d(
            x=vw[:, 0], y=vw[:, 1], z=vw[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            opacity=0.5, color="lightblue", showscale=False,
        )

    def needle_line_trace(self, T_pol_probe, length_m=0.18, offset_m=0.02):
        T_mm = self.pose_m_to_display_mm(T_pol_probe)
        start = T_mm[:3, 3] + T_mm[:3, 2] * (offset_m * 1e3)
        end = start + T_mm[:3, 2] * (length_m * 1e3)
        pts = np.vstack([start, end])
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color="#f97316", width=8),
            hoverinfo="skip",
        )

    def image_plane_surface_trace(self, T_pol_probe, frame_img):
        step_u = max(1, int(np.ceil(frame_img.shape[1] / self.us_plane_mesh_cols)))
        step_v = max(1, int(np.ceil(frame_img.shape[0] / self.us_plane_mesh_rows)))
        u_coords = np.linspace(0.0, self.video_width - 1.0, frame_img[::step_v, ::step_u].shape[1], dtype=np.float64)
        v_coords = np.linspace(0.0, self.video_height - 1.0, frame_img[::step_v, ::step_u].shape[0], dtype=np.float64)
        plane = self.image_plane_world(T_pol_probe, u_coords=u_coords, v_coords=v_coords)
        surface = frame_img[::step_v, ::step_u].astype(np.float32)
        return go.Surface(
            x=plane[:, :, 0],
            y=plane[:, :, 1],
            z=plane[:, :, 2],
            surfacecolor=surface,
            cmin=0,
            cmax=255,
            colorscale="Gray",
            showscale=False,
            opacity=0.95,
            hoverinfo="skip",
        )

    def _probe_plane_extent_mm(self):
        pose = self.poses[0].T_pol_probe if self.poses else np.eye(4, dtype=np.float64)
        corners = self.image_plane_world(
            pose,
            u_coords=np.array([0.0, self.video_width - 1.0], dtype=np.float64),
            v_coords=np.array([0.0, self.video_height - 1.0], dtype=np.float64),
        )
        u_span_mm = float(np.linalg.norm(corners[0, 1] - corners[0, 0]))
        v_span_mm = float(np.linalg.norm(corners[1, 0] - corners[0, 0]))
        return max(u_span_mm, 1e-6), max(v_span_mm, 1e-6)

    def render_probe_plane_mri(self, pose_idx: int):
        pose_idx = int(np.clip(pose_idx, 0, len(self.poses) - 1))
        pose = self.poses[pose_idx]
        u_coords = np.linspace(0.0, self.video_width - 1.0, self.resample_panel_width, dtype=np.float64)
        v_coords = np.linspace(0.0, self.video_height - 1.0, self.resample_panel_height, dtype=np.float64)
        plane_world_mm = self.image_plane_world(pose.T_pol_probe, u_coords=u_coords, v_coords=v_coords)
        plane_voxel = self.world_to_voxel_batch(plane_world_mm.reshape(-1, 3) * 1e-3)
        sampled = _sample_trilinear(self.volume, plane_voxel).reshape(self.resample_panel_height, self.resample_panel_width)
        disp = _auto_window_uint8(sampled, lo_pct=0.5, hi_pct=99.7)
        coverage = float(np.count_nonzero(sampled > 0.0)) / float(sampled.size)
        status = (
            f"Pose {pose_idx} | MRI resample coverage {coverage * 100.0:.1f}% | "
            f"mri_g_offset={self.mri_g_offset_mm:+.1f} mm"
        )
        return disp, status

    def _three_point_plane_frame(self, points_world_mm):
        pts = np.asarray(points_world_mm, dtype=np.float64).reshape(3, 3)
        p0, p1, p2 = pts
        u_dir = _normalize_vec(p1 - p0)
        normal = np.cross(u_dir, p2 - p0)
        normal = _normalize_vec(normal)
        v_dir = _normalize_vec(np.cross(normal, u_dir))
        center = np.mean(pts, axis=0)
        return center, u_dir, v_dir, normal

    def sample_three_point_plane_section(self, points_world_mm):
        center_world_mm, u_dir, v_dir, normal = self._three_point_plane_frame(points_world_mm)
        uu = np.linspace(-self.oblique_half_u_mm, self.oblique_half_u_mm, self.oblique_panel_width, dtype=np.float64)
        vv = np.linspace(-self.oblique_half_v_mm, self.oblique_half_v_mm, self.oblique_panel_height, dtype=np.float64)
        grid_u, grid_v = np.meshgrid(uu, vv, indexing="xy")
        plane_world_mm = (
            center_world_mm[None, None, :]
            + grid_u[:, :, None] * u_dir[None, None, :]
            + grid_v[:, :, None] * v_dir[None, None, :]
        )
        plane_voxel = self.world_to_voxel_batch(plane_world_mm.reshape(-1, 3) * 1e-3).reshape(
            self.oblique_panel_height,
            self.oblique_panel_width,
            3,
        )
        sampled = _sample_trilinear(self.volume, plane_voxel.reshape(-1, 3)).reshape(
            self.oblique_panel_height,
            self.oblique_panel_width,
        )
        coverage = float(np.count_nonzero(sampled > 0.0)) / float(sampled.size)
        meta = {
            "mode": "three-point",
            "center_world_mm": center_world_mm,
            "normal": normal,
            "u_dir": u_dir,
            "v_dir": v_dir,
            "coverage": coverage,
        }
        return sampled, meta

    def _build_mri_surface_mesh(self):
        if measure is None:
            raise RuntimeError("skimage.measure is unavailable")
        stride = int(max(2, np.ceil(max(self.volume.shape) / 160.0)))
        vol_ds = self.volume[::stride, ::stride, ::stride].astype(np.float32)
        if ndimage is not None:
            vol_ds = ndimage.gaussian_filter(vol_ds, sigma=0.8)
        fg = self.mri_fg_values if self.mri_fg_values.size else self.volume.reshape(-1)
        level = float(np.percentile(fg, 20.0))
        level = float(np.clip(level, float(np.min(vol_ds)) + 1e-6, float(np.max(vol_ds)) - 1e-6))
        verts_ds, faces, _, _ = measure.marching_cubes(vol_ds, level=level, allow_degenerate=False)
        verts_zyx = verts_ds * float(stride)
        patient_mm = self.voxel_to_patient_mm_batch(verts_zyx)
        verts_world_mm = self.mri_mm_to_world_mm_batch(patient_mm)
        if verts_world_mm.size == 0 or not np.all(np.isfinite(verts_world_mm)):
            raise RuntimeError("MRI surface vertices are invalid in world coordinates")
        return verts_world_mm, faces.astype(np.int32)

    def mri_surface_trace(self):
        if self.mri_surface_base is None:
            return go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip")
        verts_world_mm, faces = self.mri_surface_base
        verts_world_mm = np.asarray(verts_world_mm, dtype=np.float64) + self.mri_g_offset_world_mm()[None, :]

        return go.Mesh3d(
            x=verts_world_mm[:, 0],
            y=verts_world_mm[:, 1],
            z=verts_world_mm[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#d1d5db",
            opacity=0.50,
            hovertemplate="MRI surface<extra></extra>",
            customdata=np.full((verts_world_mm.shape[0],), MRI_SURFACE_PICK_TAG, dtype=object),
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.1, roughness=0.95, fresnel=0.02),
            flatshading=False,
            showscale=False,
        )

    def selected_surface_points_trace(self, points_world_mm):
        pts = np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip")
        labels = [f"P{i + 1}" for i in range(len(pts))]
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=6, color="#ec4899", line=dict(color="#ffffff", width=1)),
            hovertemplate="%{text}<extra></extra>",
        )

    def selected_surface_path_trace(self, points_world_mm):
        pts = np.asarray(points_world_mm, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip")
        if len(pts) == 3:
            pts = np.vstack([pts, pts[:1]])
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color="#ec4899", width=5),
            hoverinfo="skip",
        )

    def selected_plane_surface_trace(self, points_world_mm):
        try:
            center_world_mm, u_dir, v_dir, _ = self._three_point_plane_frame(points_world_mm)
        except Exception:
            return None
        corners = np.array([
            center_world_mm - self.oblique_half_u_mm * u_dir - self.oblique_half_v_mm * v_dir,
            center_world_mm + self.oblique_half_u_mm * u_dir - self.oblique_half_v_mm * v_dir,
            center_world_mm + self.oblique_half_u_mm * u_dir + self.oblique_half_v_mm * v_dir,
            center_world_mm - self.oblique_half_u_mm * u_dir + self.oblique_half_v_mm * v_dir,
        ], dtype=np.float64)
        grid = corners[[0, 1, 3, 2]].reshape(2, 2, 3)
        return go.Surface(
            x=grid[:, :, 0],
            y=grid[:, :, 1],
            z=grid[:, :, 2],
            surfacecolor=np.zeros((2, 2), dtype=np.float32),
            colorscale=[[0.0, "#22c55e"], [1.0, "#22c55e"]],
            opacity=0.18,
            showscale=False,
            hoverinfo="skip",
        )

    def _oblique_plane_geometry(self, center_z: float, center_y: float, center_x: float, rot_x_deg: float, rot_y_deg: float, rot_z_deg: float):
        center = np.array(
            [
                np.clip(float(center_z), 0.0, self.volume.shape[0] - 1.0),
                np.clip(float(center_y), 0.0, self.volume.shape[1] - 1.0),
                np.clip(float(center_x), 0.0, self.volume.shape[2] - 1.0),
            ],
            dtype=np.float64,
        )
        rot = _rotation_matrix_zyx_axes(rot_x_deg, rot_y_deg, rot_z_deg)
        basis_u_phys = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        basis_v_phys = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        center_phys_mm = center * self.spacing_zyx_mm
        corners_phys_mm = np.array([
            center_phys_mm - self.oblique_half_u_mm * basis_u_phys - self.oblique_half_v_mm * basis_v_phys,
            center_phys_mm + self.oblique_half_u_mm * basis_u_phys - self.oblique_half_v_mm * basis_v_phys,
            center_phys_mm + self.oblique_half_u_mm * basis_u_phys + self.oblique_half_v_mm * basis_v_phys,
            center_phys_mm - self.oblique_half_u_mm * basis_u_phys + self.oblique_half_v_mm * basis_v_phys,
        ], dtype=np.float64)
        corners_zyx = corners_phys_mm / self.spacing_zyx_mm[None, :]
        corners_world_mm = self.mri_mm_to_world_mm_batch(self.voxel_to_patient_mm_batch(corners_zyx))
        center_world_mm = self.mri_mm_to_world_mm_batch(self.voxel_to_patient_mm_batch(center[None, :]))[0]
        return {
            "center": center,
            "rot": (float(rot_x_deg), float(rot_y_deg), float(rot_z_deg)),
            "center_phys_mm": center_phys_mm,
            "basis_u_phys": basis_u_phys,
            "basis_v_phys": basis_v_phys,
            "corners_world_mm": corners_world_mm,
            "center_world_mm": center_world_mm,
        }

    def oblique_plane_surface_trace(self, center_z: float, center_y: float, center_x: float, rot_x_deg: float, rot_y_deg: float, rot_z_deg: float):
        geom = self._oblique_plane_geometry(center_z, center_y, center_x, rot_x_deg, rot_y_deg, rot_z_deg)
        pts = geom["corners_world_mm"]
        grid = pts[[0, 1, 3, 2]].reshape(2, 2, 3)
        return go.Surface(
            x=grid[:, :, 0],
            y=grid[:, :, 1],
            z=grid[:, :, 2],
            surfacecolor=np.zeros((2, 2), dtype=np.float32),
            colorscale=[[0.0, "#22c55e"], [1.0, "#22c55e"]],
            opacity=0.18,
            showscale=False,
            hoverinfo="skip",
        )

    def oblique_plane_outline_trace(self, center_z: float, center_y: float, center_x: float, rot_x_deg: float, rot_y_deg: float, rot_z_deg: float):
        geom = self._oblique_plane_geometry(center_z, center_y, center_x, rot_x_deg, rot_y_deg, rot_z_deg)
        pts = np.vstack([geom["corners_world_mm"], geom["corners_world_mm"][:1]])
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color="#22c55e", width=6),
            hoverinfo="skip",
        )

    def render_hq_us_slice(self, pose_idx: int):
        pose_idx = int(np.clip(pose_idx, 0, len(self.poses) - 1))
        frame_idx = self.pose_idx_to_frame_idx(pose_idx)
        img = self.read_us_frame_hq(frame_idx)
        status = (
            f"Pose {pose_idx} | Frame {frame_idx} | {self.us_hq_width}x{self.us_hq_height} | "
            f"video_offset={self.video_time_offset_sec:+.5f}s"
        )
        return img, status

    def sample_oblique_volume_section(
        self,
        center_z: float,
        center_y: float,
        center_x: float,
        rot_x_deg: float,
        rot_y_deg: float,
        rot_z_deg: float,
    ):
        geom = self._oblique_plane_geometry(center_z, center_y, center_x, rot_x_deg, rot_y_deg, rot_z_deg)
        center = geom["center"]
        center_phys_mm = geom["center_phys_mm"]
        basis_u_phys = geom["basis_u_phys"]
        basis_v_phys = geom["basis_v_phys"]
        uu = np.linspace(-self.oblique_half_u_mm, self.oblique_half_u_mm, self.oblique_panel_width, dtype=np.float64)
        vv = np.linspace(-self.oblique_half_v_mm, self.oblique_half_v_mm, self.oblique_panel_height, dtype=np.float64)
        grid_u, grid_v = np.meshgrid(uu, vv, indexing="xy")
        plane_phys_mm = (
            center_phys_mm[None, None, :]
            + grid_u[:, :, None] * basis_u_phys[None, None, :]
            + grid_v[:, :, None] * basis_v_phys[None, None, :]
        )
        plane_zyx = plane_phys_mm / self.spacing_zyx_mm[None, None, :]
        sampled = _sample_trilinear(self.volume, plane_zyx.reshape(-1, 3)).reshape(self.oblique_panel_height, self.oblique_panel_width)
        coverage = float(np.count_nonzero(sampled > 0.0)) / float(sampled.size)
        meta = {
            "center": center,
            "rot": geom["rot"],
            "coverage": coverage,
            "center_world_mm": geom["center_world_mm"],
        }
        return sampled, meta

    def _straight_line_segmentation(self, slice_img):
        if straight_line_planner_module is None:
            raise RuntimeError(f"straight_line_planner import failed: {STRAIGHT_LINE_IMPORT_ERROR}")

        result = straight_line_planner_module.segment_workspace_and_obstacles(
            np.asarray(slice_img, dtype=np.float32),
            outer_threshold=0.03,
            tissue_threshold=0.08,
            dark_threshold=0.18,
            dark_inner_margin=13,
            bright_seed_percentile=99.6,
            bright_edge_percentile=95.0,
            bright_inner_percentile=87.0,
            min_workspace_size=500,
            min_dark_size=6,
            min_bright_size=10,
            hole_size_workspace=200,
            hole_size_dark=20,
            hole_size_bright=30,
            smooth_sigma=0.8,
            dark_close_radius=2,
            dark_dilation_radius=1,
            bright_close_radius=4,
            bright_dilation_radius=1,
            bright_local_dilation_radius=8,
            bright_secondary_close_radius=5,
        )
        return result

    def render_straight_line_planner(
        self,
        center_z: float,
        center_y: float,
        center_x: float,
        rot_x_deg: float,
        rot_y_deg: float,
        rot_z_deg: float,
        click_xy=None,
        plane_points_world_mm=None,
    ):
        if straight_line_planner_module is None:
            blank = np.zeros((480, 480, 3), dtype=np.uint8)
            status = f"Planner unavailable | {STRAIGHT_LINE_IMPORT_ERROR or 'missing dependency'}"
            return blank, status, None

        if plane_points_world_mm is not None:
            sampled, meta = self.sample_three_point_plane_section(plane_points_world_mm)
        else:
            sampled, meta = self.sample_oblique_volume_section(
                center_z,
                center_y,
                center_x,
                rot_x_deg,
                rot_y_deg,
                rot_z_deg,
            )
        segmentation = self._straight_line_segmentation(sampled)
        workspace_mask = segmentation["workspace_mask"]
        obstacle_mask = segmentation["total_obstacle_mask"]
        base_img = straight_line_planner_module.render_planning_result_image(
            background_img=segmentation["slice_norm"],
            workspace_mask=workspace_mask,
            dark_region_mask=segmentation["dark_region_mask"],
            bright_region_mask=segmentation["bright_region_mask"],
        )
        if meta.get("mode") == "three-point":
            center = meta["center_world_mm"]
            normal = meta["normal"]
            prefix = (
                f"3-point plane | center_world_mm=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) | "
                f"normal=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}) | "
                f"coverage {meta['coverage'] * 100.0:.1f}%"
            )
        else:
            center = meta["center"]
            rot = meta["rot"]
            prefix = (
                f"oblique center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) | "
                f"rot=({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}) | "
                f"coverage {meta['coverage'] * 100.0:.1f}%"
            )

        if click_xy is None:
            status = f"{prefix} | click image to choose target"
            return base_img, status, None

        row = int(np.clip(round(click_xy[1]), 0, workspace_mask.shape[0] - 1))
        col = int(np.clip(round(click_xy[0]), 0, workspace_mask.shape[1] - 1))
        target = (row, col)

        status_parts = [prefix, f"target=({target[0]}, {target[1]})"]
        render_kwargs = dict(
            background_img=segmentation["slice_norm"],
            workspace_mask=workspace_mask,
            dark_region_mask=segmentation["dark_region_mask"],
            bright_region_mask=segmentation["bright_region_mask"],
            target=target,
        )

        if not workspace_mask[target]:
            status_parts.append("target outside workspace")
            img = straight_line_planner_module.render_planning_result_image(**render_kwargs)
            return img, " | ".join(status_parts), (target[1], target[0])

        if obstacle_mask[target]:
            status_parts.append("target inside obstacle")
            img = straight_line_planner_module.render_planning_result_image(**render_kwargs)
            return img, " | ".join(status_parts), (target[1], target[0])

        try:
            plan = straight_line_planner_module.plan_path_from_masks(
                workspace_mask=workspace_mask,
                obstacle_mask=obstacle_mask,
                target=target,
                margin=3,
                outside_offset=8,
                n_angles=360,
                inward_start_offset=20,
            )
            img = straight_line_planner_module.render_planning_result_image(
                **render_kwargs,
                start_inside=plan["start_inside"],
                start_outside=plan["start_outside"],
                ray_inside=plan["ray_inside"],
            )
            status_parts.append(f"start=({plan['start_inside'][0]}, {plan['start_inside'][1]})")
            status_parts.append(f"path_len={len(plan['ray_inside'])}")
        except RuntimeError as exc:
            img = straight_line_planner_module.render_planning_result_image(**render_kwargs)
            status_parts.append(str(exc))

        return img, " | ".join(status_parts), (target[1], target[0])

    def render_auto_mri_section(self, pose_idx: int, click_xy=None):
        pose_idx = int(np.clip(pose_idx, 0, len(self.poses) - 1))
        pose = self.poses[pose_idx]
        frame_idx = self.pose_idx_to_frame_idx(pose_idx)
        slice_indices = self.pose_to_slice_indices(pose_idx)

        u_mid = 0.5 * (self.video_width - 1.0)
        v_mid = 0.5 * (self.video_height - 1.0)
        du = min(self.video_width - 1.0, u_mid + max(8.0, 0.05 * self.video_width))
        dv = min(self.video_height - 1.0, v_mid + max(8.0, 0.05 * self.video_height))
        plane_world_mm = self.image_plane_world(
            pose.T_pol_probe,
            u_coords=np.array([u_mid, du], dtype=np.float64),
            v_coords=np.array([v_mid, dv], dtype=np.float64),
        )
        plane_voxel = self.world_to_voxel_batch(plane_world_mm.reshape(-1, 3) * 1e-3).reshape(2, 2, 3)
        center_voxel = plane_voxel[0, 0]
        vec_u = plane_voxel[0, 1] - plane_voxel[0, 0]
        vec_v = plane_voxel[1, 0] - plane_voxel[0, 0]
        normal_voxel = np.cross(vec_u, vec_v)
        normal_norm = float(np.linalg.norm(normal_voxel))
        if normal_norm > 1e-8:
            normal_voxel = normal_voxel / normal_norm

        source_axis_to_panel = {
            cfg["source_axis"]: panel_id
            for panel_id, cfg in self.panel_configs.items()
        }
        dominant_axis = int(np.argmax(np.abs(normal_voxel)))
        panel_id = source_axis_to_panel[dominant_axis]
        slice_idx = int(slice_indices[panel_id])
        img, marker_count = self.render_mri_slice(panel_id, slice_idx, panel_id.title())
        center_coords = ", ".join(f"{v:.1f}" for v in center_voxel)

        if click_xy is None:
            status = (
                f"Pose {pose_idx} | Frame {frame_idx} | MRI auto section: {panel_id} "
                f"| slice {slice_idx} | center_voxel=({center_coords}) | markers {marker_count} "
                f"| click upper image to choose sample"
            )
            return img, _blank_patch(self.auto_section_patch_radius * 2), status, None

        sample_xy = click_xy
        patch_img, marker_xy = _crop_patch(img, sample_xy, patch_radius=self.auto_section_patch_radius)
        section_img = _draw_crosshair(img, marker_xy)
        status = (
            f"Pose {pose_idx} | Frame {frame_idx} | MRI auto section: {panel_id} "
            f"| slice {slice_idx} | sample=({int(round(marker_xy[0]))}, {int(round(marker_xy[1]))}) "
            f"| center_voxel=({center_coords}) | markers {marker_count}"
        )
        return section_img, patch_img, status, marker_xy

    def render_oblique_volume_section(
        self,
        center_z: float,
        center_y: float,
        center_x: float,
        rot_x_deg: float,
        rot_y_deg: float,
        rot_z_deg: float,
        plane_points_world_mm=None,
    ):
        if plane_points_world_mm is not None:
            sampled, meta = self.sample_three_point_plane_section(plane_points_world_mm)
        else:
            sampled, meta = self.sample_oblique_volume_section(
                center_z,
                center_y,
                center_x,
                rot_x_deg,
                rot_y_deg,
                rot_z_deg,
            )
        disp = _auto_window_uint8(sampled, lo_pct=0.5, hi_pct=99.7)
        if meta.get("mode") == "three-point":
            center = meta["center_world_mm"]
            normal = meta["normal"]
            status = (
                f"3-point plane | center_world_mm=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) | "
                f"normal=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}) | "
                f"coverage {meta['coverage'] * 100.0:.1f}%"
            )
        else:
            center = meta["center"]
            rot = meta["rot"]
            status = (
                f"center(z,y,x)=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) | "
                f"rot(x,y,z)=({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}) deg | "
                f"coverage {meta['coverage'] * 100.0:.1f}%"
            )
        return disp, status

    def image_plane_outline_trace(self, T_pol_probe):
        corners = self.image_plane_world(T_pol_probe)
        pts = np.array([
            corners[0, 0],
            corners[0, 1],
            corners[1, 1],
            corners[1, 0],
            corners[0, 0],
        ])
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color="#ef4444", width=10),
            hoverinfo="skip",
        )

    def slice_plane_world_mm(self, axis_name: str, slice_idx: int):
        axis_name = axis_name.lower()
        cfg = self.panel_configs.get(axis_name)
        if cfg is None:
            raise ValueError(f"Unknown axis name: {axis_name}")
        z_max, y_max, x_max = (dim - 1 for dim in self.volume.shape)
        slice_idx = float(np.clip(slice_idx, 0, self.volume.shape[cfg["source_axis"]] - 1))
        if cfg["source_axis"] == 0:
            corners_zyx = np.array([
                [slice_idx, 0.0, 0.0],
                [slice_idx, y_max, 0.0],
                [slice_idx, y_max, x_max],
                [slice_idx, 0.0, x_max],
            ], dtype=np.float64)
        elif cfg["source_axis"] == 1:
            corners_zyx = np.array([
                [0.0, slice_idx, 0.0],
                [z_max, slice_idx, 0.0],
                [z_max, slice_idx, x_max],
                [0.0, slice_idx, x_max],
            ], dtype=np.float64)
        else:
            corners_zyx = np.array([
                [0.0, 0.0, slice_idx],
                [z_max, 0.0, slice_idx],
                [z_max, y_max, slice_idx],
                [0.0, y_max, slice_idx],
            ], dtype=np.float64)
        patient_mm = self.voxel_to_patient_mm_batch(corners_zyx)
        return self.mri_mm_to_world_mm_batch(patient_mm)

    def mri_volume_outline_trace(self):
        z_max, y_max, x_max = (dim - 1 for dim in self.volume.shape)
        corners_zyx = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, x_max],
            [0.0, y_max, 0.0],
            [0.0, y_max, x_max],
            [z_max, 0.0, 0.0],
            [z_max, 0.0, x_max],
            [z_max, y_max, 0.0],
            [z_max, y_max, x_max],
        ], dtype=np.float64)
        pts = self.mri_mm_to_world_mm_batch(self.voxel_to_patient_mm_batch(corners_zyx))
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]
        x_vals = []
        y_vals = []
        z_vals = []
        for i, j in edges:
            x_vals.extend([pts[i, 0], pts[j, 0], None])
            y_vals.extend([pts[i, 1], pts[j, 1], None])
            z_vals.extend([pts[i, 2], pts[j, 2], None])
        return go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines",
            line=dict(color="#475569", width=4),
            opacity=0.8,
            hoverinfo="skip",
        )

    def mri_slice_surface_trace(self, axis_name: str, slice_idx: int):
        pts = self.slice_plane_world_mm(axis_name, slice_idx)
        color = self.panel_configs[axis_name]["color"]
        return go.Surface(
            x=np.array([[pts[0, 0], pts[1, 0]], [pts[3, 0], pts[2, 0]]], dtype=np.float64),
            y=np.array([[pts[0, 1], pts[1, 1]], [pts[3, 1], pts[2, 1]]], dtype=np.float64),
            z=np.array([[pts[0, 2], pts[1, 2]], [pts[3, 2], pts[2, 2]]], dtype=np.float64),
            surfacecolor=np.zeros((2, 2), dtype=np.float32),
            colorscale=[[0.0, color], [1.0, color]],
            showscale=False,
            opacity=0.12,
            hovertemplate=f"{axis_name.title()} slice {int(slice_idx)}<extra></extra>",
        )

    def mri_slice_outline_trace(self, axis_name: str, slice_idx: int):
        pts = self.slice_plane_world_mm(axis_name, slice_idx)
        pts = np.vstack([pts, pts[:1]])
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color=self.panel_configs[axis_name]["color"], width=6),
            hovertemplate=f"{axis_name.title()} slice {int(slice_idx)}<extra></extra>",
        )

    def mri_marker_trace(self):
        pts = np.asarray(self.mri_marker_points_world_mm, dtype=np.float64)
        if pts.size == 0 or not np.all(np.isfinite(pts)):
            return go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip")
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=6, color="#f97316", line=dict(color="#7c2d12", width=1)),
            text=self.mri_marker_hover,
            hovertemplate="%{text}<extra></extra>",
        )

def _render_3d_view(
    pose_idx,
    axial_slice,
    coronal_slice,
    sagittal_slice,
    oblique_center_z,
    oblique_center_y,
    oblique_center_x,
    oblique_rot_x,
    oblique_rot_y,
    oblique_rot_z,
    show_us_probe_items=True,
    show_mri_surface=True,
    show_mri_markers=True,
):
    pose_idx = int(np.clip(pose_idx, 0, len(DATA.poses) - 1))
    pose = DATA.poses[pose_idx]
    frame_idx = DATA.pose_idx_to_frame_idx(pose_idx)
    us_img = DATA.read_us_frame(frame_idx)
    probe_trace = DATA.mesh_trace(DATA.probe_mesh, pose.T_pol_probe, model_to_local=DATA.probe_model_in_probe)
    needle_trace = DATA.needle_line_trace(pose.T_pol_probe)
    plane_surface = DATA.image_plane_surface_trace(pose.T_pol_probe, us_img)
    axial_surface = DATA.mri_slice_surface_trace("axial", axial_slice)
    axial_outline = DATA.mri_slice_outline_trace("axial", axial_slice)
    coronal_surface = DATA.mri_slice_surface_trace("coronal", coronal_slice)
    coronal_outline = DATA.mri_slice_outline_trace("coronal", coronal_slice)
    sagittal_surface = DATA.mri_slice_surface_trace("sagittal", sagittal_slice)
    sagittal_outline = DATA.mri_slice_outline_trace("sagittal", sagittal_slice)
    oblique_surface = DATA.oblique_plane_surface_trace(
        oblique_center_z,
        oblique_center_y,
        oblique_center_x,
        oblique_rot_x,
        oblique_rot_y,
        oblique_rot_z,
    )
    oblique_outline = DATA.oblique_plane_outline_trace(
        oblique_center_z,
        oblique_center_y,
        oblique_center_x,
        oblique_rot_x,
        oblique_rot_y,
        oblique_rot_z,
    )
    fig_traces = [
        DATA.mri_volume_outline_trace(),
        axial_surface,
        coronal_surface,
        sagittal_surface,
        axial_outline,
        coronal_outline,
        sagittal_outline,
        oblique_surface,
        oblique_outline,
    ]
    if show_mri_markers:
        fig_traces.append(DATA.mri_marker_3d_trace)
    if show_mri_surface:
        fig_traces.append(DATA.mri_surface_trace())
    if show_us_probe_items:
        fig_traces.extend([
            probe_trace,
            needle_trace,
            plane_surface,
        ])
    fig = go.Figure(data=fig_traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            bgcolor="#ffffff",
            xaxis=dict(title="X", backgroundcolor="#ffffff", gridcolor="#d9d9d9"),
            yaxis=dict(title="Y", backgroundcolor="#ffffff", gridcolor="#d9d9d9"),
            zaxis=dict(title="Z", backgroundcolor="#ffffff", gridcolor="#d9d9d9"),
        ),
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        uirevision="view-3d-static",
    )
    msg = (
        f"pose_idx={pose_idx} | pose_time_ns={pose.t_ns} | "
        f"matched_frame={frame_idx} | frame_time_ns={DATA.frame_time_ns(frame_idx)} | "
        f"video_offset={DATA.video_time_offset_sec:+.5f}s | "
        f"mri_g_offset={DATA.mri_g_offset_mm:+.1f}mm | "
        f"oblique_center=({float(oblique_center_z):.1f}, {float(oblique_center_y):.1f}, {float(oblique_center_x):.1f}) | "
        f"oblique_rot=({float(oblique_rot_x):.1f}, {float(oblique_rot_y):.1f}, {float(oblique_rot_z):.1f}) | "
        f"mri_markers={len(DATA.mri_marker_names)}"
    )
    return fig, msg


DATA = AppData()
INITIAL_POSE_SLICES = DATA.pose_to_slice_indices(0)
DATA.slice_default.update(INITIAL_POSE_SLICES)
INITIAL_AXIAL_IMG, INITIAL_AXIAL_MARKERS = DATA.render_mri_slice("axial", DATA.slice_default["axial"], DATA.panel_configs["axial"]["title"])
INITIAL_CORONAL_IMG, INITIAL_CORONAL_MARKERS = DATA.render_mri_slice("coronal", DATA.slice_default["coronal"], DATA.panel_configs["coronal"]["title"])
INITIAL_SAGITTAL_IMG, INITIAL_SAGITTAL_MARKERS = DATA.render_mri_slice("sagittal", DATA.slice_default["sagittal"], DATA.panel_configs["sagittal"]["title"])
INITIAL_AXIAL = _b64_img(INITIAL_AXIAL_IMG)
INITIAL_CORONAL = _b64_img(INITIAL_CORONAL_IMG)
INITIAL_SAGITTAL = _b64_img(INITIAL_SAGITTAL_IMG)
INITIAL_RESAMPLE_IMG, INITIAL_RESAMPLE_STATUS = DATA.render_probe_plane_mri(0)
INITIAL_RESAMPLE = _b64_img(INITIAL_RESAMPLE_IMG)
INITIAL_US_HQ_IMG, INITIAL_US_HQ_STATUS = DATA.render_hq_us_slice(0)
INITIAL_US_HQ = _b64_img(INITIAL_US_HQ_IMG)
INITIAL_STRAIGHT_LINE_IMG, INITIAL_STRAIGHT_LINE_STATUS, INITIAL_STRAIGHT_LINE_MARKER = DATA.render_straight_line_planner(
    DATA.oblique_center_default["z"],
    DATA.oblique_center_default["y"],
    DATA.oblique_center_default["x"],
    0.0,
    0.0,
    0.0,
)
INITIAL_STRAIGHT_LINE = _b64_img(INITIAL_STRAIGHT_LINE_IMG)
INITIAL_OBLIQUE_IMG, INITIAL_OBLIQUE_STATUS = DATA.render_oblique_volume_section(
    DATA.oblique_center_default["z"],
    DATA.oblique_center_default["y"],
    DATA.oblique_center_default["x"],
    0.0,
    0.0,
    0.0,
)
INITIAL_OBLIQUE = _b64_img(INITIAL_OBLIQUE_IMG)
INITIAL_FIG, INITIAL_MSG = _render_3d_view(
    0,
    DATA.slice_default["axial"],
    DATA.slice_default["coronal"],
    DATA.slice_default["sagittal"],
    DATA.oblique_center_default["z"],
    DATA.oblique_center_default["y"],
    DATA.oblique_center_default["x"],
    0.0,
    0.0,
    0.0,
)

_PANEL_STYLE = {
    "background": "#050505",
    "border": "1px solid #1d1d1d",
    "borderRadius": "0px",
    "overflow": "hidden",
    "display": "flex",
    "flexDirection": "column",
    "minHeight": "0",
}

_MRI_IMG_STYLE = {
    "width": "100%",
    "height": "100%",
    "display": "block",
    "objectFit": "contain",
    "background": "#000",
    "flex": "1 1 auto",
    "minHeight": "0",
}

_COMPACT_PANEL_IMAGE_STYLE = {
    "display": "block",
    "width": "88%",
    "height": "88%",
    "objectFit": "contain",
    "margin": "auto",
    "background": "#000",
}

_COMPACT_PANEL_IMAGE_WRAP_STYLE = {
    "flex": "1 1 auto",
    "minHeight": "0",
    "background": "#000",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "padding": "16px",
}

_CONTROL_BLOCK_STYLE = {
    "padding": "4px 10px 8px",
    "background": "#111",
    "color": "#ddd",
    "flex": "0 0 auto",
}

_PLAY_BUTTON_STYLE = {
    "padding": "8px 16px",
    "border": "0",
    "background": "#2563eb",
    "color": "#fff",
    "fontWeight": "700",
    "cursor": "pointer",
    "borderRadius": "999px",
}

_CLICK_COORD_STYLE = {
    "padding": "6px 10px",
    "background": "#fef08a",
    "color": "#111827",
    "fontSize": "13px",
    "fontWeight": "700",
    "borderTop": "1px solid #1f2937",
    "borderBottom": "1px solid #1f2937",
}

def _slice_status_text(slice_idx: int, marker_count: int):
    suffix = "marker" if marker_count == 1 else "markers"
    return f"Slice {int(slice_idx)} | {marker_count} {suffix}"


def _sync_offset_sec_from_mode(mode_value: str):
    mode_value = str(mode_value or "none")
    offset_sec = float(getattr(DATA, "tracker_local_time_offset_sec", 0.0))
    if mode_value == "plus":
        return offset_sec
    if mode_value == "minus":
        return -offset_sec
    return 0.0


def _apply_runtime_offsets(video_sync_mode=None):
    if video_sync_mode is not None:
        DATA.video_time_offset_sec = _sync_offset_sec_from_mode(video_sync_mode)


def _timeline_ids(prefix: str):
    if not prefix:
        return {
            "interval": "play-interval",
            "toggle": "play-toggle",
            "slider": "pose-slider",
            "frame_info": "frame-info",
            "sync_mode": "video-sync-mode",
        }
    return {
        "interval": f"{prefix}-play-interval",
        "toggle": f"{prefix}-play-toggle",
        "slider": f"{prefix}-pose-slider",
        "frame_info": f"{prefix}-frame-info",
        "sync_mode": f"{prefix}-video-sync-mode",
    }


def _timeline_controls(prefix: str, initial_msg: str):
    ids = _timeline_ids(prefix)
    return html.Div([
        dcc.Interval(id=ids["interval"], interval=DATA.play_interval_ms, n_intervals=0, disabled=True),
        html.Div([
            html.Div([
                html.Div("Probe Pose Timeline", style={"fontWeight": "700", "marginBottom": "6px"}),
                html.Div([
                    html.Button("Play", id=ids["toggle"], n_clicks=0, style=_PLAY_BUTTON_STYLE),
                    html.Div(
                        f"{DATA.play_interval_ms} ms / step",
                        style={"fontSize": "12px", "color": "#4b5563"},
                    ),
                    dcc.Dropdown(
                        id=ids["sync_mode"],
                        options=[
                            {"label": "Video Sync: None", "value": "none"},
                            {"label": f"Video Sync: +{DATA.tracker_local_time_offset_sec:.5f}s", "value": "plus"},
                            {"label": f"Video Sync: -{DATA.tracker_local_time_offset_sec:.5f}s", "value": "minus"},
                        ],
                        value="none",
                        clearable=False,
                        searchable=False,
                        style={"width": "210px", "fontSize": "12px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
            dcc.Slider(
                0,
                len(DATA.poses) - 1,
                1,
                value=0,
                id=ids["slider"],
                updatemode="drag",
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(id=ids["frame_info"], children=initial_msg, style={"marginTop": "8px", "fontSize": "14px"}),
        ], style={"padding": "12px 16px", "background": "#f3f4f6", "borderBottom": "1px solid #d1d5db"}),
    ])


def _oblique_ids():
    return {
        "center_z": "oblique-center-z",
        "center_y": "oblique-center-y",
        "center_x": "oblique-center-x",
        "rot_x": "oblique-rot-x",
        "rot_y": "oblique-rot-y",
        "rot_z": "oblique-rot-z",
    }


OBLIQUE_IDS = _oblique_ids()


def _oblique_slider_control(label: str, slider_id: str, min_value: float, max_value: float, value: float, step: float = 1.0):
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "marginBottom": "2px"}),
        dcc.Slider(
            min_value,
            max_value,
            step,
            value=value,
            id=slider_id,
            marks=None,
            updatemode="drag",
            tooltip={"placement": "top", "always_visible": False},
        ),
    ], style={"minWidth": "0"})


def _oblique_controls():
    ids = OBLIQUE_IDS
    return html.Div([
        html.Div(
            "Adjust the arbitrary section with these sliders. The same plane is shown in the 3D view.",
            style={"fontSize": "12px", "lineHeight": "1.4", "marginBottom": "8px"},
        ),
        html.Div([
            _oblique_slider_control("Center Z", ids["center_z"], 0, DATA.volume.shape[0] - 1, DATA.oblique_center_default["z"]),
            _oblique_slider_control("Center Y", ids["center_y"], 0, DATA.volume.shape[1] - 1, DATA.oblique_center_default["y"]),
            _oblique_slider_control("Center X", ids["center_x"], 0, DATA.volume.shape[2] - 1, DATA.oblique_center_default["x"]),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "10px"}),
        html.Div([
            _oblique_slider_control("Rotate X", ids["rot_x"], -180, 180, 0),
            _oblique_slider_control("Rotate Y", ids["rot_y"], -180, 180, 0),
            _oblique_slider_control("Rotate Z", ids["rot_z"], -180, 180, 0),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "10px", "marginTop": "8px"}),
    ], style=_CONTROL_BLOCK_STYLE)


def _mri_panel(panel_id: str, title: str, color: str, img_src: str, slice_max: int, slice_default: int, status_text: str):
    return html.Div([
        html.Div([
            html.Span(title),
            html.Span(id=f"{panel_id}-status", children=status_text),
        ], style=_slice_header_style(color)),
        html.Div([
            html.Div("Slice", style={"fontSize": "12px", "marginBottom": "2px"}),
            dcc.Slider(
                0,
                slice_max,
                1,
                value=slice_default,
                id=f"{panel_id}-slice",
                marks=None,
                tooltip={"placement": "top", "always_visible": False},
            ),
        ], style=_CONTROL_BLOCK_STYLE),
        html.Div(
            html.Img(id=f"{panel_id}-img", src=img_src, style=_MRI_IMG_STYLE),
            style={"flex": "1 1 auto", "minHeight": "0", "background": "#000"},
        ),
    ], style={**_PANEL_STYLE, "height": "100%"})


def _resample_panel(img_src: str, status_text: str):
    return html.Div([
        html.Div([
            html.Span("MRI Probe Plane Resample"),
            html.Span(id="probe-resample-status", children=status_text),
        ], style=_slice_header_style("#38bdf8")),
        html.Div(
            html.Img(id="probe-resample-img", src=img_src, style=_MRI_IMG_STYLE),
            style={"flex": "1 1 auto", "minHeight": "0", "background": "#000"},
        ),
    ], style={**_PANEL_STYLE, "height": "100%"})


def _us_hq_panel(img_src: str, status_text: str):
    return html.Div([
        html.Div([
            html.Span("High-Quality US Slice"),
            html.Span(id="us-hq-status", children=status_text),
        ], style=_slice_header_style("#f59e0b")),
        html.Div(
            html.Img(id="us-hq-img", src=img_src, style=_MRI_IMG_STYLE),
            style={"flex": "1 1 auto", "minHeight": "0", "background": "#000"},
        ),
    ], style={**_PANEL_STYLE, "height": "100%"})


def _clickable_image_panel(img_id: str, overlay_id: str, img_src: str):
    return html.Div([
        html.Img(
            id=img_id,
            src=img_src,
            style=_COMPACT_PANEL_IMAGE_STYLE,
        ),
        html.Div(
            id=overlay_id,
            style={
                "position": "absolute",
                "inset": "0",
                "cursor": "crosshair",
                "background": "rgba(0,0,0,0.001)",
            },
        ),
    ], style={
        "position": "relative",
        "flex": "1 1 auto",
        "minHeight": "0",
        "background": "#000",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "16px",
    })


def _straight_line_panel(img_src: str, status_text: str):
    return html.Div([
        html.Div([
            html.Span("Straight Line Planner"),
            html.Span(id="straight-line-status", children=status_text),
        ], style=_slice_header_style("#a78bfa")),
        html.Div(
            "(3,1) directly uses the current resample shown in (3,2). Adjust the section on the right, then click here to place target.",
            style={**_CONTROL_BLOCK_STYLE, "fontSize": "12px", "lineHeight": "1.4"},
        ),
        html.Div(id="straight-line-click-text", children="Clicked target: waiting for click", style=_CLICK_COORD_STYLE),
        html.Div(
            _clickable_image_panel("straight-line-img", "straight-line-overlay", img_src),
            style={"flex": "1 1 auto", "minHeight": "0", "background": "#000"},
        ),
    ], style={**_PANEL_STYLE, "height": "100%"})
def _volume_sampler_panel(img_src: str, status_text: str):
    return html.Div([
        html.Div([
            html.Span("3D Volume Arbitrary Section"),
            html.Span(id="volume-sampler-status", children=status_text),
        ], style=_slice_header_style("#22c55e")),
        _oblique_controls(),
        html.Div([
            html.Div(
                html.Img(id="volume-sampler-img", src=img_src, style=_COMPACT_PANEL_IMAGE_STYLE),
                style=_COMPACT_PANEL_IMAGE_WRAP_STYLE,
            ),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "minmax(0, 1fr)",
            "gap": "1px",
            "background": "#1d1d1d",
            "flex": "1 1 auto",
            "minHeight": "0",
        }),
    ], style={**_PANEL_STYLE, "height": "100%"})


def _main_page_layout():
    return html.Div([
        dcc.Store(id="straight-line-target-store"),
        dcc.Store(id="view-3d-visibility", data={"show_us_probe": True, "show_mri_surface": True, "show_mri_markers": True}),
        _timeline_controls("", INITIAL_MSG),
        html.Div([
            _mri_panel("axial", DATA.panel_configs["axial"]["title"], DATA.panel_configs["axial"]["color"], INITIAL_AXIAL, DATA.slice_max["axial"], DATA.slice_default["axial"], _slice_status_text(DATA.slice_default["axial"], INITIAL_AXIAL_MARKERS)),
            html.Div([
                html.Div([
                    html.Span("3D"),
                    html.Span("Probe / MRI / Needle"),
                ], style=_slice_header_style("#8b93ff")),
                html.Div([
                    dcc.Checklist(
                        id="view-3d-visibility-toggle",
                        options=[
                            {"label": "Show US / Probe", "value": "show_us_probe"},
                            {"label": "Show MRI Surface", "value": "show_mri_surface"},
                            {"label": "Show MRI Markers", "value": "show_mri_markers"},
                        ],
                        value=["show_us_probe", "show_mri_surface", "show_mri_markers"],
                        inline=True,
                        style={"fontSize": "13px", "color": "#111827"},
                        inputStyle={"marginRight": "6px", "marginLeft": "10px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px", "padding": "8px 12px", "background": "#f3f4f6", "borderBottom": "1px solid #d1d5db", "flexWrap": "wrap"}),
                dcc.Graph(id="view-3d", figure=INITIAL_FIG, style={"height": "calc(100% - 92px)"}),
            ], style={**_PANEL_STYLE, "height": "100%", "background": "#ffffff"}),
            html.Div(
                _resample_panel(INITIAL_RESAMPLE, INITIAL_RESAMPLE_STATUS),
                style={"gridColumn": "3", "gridRow": "1", "minHeight": "0"},
            ),
            _mri_panel("coronal", DATA.panel_configs["coronal"]["title"], DATA.panel_configs["coronal"]["color"], INITIAL_CORONAL, DATA.slice_max["coronal"], DATA.slice_default["coronal"], _slice_status_text(DATA.slice_default["coronal"], INITIAL_CORONAL_MARKERS)),
            _mri_panel("sagittal", DATA.panel_configs["sagittal"]["title"], DATA.panel_configs["sagittal"]["color"], INITIAL_SAGITTAL, DATA.slice_max["sagittal"], DATA.slice_default["sagittal"], _slice_status_text(DATA.slice_default["sagittal"], INITIAL_SAGITTAL_MARKERS)),
            _us_hq_panel(INITIAL_US_HQ, INITIAL_US_HQ_STATUS),
            html.Div(
                html.Div([
                    _straight_line_panel(INITIAL_STRAIGHT_LINE, INITIAL_STRAIGHT_LINE_STATUS),
                    _volume_sampler_panel(INITIAL_OBLIQUE, INITIAL_OBLIQUE_STATUS),
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(560px, 1.2fr) minmax(620px, 1.35fr)",
                    "gridTemplateRows": "780px",
                    "gap": "1px",
                    "background": "#d1d5db",
                    "height": "100%",
                    "minHeight": "0",
                }),
                style={"gridColumn": "1 / 4", "gridRow": "3", "minHeight": "0"},
            ),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr) minmax(320px, 0.95fr)",
            "gridTemplateRows": "520px 520px 780px",
            "gap": "1px",
            "background": "#d1d5db",
            "paddingBottom": "1px",
            "alignContent": "start",
        }),
    ], style={
        "minHeight": "100vh",
        "background": "#111827",
        "margin": "0",
        "overflowY": "auto",
    })


app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.server.wsgi_app = ProxyFix(app.server.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.layout = _main_page_layout()


@app.callback(
    Output("pose-slider", "value"),
    Output("play-interval", "disabled"),
    Output("play-toggle", "children"),
    Input("play-toggle", "n_clicks"),
    Input("play-interval", "n_intervals"),
    State("pose-slider", "value"),
    State("play-interval", "disabled"),
)
def handle_playback(play_clicks, n_intervals, pose_idx, is_disabled):
    del play_clicks, n_intervals
    pose_idx = int(0 if pose_idx is None else pose_idx)
    is_disabled = bool(is_disabled)
    triggered = dash.ctx.triggered_id
    max_pose_idx = len(DATA.poses) - 1

    if triggered == "play-toggle":
        if is_disabled:
            if pose_idx >= max_pose_idx:
                pose_idx = 0
            return pose_idx, False, "Pause"
        return pose_idx, True, "Play"

    if triggered == "play-interval":
        next_idx = min(pose_idx + 1, max_pose_idx)
        if next_idx >= max_pose_idx:
            return next_idx, True, "Play"
        return next_idx, False, "Pause"

    return pose_idx, True, "Play"


@app.callback(
    Output("view-3d-visibility", "data"),
    Input("view-3d-visibility-toggle", "value"),
)
def update_3d_visibility(toggle_values):
    values = set(toggle_values or [])
    return {
        "show_us_probe": "show_us_probe" in values,
        "show_mri_surface": "show_mri_surface" in values,
        "show_mri_markers": "show_mri_markers" in values,
    }


@app.callback(
    Output("axial-slice", "value"),
    Output("coronal-slice", "value"),
    Output("sagittal-slice", "value"),
    Input("pose-slider", "value"),
)
def sync_slices_to_pose(pose_idx):
    slice_indices = DATA.pose_to_slice_indices(pose_idx)
    return (
        slice_indices["axial"],
        slice_indices["coronal"],
        slice_indices["sagittal"],
    )


@app.callback(
    Output("view-3d", "figure"),
    Output("frame-info", "children"),
    Input("pose-slider", "value"),
    Input("video-sync-mode", "value"),
    Input("axial-slice", "value"),
    Input("coronal-slice", "value"),
    Input("sagittal-slice", "value"),
    Input(OBLIQUE_IDS["center_z"], "value"),
    Input(OBLIQUE_IDS["center_y"], "value"),
    Input(OBLIQUE_IDS["center_x"], "value"),
    Input(OBLIQUE_IDS["rot_x"], "value"),
    Input(OBLIQUE_IDS["rot_y"], "value"),
    Input(OBLIQUE_IDS["rot_z"], "value"),
    Input("view-3d-visibility", "data"),
)
def update_3d_view(
    pose_idx,
    video_sync_mode,
    axial_slice,
    coronal_slice,
    sagittal_slice,
    oblique_center_z,
    oblique_center_y,
    oblique_center_x,
    oblique_rot_x,
    oblique_rot_y,
    oblique_rot_z,
    visibility_cfg,
):
    _apply_runtime_offsets(video_sync_mode=video_sync_mode)
    visibility_cfg = visibility_cfg or {}
    fig, msg = _render_3d_view(
        pose_idx,
        axial_slice,
        coronal_slice,
        sagittal_slice,
        oblique_center_z,
        oblique_center_y,
        oblique_center_x,
        oblique_rot_x,
        oblique_rot_y,
        oblique_rot_z,
        show_us_probe_items=bool(visibility_cfg.get("show_us_probe", True)),
        show_mri_surface=bool(visibility_cfg.get("show_mri_surface", True)),
        show_mri_markers=bool(visibility_cfg.get("show_mri_markers", True)),
    )
    return fig, msg


@app.callback(
    Output("probe-resample-img", "src"),
    Output("probe-resample-status", "children"),
    Input("pose-slider", "value"),
)
def update_probe_resample(pose_idx):
    img_arr, status = DATA.render_probe_plane_mri(pose_idx)
    return _b64_img(img_arr), status


@app.callback(
    Output("us-hq-img", "src"),
    Output("us-hq-status", "children"),
    Input("pose-slider", "value"),
    Input("video-sync-mode", "value"),
)
def update_us_hq(pose_idx, video_sync_mode):
    _apply_runtime_offsets(video_sync_mode=video_sync_mode)
    img_arr, status = DATA.render_hq_us_slice(pose_idx)
    return _b64_img(img_arr), status


@app.callback(
    Output("straight-line-target-store", "data"),
    Input(OBLIQUE_IDS["center_z"], "value"),
    Input(OBLIQUE_IDS["center_y"], "value"),
    Input(OBLIQUE_IDS["center_x"], "value"),
    Input(OBLIQUE_IDS["rot_x"], "value"),
    Input(OBLIQUE_IDS["rot_y"], "value"),
    Input(OBLIQUE_IDS["rot_z"], "value"),
)
def reset_straight_line_target(center_z, center_y, center_x, rot_x, rot_y, rot_z):
    del center_z, center_y, center_x, rot_x, rot_y, rot_z
    return None


@app.callback(
    Output("straight-line-img", "src"),
    Output("straight-line-status", "children"),
    Output("straight-line-click-text", "children"),
    Input("straight-line-target-store", "data"),
    Input(OBLIQUE_IDS["center_z"], "value"),
    Input(OBLIQUE_IDS["center_y"], "value"),
    Input(OBLIQUE_IDS["center_x"], "value"),
    Input(OBLIQUE_IDS["rot_x"], "value"),
    Input(OBLIQUE_IDS["rot_y"], "value"),
    Input(OBLIQUE_IDS["rot_z"], "value"),
)
def update_straight_line_panel(target_data, center_z, center_y, center_x, rot_x, rot_y, rot_z):
    click_xy = None
    click_text = "Clicked target: waiting for click"
    if target_data and "x" in target_data and "y" in target_data:
        click_xy = (target_data["x"], target_data["y"])
        click_text = f"Clicked target: x={click_xy[0]:.1f}, y={click_xy[1]:.1f}"
    img_arr, status, marker_xy = DATA.render_straight_line_planner(
        center_z,
        center_y,
        center_x,
        rot_x,
        rot_y,
        rot_z,
        click_xy=click_xy,
    )
    del marker_xy
    return _b64_img(img_arr), status, click_text


@app.callback(
    Output("volume-sampler-img", "src"),
    Output("volume-sampler-status", "children"),
    Input(OBLIQUE_IDS["center_z"], "value"),
    Input(OBLIQUE_IDS["center_y"], "value"),
    Input(OBLIQUE_IDS["center_x"], "value"),
    Input(OBLIQUE_IDS["rot_x"], "value"),
    Input(OBLIQUE_IDS["rot_y"], "value"),
    Input(OBLIQUE_IDS["rot_z"], "value"),
)
def update_volume_sampler(center_z, center_y, center_x, rot_x, rot_y, rot_z):
    img_arr, status = DATA.render_oblique_volume_section(
        center_z,
        center_y,
        center_x,
        rot_x,
        rot_y,
        rot_z,
    )
    return _b64_img(img_arr), status


@app.callback(
    Output("axial-img", "src"),
    Output("axial-status", "children"),
    Input("axial-slice", "value"),
)
def update_axial_slice(axial_slice):
    axial_img_arr, axial_markers = DATA.render_mri_slice("axial", axial_slice, DATA.panel_configs["axial"]["title"])
    axial_img = _b64_img(axial_img_arr)
    axial_status = _slice_status_text(axial_slice, axial_markers)
    return axial_img, axial_status


@app.callback(
    Output("coronal-img", "src"),
    Output("coronal-status", "children"),
    Input("coronal-slice", "value"),
)
def update_coronal_slice(coronal_slice):
    coronal_img_arr, coronal_markers = DATA.render_mri_slice("coronal", coronal_slice, DATA.panel_configs["coronal"]["title"])
    coronal_img = _b64_img(coronal_img_arr)
    coronal_status = _slice_status_text(coronal_slice, coronal_markers)
    return coronal_img, coronal_status


@app.callback(
    Output("sagittal-img", "src"),
    Output("sagittal-status", "children"),
    Input("sagittal-slice", "value"),
)
def update_sagittal_slice(sagittal_slice):
    sagittal_img_arr, sagittal_markers = DATA.render_mri_slice("sagittal", sagittal_slice, DATA.panel_configs["sagittal"]["title"])
    sagittal_img = _b64_img(sagittal_img_arr)
    sagittal_status = _slice_status_text(sagittal_slice, sagittal_markers)
    return sagittal_img, sagittal_status


def _resolve_host():
    return os.environ.get("UI_HOST") or os.environ.get("HOST") or "127.0.0.1"


def _resolve_port():
    return int(os.environ.get("UI_PORT") or os.environ.get("PORT") or "8050")


def _resolve_debug():
    return os.environ.get("UI_DEBUG", "").lower() in {"1", "true", "yes"}


def _display_host(host: str):
    if host == "0.0.0.0":
        return "localhost"
    if host == "::":
        return socket.gethostname() or "localhost"
    return host


if __name__ == "__main__":
    host = _resolve_host()
    port = _resolve_port()
    debug = _resolve_debug()
    print(f"UI data root: {DATA_ROOT}")
    print(f"Open http://{_display_host(host)}:{port} in your browser")
    app.run(host=host, port=port, debug=debug)
