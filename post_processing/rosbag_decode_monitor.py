#!/usr/bin/env python3
"""
Decode ROS 2 topics (listed in metadata.yaml) from MCAP files without needing a ROS installation.

Behavior:
  * Only logs/prints when actually decoding a rosbag.
  * No messages/logs during scanning, sleeping, or when skipping (e.g., output exists).
  * Logs are appended to <output_dir>/decode.log, flushed per message during decode.

Features:
  * --input-dir: root containing multiple rosbag folders (each with metadata.yaml and *.mcap / *.mcap.zstd).
  * Scans all rosbag folders; decodes if corresponding output folder under output-dir does NOT exist,
    or overwrite is requested.
  * --monitoring-int (seconds): <=0 scans once and exits; >0 loops at interval to decode newly appearing rosbags.
  * Output folder name: <bag_name>_{TASK_LABEL}_{TASK_OUTCOME}
    - TASK_LABEL from task_info.task_label (or label)
    - TASK_OUTCOME inferred from task_info_collection_states (success/failure/recovery, else unknown)
  * Non-video topics -> NDJSON (+ messages_info.json)
  * Video topics (/vega_vt/image_raw/compressed, /image_raw/compressed,
    /camera/camera/color/image_raw/compressed,
    /camera/camera/depth/image_rect_raw/compressedDepth,
    /visualize/us_imaging/compressed, /visualize/us_imaging_sync/compressed)
    -> MP4 (+ video_info.json)
  * Handles sensor_msgs/msg/Image and sensor_msgs/msg/CompressedImage, including Zstd payloads
    with width/height/encoding in CompressedImage.format.

Outputs per bag: <output_dir>/<bag_name>_{TASK_LABEL}_{TASK_OUTCOME}/
  - <topic>/messages.ndjson
  - <topic>/messages_info.json
  - <topic>/video.mp4
  - <topic>/video_info.json
  - rosbag_info.json
Global log: <output_dir>/decode.log (append-only)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import re
import atexit
from array import array
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import zstandard as zstd  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import numpy as tnp


VIDEO_TOPICS: Set[str] = {
    "/vega_vt/image_raw/compressed",
    "/image_raw/compressed",
    "/camera/camera/color/image_raw/compressed",
    "/camera/camera/depth/image_rect_raw/compressedDepth",
    "/visualize/us_imaging/compressed",
    "/visualize/us_imaging_sync/compressed",
    "/zed/zed_node/depth/depth_registered/compressedDepth",
    "/zed/zed_node/rgb/color/rect/image/compressed"
}
POINTCLOUD_TOPICS: Set[str] = {
    "/zed/zed_node/point_cloud/cloud_registered",
}

KNOWN_ENCODINGS = {
    "bgr8",
    "rgb8",
    "bgra8",
    "rgba8",
    "mono8",
    "8uc1",
    "mono16",
    "16uc1",
    "y16",
}

DEPTH_NORM_MIN: Optional[float] = 0
DEPTH_NORM_MAX: Optional[float] = 65535  # set to None to auto-scale per image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
MAGIC_SIGNATURES = (PNG_SIGNATURE, JPEG_SIGNATURE)


class BagLogger:
    """Simple logger writing to stdout/stderr and appending to a log file (flushed every call)."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _write(self, level: str, msg: str, to_stderr: bool = False) -> None:
        line = f"[{level}] {msg}"
        with self._lock:
            if to_stderr:
                print(line, file=sys.stderr)
            else:
                print(line)
            self.log_file.write(line + "\n")
            self.log_file.flush()

    def info(self, msg: str) -> None:
        self._write("INFO", msg, to_stderr=False)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg, to_stderr=True)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg, to_stderr=True)

    def close(self) -> None:
        try:
            self.log_file.close()
        except Exception:
            pass


class _PseudoImage:
    __slots__ = ("height", "width", "encoding", "step", "data", "is_bigendian")

    def __init__(self, height: int, width: int, encoding: str, step: int, data: bytes, bigendian: bool = False):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step
        self.data = data
        self.is_bigendian = 1 if bigendian else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode ROS 2 MCAP topics without a ROS environment.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Root dir containing rosbag folders.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output root dir; one subfolder per bag.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to decode bags in parallel.")
    parser.add_argument("--topics", nargs="*", help="Optional subset of topics to decode.")
    parser.add_argument(
        "--monitoring-int",
        type=float,
        default=0.0,
        help="Scan interval in seconds. <=0 scans once and exits; >0 loops to decode new bags.",
    )
    return parser.parse_args()


def load_topic_specs(metadata_path: Path) -> Dict[str, str]:
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    info = meta.get("rosbag2_bagfile_information")
    if not info:
        raise KeyError(f"rosbag2_bagfile_information missing in {metadata_path}")
    topics_section = info.get("topics_with_message_count", [])
    topic_to_type: Dict[str, str] = {}
    for entry in topics_section:
        topic_md = entry["topic_metadata"]
        topic_to_type[topic_md["name"]] = topic_md["type"]
    if not topic_to_type:
        raise ValueError(f"No topics listed in {metadata_path}")
    return topic_to_type


def sanitize_topic_name(topic: str) -> str:
    stripped = topic.strip("/")
    if not stripped:
        stripped = "root"
    return stripped.replace("/", "__")


def sanitize_label(value: str) -> str:
    v = value.strip().lower()
    # Keep only a–z and 0–9; drop everything else (spaces, dots, punctuation)
    v = re.sub(r"[^a-z0-9]+", "", v)
    return v or "unknown"


def sanitize_outcome(value: str) -> str:
    v = value.strip().lower()
    if v in {"success", "failure", "recovery"}:
        return v
    return "unknown"


def _extract_task_label_from_msg(msg: Any) -> Tuple[Optional[str], bool]:
    """
    Extract task label and whether it was sourced from task_label_FORCE.
      - objects with .task_label_FORCE, .task_label or .label (case-insensitive)
      - dicts with keys "task_label_FORCE", "task_label" / "label" (case-insensitive)
      - std_msgs/String-like objects where .data is a JSON string containing {"task_label_FORCE": "..."} or {"task_label": "..."}.
    Returns (label, is_force)
    """
    if msg is None:
        return None, False

    def _pick_label(obj: Any) -> Tuple[Optional[str], bool]:
        if isinstance(obj, dict):
            lower_map = {str(k).lower(): v for k, v in obj.items()}
            for key in ("task_label_force", "task_label", "label"):
                val = lower_map.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip(), key == "task_label_force"
        return None, False

    def _regex_extract_label(text: str) -> Tuple[Optional[str], bool]:
        m = re.search(r'"?(task_label_force|task_label)"?\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if m:
            return m.group(2).strip(), m.group(1).lower() == "task_label_force"
        return None, False

    def _from_data_field(raw: Any) -> Optional[str]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        s = raw.strip()
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            picked = _pick_label(parsed)
            if picked:
                return picked
            nested = parsed.get("data")
            if isinstance(nested, str) and nested.strip():
                try:
                    parsed2 = json.loads(nested)
                except Exception:
                    parsed2 = None
                if isinstance(parsed2, dict):
                    picked2 = _pick_label(parsed2)
                    if picked2:
                        return picked2
        regex_lbl = _regex_extract_label(s)
        if regex_lbl:
            return regex_lbl
        return None

    # 1) Direct attributes on decoded object (prefer FORCE), case-insensitive
    for attr in ("task_label_FORCE", "task_label", "label"):
        for candidate in (attr, attr.lower()):
            if hasattr(msg, candidate):
                val = getattr(msg, candidate)
                if isinstance(val, str) and val.strip():
                    return val.strip(), attr.lower() == "task_label_force"

    # 2) Convert to jsonable (dict/primitive) and check directly
    try:
        obj = to_jsonable(msg)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        direct, is_force = _pick_label(obj)
        if direct:
            return direct, is_force

        inner = obj.get("data")
        lbl, is_force = _from_data_field(inner)
        if lbl:
            return lbl, is_force
    else:
        if hasattr(msg, "data"):
            lbl, is_force = _from_data_field(getattr(msg, "data"))
            if lbl:
                return lbl, is_force
        text = str(msg).strip()
        if text:
            regex_lbl, is_force = _regex_extract_label(text)
            if regex_lbl:
                return regex_lbl, is_force
    return None, False


def _extract_task_outcome_from_msg(msg: Any) -> Optional[str]:
    text = ""
    try:
        obj = to_jsonable(msg)
        text = json.dumps(obj, ensure_ascii=False).lower()
    except Exception:
        text = str(msg).lower()
    if "success" in text:
        return "success"
    if "failure" in text:
        return "failure"
    if "recovery" in text:
        return "recovery"
    return None


def extract_task_label_and_outcome(bag_dir: Path, topic_specs: Dict[str, str]) -> Tuple[str, str]:
    task_label_force: Optional[str] = None
    task_label_any: Optional[str] = None
    task_outcome = "unknown"

    candidate_label_topics = [t for t in topic_specs if "task_info" in t]
    candidate_outcome_topics = [t for t in topic_specs if "task_info_collection_states" in t or "task_info" in t]

    if not candidate_label_topics and not candidate_outcome_topics:
        return task_label_force or "unknown", task_outcome

    decoder_factory = DecoderFactory()
    decoder_cache: Dict[Tuple[int, str], Callable[[bytes], Any]] = {}

    def get_decoder(schema, channel) -> Callable[[bytes], Any]:
        key = (schema.id, channel.message_encoding)
        dec = decoder_cache.get(key)
        if dec is None:
            dec = decoder_factory.decoder_for(
                message_encoding=channel.message_encoding,
                schema=schema,
            )
            decoder_cache[key] = dec
        return dec

    found_outcome = False

    for mcap_path in iter_mcap_files(bag_dir):
        is_zstd = mcap_path.suffix == ".zstd"
        if is_zstd:
            if zstd is None:
                continue
            from io import BytesIO
            try:
                with mcap_path.open("rb") as compressed_file:
                    dctx = zstd.ZstdDecompressor()
                    decompressed = dctx.decompress(compressed_file.read())
                stream = BytesIO(decompressed)
            except Exception:
                continue
        else:
            stream = mcap_path.open("rb")

        try:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic

                if topic in candidate_label_topics and task_label_force is None:
                    try:
                        dec = get_decoder(schema, channel)
                        msg = dec(message.data)
                        lbl, is_force = _extract_task_label_from_msg(msg)
                        if lbl:
                            if is_force:
                                task_label_force = sanitize_label(lbl)
                            elif task_label_any is None:
                                task_label_any = sanitize_label(lbl)
                    except Exception:
                        pass

                if (not found_outcome) and topic in candidate_outcome_topics:
                    try:
                        dec = get_decoder(schema, channel)
                        msg = dec(message.data)
                        outcome = _extract_task_outcome_from_msg(msg)
                        if outcome:
                            task_outcome = sanitize_outcome(outcome)
                            found_outcome = True
                    except Exception:
                        pass

                if (task_label_force is not None) and found_outcome:
                    break
            if (task_label_force is not None) and found_outcome:
                break
        finally:
            if not is_zstd:
                stream.close()

    final_label = task_label_force or task_label_any or "unknown"
    return final_label, task_outcome


def iter_mcap_files(bag_dir: Path) -> Iterable[Path]:
    files = sorted(bag_dir.glob("*.mcap"))
    zstd_files = sorted(bag_dir.glob("*.mcap.zstd"))
    all_files = sorted(list(files) + list(zstd_files))
    if not all_files:
        raise FileNotFoundError(f"No .mcap or .mcap.zstd files found in {bag_dir}")
    return all_files


def ensure_parent(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    if path.exists() and overwrite:
        path.unlink()


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, array):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}

    if hasattr(value, "__slots__"):
        result = {}
        for slot in value.__slots__:
            if not isinstance(slot, str):
                continue
            attr_name = slot
            clean_name = attr_name.lstrip("_") or attr_name
            result[clean_name] = to_jsonable(getattr(value, attr_name))
        return result

    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(value).items()}

    return str(value)


def require_video_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if cv2 is None:
        missing.append("opencv-python")
    if missing:
        raise RuntimeError(
            "Saving video topics requires the following Python packages to be installed: "
            + ", ".join(missing)
        )


def require_numpy_dependency() -> None:
    if np is None:
        raise RuntimeError(
            "Decoding pointcloud topics requires the 'numpy' Python package. Install it via `pip install numpy`."
        )


def require_zstd_dependency() -> None:
    if zstd is None:
        raise RuntimeError(
            "Decoding zstd-compressed image topics requires the 'zstandard' Python package. "
            "Install it via `pip install zstandard`."
        )


def image_msg_to_bgr(msg: Any, msg_type: Optional[str] = None) -> "tnp.ndarray":
    if np is None:
        raise RuntimeError("numpy is required to convert images.")

    has_height_width = all(hasattr(msg, attr) for attr in ("height", "width", "step"))
    has_format = hasattr(msg, "format")
    has_data = hasattr(msg, "data")

    if has_height_width and has_data:
        return _raw_image_to_bgr(msg)

    if has_format and has_data:
        return _compressed_image_to_bgr(msg)

    raise ValueError(
        f"Unsupported image message type {msg_type or type(msg).__name__}; "
        "expected sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage."
    )


def _raw_image_to_bgr(msg: Any) -> "tnp.ndarray":
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    encoding = getattr(msg, "encoding", "")
    step = int(getattr(msg, "step", 0))
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or step <= 0 or not encoding or data is None:
        raise ValueError("Invalid sensor_msgs/Image message fields.")
    encoding_l = str(encoding).lower()
    row_bytes = step
    buffer = memoryview(data)
    if len(buffer) < height * row_bytes:
        raise ValueError("Image buffer shorter than expected from height*step.")

    def rows_to_array(dtype: Any, bytes_per_elem: int) -> "tnp.ndarray":
        if row_bytes % bytes_per_elem != 0:
            raise ValueError(f"Row step {row_bytes} not divisible by element size {bytes_per_elem}.")
        row_elems = row_bytes // bytes_per_elem
        arr = np.frombuffer(buffer, dtype=dtype)
        arr = arr[: height * row_elems].reshape(height, row_elems)
        return arr

    if encoding_l in {"bgr8", "rgb8"}:
        arr = rows_to_array(np.uint8, 1)
        channels = 3
        arr = arr[:, : width * channels].reshape(height, width, channels)
        frame = arr if encoding_l == "bgr8" else arr[:, :, ::-1]
    elif encoding_l in {"bgra8", "rgba8"}:
        arr = rows_to_array(np.uint8, 1)
        channels = 4
        arr = arr[:, : width * channels].reshape(height, width, channels)
        frame = arr[:, :, :3] if encoding_l == "bgra8" else arr[:, :, [2, 1, 0]]
    elif encoding_l in {"mono8", "8uc1"}:
        gray = rows_to_array(np.uint8, 1)[:, :width]
        frame = np.stack([gray, gray, gray], axis=-1)
    elif encoding_l in {"mono16", "16uc1", "y16"}:
        gray16 = rows_to_array(np.uint16, 2)[:, :width]
        is_bigendian = bool(getattr(msg, "is_bigendian", False))
        if (is_bigendian and sys.byteorder == "little") or (not is_bigendian and sys.byteorder == "big"):
            gray16 = gray16.byteswap().newbyteorder()
        gray_float = gray16.astype(np.float32)
        max_val = float(np.max(gray_float))
        min_val = float(np.min(gray_float))
        if max_val > min_val:
            norm = (gray_float - min_val) / (max_val - min_val)
        else:
            norm = gray_float / 65535.0
        gray8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        frame = np.stack([gray8, gray8, gray8], axis=-1)
    else:
        raise ValueError(f"Unsupported image encoding '{encoding}'.")
    return np.ascontiguousarray(frame)


def _decode_cv_image(buffer: "tnp.ndarray") -> Optional["tnp.ndarray"]:
    if cv2 is None:
        require_video_dependencies()

    frame = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if frame is not None:
        return frame

    data_bytes = buffer.tobytes()
    for signature in MAGIC_SIGNATURES:
        idx = data_bytes.find(signature)
        if 0 < idx < len(data_bytes):
            trimmed = np.frombuffer(data_bytes[idx:], dtype=np.uint8)
            frame = cv2.imdecode(trimmed, cv2.IMREAD_UNCHANGED)
            if frame is not None:
                return frame
    return None


def _compressed_image_to_bgr(msg: Any) -> "tnp.ndarray":
    if np is None or cv2 is None:
        require_video_dependencies()
    fmt = str(getattr(msg, "format", "")).strip()
    data = getattr(msg, "data", None)
    if data is None:
        raise ValueError("CompressedImage message missing data.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError("CompressedImage data is empty.")
    fmt_lower = fmt.lower()

    if "compresseddepth" in fmt_lower:
        return _decode_compressed_depth(data)

    if "zstd" not in fmt_lower:
        frame = _decode_cv_image(buffer)
        if frame is None:
            raise ValueError(f"cv2.imdecode failed to decode CompressedImage data (format='{fmt}').")
        return _ensure_bgr(frame)

    require_zstd_dependency()
    try:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(buffer.tobytes())
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to decompress Zstd payload: {exc}") from exc

    as_np = np.frombuffer(decompressed, dtype=np.uint8)
    if as_np.size > 0:
        decoded = _decode_cv_image(as_np)
        if decoded is not None:
            return _ensure_bgr(decoded)

    meta = _parse_image_format_metadata(fmt)
    width = meta.get("width")
    height = meta.get("height")
    encoding = meta.get("encoding")
    bigendian = meta.get("bigendian", False)
    if not (width and height and encoding):
        raise ValueError(
            "Zstd payload decompressed but width/height/encoding could not be inferred from "
            f"format='{fmt}'. Include tokens such as 'encoding=16UC1; width=640; height=480'."
        )
    return _raw_bytes_to_bgr(decompressed, width, height, encoding, bigendian)


def _decode_compressed_depth(data: bytes) -> "tnp.ndarray":
    if cv2 is None or np is None:
        require_video_dependencies()

    if len(data) < 12:
        raise RuntimeError("CompressedDepth data too short")

    png_start = data.find(b"\x89PNG")
    if png_start == -1:
        raise RuntimeError("PNG header not found in compressedDepth data")

    png_data = np.frombuffer(data[png_start:], dtype=np.uint8)
    img = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imdecode failed for compressedDepth frame")

    # Ensure uint16 depth
    if img.dtype == np.uint16:
        depth_image = img
    else:
        depth_image = img.astype(np.uint16)

    # Determine normalization bounds
    dmin = DEPTH_NORM_MIN if DEPTH_NORM_MIN is not None else float(np.min(depth_image))
    dmax = DEPTH_NORM_MAX if DEPTH_NORM_MAX is not None else float(np.max(depth_image))

    if dmax <= dmin:
        normalized = np.zeros_like(depth_image, dtype=np.uint8)
    else:
        depth_f = depth_image.astype(np.float32)
        depth_clipped = np.clip(depth_f, dmin, dmax)
        normalized = ((depth_clipped - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    grayscale_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return grayscale_bgr



def _ensure_bgr(image: "tnp.ndarray") -> "tnp.ndarray":
    if cv2 is None:
        require_video_dependencies()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return np.ascontiguousarray(image)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported decoded image shape {image.shape}.")


def _parse_image_format_metadata(fmt: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    fmt_lower = fmt.lower()
    tokens = [token.strip() for token in fmt.replace(",", ";").split(";") if token.strip()]
    for token in tokens:
        lower = token.lower()
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"width", "w"}:
                meta["width"] = int(value)
            elif key in {"height", "h"}:
                meta["height"] = int(value)
            elif key in {"encoding", "format", "type"}:
                meta["encoding"] = value
            elif key in {"bigendian", "endian"}:
                meta["bigendian"] = value.lower() in {"1", "true", "big", "be"}
            elif key == "step":
                meta["step"] = int(value)
            else:
                meta[key] = value
            continue
        if "x" in lower:
            parts = [part.strip() for part in lower.split("x")]
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                first = int(parts[0])
                second = int(parts[1])
                if "compresseddepth" in fmt_lower:
                    meta.setdefault("height", first)
                    meta.setdefault("width", second)
                else:
                    meta.setdefault("width", first)
                    meta.setdefault("height", second)
                continue
        if lower in KNOWN_ENCODINGS:
            meta.setdefault("encoding", token)
            continue
        if lower in {"big endian", "big-endian", "be"}:
            meta["bigendian"] = True
        elif lower in {"little endian", "little-endian", "le"}:
            meta["bigendian"] = False
    return meta


def _raw_bytes_to_bgr(buffer: bytes, width: int, height: int, encoding: str, bigendian: bool = False) -> "tnp.ndarray":
    encoding = encoding.strip()
    encoding_l = encoding.lower()
    if encoding_l in {"bgr8", "rgb8", "bgra8", "rgba8", "mono8", "8uc1"}:
        bytes_per_channel = 1
    elif encoding_l in {"mono16", "16uc1", "y16"}:
        bytes_per_channel = 2
    else:
        raise ValueError(f"Unsupported encoding '{encoding}' for raw byte conversion.")

    if encoding_l in {"bgr8", "rgb8"}:
        channels = 3
    elif encoding_l in {"bgra8", "rgba8"}:
        channels = 4
    else:
        channels = 1

    step = width * channels * bytes_per_channel
    expected_len = step * height
    if len(buffer) < expected_len:
        raise ValueError(
            f"Decompressed buffer too short for {width}x{height} {encoding} image "
            f"(expected {expected_len} bytes, got {len(buffer)})."
        )

    pseudo = _PseudoImage(
        height=height,
        width=width,
        encoding=encoding,
        step=step,
        data=buffer[:expected_len],
        bigendian=bigendian,
    )
    return _raw_image_to_bgr(pseudo)


POINTFIELD_DATATYPES = {
    1: "int8",
    2: "uint8",
    3: "int16",
    4: "uint16",
    5: "int32",
    6: "uint32",
    7: "float32",
    8: "float64",
}

def pointcloud_msg_to_array(msg: Any) -> Tuple["tnp.ndarray", Dict[str, Any]]:
    if np is None:
        require_numpy_dependency()
    required = ("width", "height", "fields", "point_step", "row_step", "data")
    if not all(hasattr(msg, attr) for attr in required):
        raise ValueError("Unsupported pointcloud message; expected sensor_msgs/msg/PointCloud2-like fields.")
    width = int(getattr(msg, "width", 0))
    height = int(getattr(msg, "height", 0))
    point_step = int(getattr(msg, "point_step", 0))
    row_step = int(getattr(msg, "row_step", 0))
    fields = list(getattr(msg, "fields", []) or [])
    data = getattr(msg, "data", None)
    is_bigendian = bool(getattr(msg, "is_bigendian", False))
    is_dense = bool(getattr(msg, "is_dense", False))
    if width <= 0 or height <= 0 or point_step <= 0 or data is None:
        raise ValueError("Invalid PointCloud2 message dimensions or data.")
    buf = memoryview(data)
    expected_bytes = point_step * width * height
    if len(buf) < expected_bytes:
        raise ValueError(
            f"PointCloud2 data buffer shorter than expected ({len(buf)} < {expected_bytes})."
        )
    names: List[str] = []
    formats: List[Any] = []
    offsets: List[int] = []
    field_meta: List[Dict[str, Any]] = []
    endian = ">" if is_bigendian else "<"
    for f in fields:
        name = getattr(f, "name", None)
        datatype = int(getattr(f, "datatype", 0))
        offset = int(getattr(f, "offset", 0))
        count = int(getattr(f, "count", 1)) or 1
        base = POINTFIELD_DATATYPES.get(datatype)
        if not name or base is None:
            continue
        dtype = np.dtype(base).newbyteorder(endian)
        if count > 1:
            dtype = np.dtype((dtype, (count,)))
        names.append(name)
        formats.append(dtype)
        offsets.append(offset)
        field_meta.append(
            {"name": name, "datatype": datatype, "offset": offset, "count": count}
        )
    if not names:
        raise ValueError("PointCloud2 fields could not be interpreted.")
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step})
    cloud = np.frombuffer(buf[:expected_bytes], dtype=dtype, count=width * height)
    if height > 1:
        cloud = cloud.reshape((height, width))
    meta = {
        "width": width,
        "height": height,
        "point_step": point_step,
        "row_step": row_step,
        "is_bigendian": is_bigendian,
        "is_dense": is_dense,
        "fields": field_meta,
        "dtype": dtype.descr,
    }
    return cloud, meta


class VideoTopicWriter:
    DEFAULT_FPS = 30.0
    MIN_FPS = 1.0
    MAX_FPS = 240.0

    def __init__(self, topic: str, output_dir: Path, overwrite: bool):
        if np is None or cv2 is None:
            require_video_dependencies()
        self.topic = topic
        self.output_dir = output_dir / sanitize_topic_name(topic)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / "video.mp4"
        self.info_path = self.output_dir / "video_info.json"
        self.overwrite = overwrite
        self._prepare_outputs()

        # Frame buffering to allow correct FPS writing at finalize
        self._temp_dir = tempfile.TemporaryDirectory(prefix="video_frames_")
        self._temp_dir_path = Path(self._temp_dir.name)
        self._frame_paths: List[Path] = []

        self.first_timestamp_ns: Optional[int] = None
        self.last_timestamp_ns: Optional[int] = None
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.frame_count = 0
        self.bag_files: Set[str] = set()

    def _prepare_outputs(self) -> None:
        for path in (self.video_path, self.info_path):
            if path.exists():
                if not self.overwrite:
                    raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
                path.unlink()

    def add_frame(self, frame: "tnp.ndarray", timestamp_ns: int, bag_file: str) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be BGR with 3 channels.")
        h, w = frame.shape[:2]
        if self.height is None or self.width is None:
            self.height, self.width = h, w
        elif (self.height, self.width) != (h, w):
            raise ValueError(
                f"Frame size mismatch for {self.topic}: expected {self.width}x{self.height}, got {w}x{h}"
            )

        # Persist frame to temp storage to avoid holding all frames in RAM
        frame_path = self._temp_dir_path / f"frame_{self.frame_count:06d}.npy"
        np.save(frame_path, frame)
        self._frame_paths.append(frame_path)

        self.frame_count += 1
        self.first_timestamp_ns = timestamp_ns if self.first_timestamp_ns is None else self.first_timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        self.bag_files.add(bag_file)

    def _compute_measured_fps(self) -> float:
        if self.frame_count < 2 or self.first_timestamp_ns is None or self.last_timestamp_ns is None:
            return self.DEFAULT_FPS
        duration_ns = self.last_timestamp_ns - self.first_timestamp_ns
        if duration_ns <= 0:
            return self.DEFAULT_FPS
        fps = (self.frame_count - 1) / (duration_ns / 1e9)
        fps = min(max(fps, self.MIN_FPS), self.MAX_FPS)
        return float(fps)

    def finalize(self) -> Optional[Tuple[Path, Path]]:
        if self.frame_count == 0:
            self._temp_dir.cleanup()
            return None
        if cv2 is None:
            require_video_dependencies()
        if self.width is None or self.height is None:
            self._temp_dir.cleanup()
            raise RuntimeError("Cannot write video without frame dimensions.")

        measured_fps = self._compute_measured_fps()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, measured_fps, (self.width, self.height))
        if not writer.isOpened():
            self._temp_dir.cleanup()
            raise RuntimeError(f"Failed to open video writer for {self.video_path}")

        try:
            for frame_path in self._frame_paths:
                frame = np.load(frame_path, mmap_mode="r")
                writer.write(frame)
        finally:
            writer.release()
            # Always clean up temp frames
            self._temp_dir.cleanup()

        duration_ns = 0
        if self.first_timestamp_ns is not None and self.last_timestamp_ns is not None:
            duration_ns = self.last_timestamp_ns - self.first_timestamp_ns
        duration_seconds = max(duration_ns / 1e9, 0.0)

        info = {
            "topic": self.topic,
            "video_path": str(self.video_path),
            "frame_count": self.frame_count,
            "duration_seconds": duration_seconds,
            "measured_fps": measured_fps,
            "resolution": {"width": self.width, "height": self.height},
            "start_time_ns": self.first_timestamp_ns,
            "end_time_ns": self.last_timestamp_ns,
            "bag_files": sorted(self.bag_files),
        }
        with self.info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return self.video_path, self.info_path


class PointCloudTopicWriter:
    def __init__(self, topic: str, output_dir: Path, overwrite: bool):
        require_numpy_dependency()
        self.topic = topic
        self.output_dir = output_dir / sanitize_topic_name(topic)
        self.overwrite = overwrite
        self.info_path = self.output_dir / "pointcloud_info.json"
        self._prepare_outputs()
        self.cloud_count = 0
        self.cloud_paths: List[Path] = []
        self.timestamps: List[int] = []
        self.bag_files: Set[str] = set()
        self.first_meta: Optional[Dict[str, Any]] = None

    def _prepare_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.info_path.exists():
            if not self.overwrite:
                raise FileExistsError(f"{self.info_path} already exists. Use --overwrite to replace it.")
            self.info_path.unlink()
        if self.overwrite:
            for f in self.output_dir.glob("cloud_*.npz"):
                try:
                    f.unlink()
                except Exception:
                    pass
        elif any(self.output_dir.glob("cloud_*.npz")):
            raise FileExistsError(
                f"Pointcloud outputs already exist for {self.topic}. Use --overwrite to replace them."
            )

    def add_cloud(self, cloud: "tnp.ndarray", timestamp_ns: int, bag_file: str, meta: Dict[str, Any]) -> None:
        path = self.output_dir / f"cloud_{self.cloud_count:06d}.npz"
        np.savez_compressed(path, points=cloud)
        self.cloud_paths.append(path)
        self.timestamps.append(int(timestamp_ns))
        self.bag_files.add(bag_file)
        if self.first_meta is None:
            self.first_meta = meta
        self.cloud_count += 1

    def finalize(self) -> Optional[Tuple[List[Path], Path]]:
        if self.cloud_count == 0:
            return None
        start_ns = min(self.timestamps) if self.timestamps else None
        end_ns = max(self.timestamps) if self.timestamps else None
        info = {
            "topic": self.topic,
            "cloud_count": self.cloud_count,
            "cloud_files": [str(p) for p in self.cloud_paths],
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "bag_files": sorted(self.bag_files),
            "pointcloud_meta": self.first_meta,
        }
        with self.info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return self.cloud_paths, self.info_path


def _safe_get_duration_seconds(info: Dict[str, Any]) -> float:
    dur = info.get("duration")
    if isinstance(dur, dict) and "nanoseconds" in dur:
        try:
            return max(float(dur["nanoseconds"]) / 1e9, 0.0)
        except Exception:
            pass
    start_ns = info.get("start_time", {}).get("nanoseconds") if isinstance(info.get("start_time"), dict) else info.get("starting_time", None)
    end_ns = info.get("end_time", {}).get("nanoseconds") if isinstance(info.get("end_time"), dict) else info.get("ending_time", None)
    try:
        if isinstance(start_ns, (int, float)) and isinstance(end_ns, (int, float)):
            return max((float(end_ns) - float(start_ns)) / 1e9, 0.0)
    except Exception:
        pass
    return 0.0


def _safe_get_start_end_ns(info: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start = None
    end = None
    st = info.get("start_time")
    et = info.get("end_time")
    if isinstance(st, dict) and "nanoseconds" in st:
        try:
            start = int(st["nanoseconds"])
        except Exception:
            pass
    if isinstance(et, dict) and "nanoseconds" in et:
        try:
            end = int(et["nanoseconds"])
        except Exception:
            pass
    if start is None and isinstance(info.get("starting_time"), int):
        start = int(info["starting_time"])
    if end is None and isinstance(info.get("ending_time"), int):
        end = int(info["ending_time"])
    return start, end


def write_rosbag_info(metadata_path: Path, output_dir: Path, overwrite: bool, mcap_path: Optional[Path] = None) -> Path:
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    info = meta.get("rosbag2_bagfile_information", {}) or {}
    topics = info.get("topics_with_message_count", []) or []

    payload_topics: List[Dict[str, Any]] = []
    for t in topics:
        try:
            tm = t.get("topic_metadata", {}) or {}
            topic_entry: Dict[str, Any] = {
                "name": tm.get("name"),
                "type": tm.get("type"),
                "serialization_format": tm.get("serialization_format"),
                "message_count": t.get("message_count"),
            }
            topic_name = topic_entry.get("name")
            if topic_name:
                sanitized = sanitize_topic_name(topic_name)
                topic_dir = output_dir / sanitized
                for info_fname in ("messages_info.json", "video_info.json", "pointcloud_info.json"):
                    candidate = topic_dir / info_fname
                    if candidate.is_file():
                        try:
                            with candidate.open("r", encoding="utf-8") as fh:
                                topic_entry["decoded_info"] = json.load(fh)
                        except Exception:
                            pass
                        break

            payload_topics.append(topic_entry)
        except Exception:
            continue

    duration_seconds = _safe_get_duration_seconds(info)
    start_ns, end_ns = _safe_get_start_end_ns(info)

    total_messages = info.get("message_count")
    bag_size = None
    if isinstance(info.get("bag_size"), dict) and "bytes" in info["bag_size"]:
        bag_size = info["bag_size"]["bytes"]
    elif isinstance(info.get("bag_size"), (int, float)):
        bag_size = info["bag_size"]

    mcap_size_bytes: Optional[int] = None
    mcap_size_gb: Optional[float] = None
    mcap_name: Optional[str] = None
    if mcap_path is not None:
        try:
            if mcap_path.is_file():
                mcap_size_bytes = int(mcap_path.stat().st_size)
                mcap_size_gb = round(float(mcap_size_bytes) / 1e9, 6)
                mcap_name = mcap_path.name
        except Exception:
            mcap_size_bytes = None
            mcap_size_gb = None
            mcap_name = None

    rosbag_info = {
        "decoded_at": datetime.now().isoformat(),
        "source_metadata_path": str(metadata_path),
        "mcap_file": mcap_name,
        "mcap_size_bytes": mcap_size_bytes,
        "mcap_size_gb": mcap_size_gb,
        "duration_seconds": duration_seconds,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "total_messages": total_messages,
        "bag_size_bytes": bag_size,
        "compression_format": info.get("compression_format"),
        "compression_mode": info.get("compression_mode"),
        "topics": payload_topics,
        "output_dir": str(output_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rosbag_info.json"
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")
        out_path.unlink()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rosbag_info, f, indent=2)
    return out_path


def find_rosbag_folders(root: Path) -> List[Path]:
    bags: List[Path] = []
    if not root.is_dir():
        return bags
    for metadata_path in sorted(root.rglob("metadata.yaml")):
        bag_dir = metadata_path.parent
        if bag_dir.is_dir():
            bags.append(bag_dir)
    return bags


def decode_one_bag(
    bag_dir: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    logger: BagLogger,
) -> bool:
    """
    Returns True if decode was attempted and progressed (logs produced); False if skipped/invalid with no logs.
    """
    bag_name = bag_dir.name
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return False

    try:
        topic_specs = load_topic_specs(metadata_path)
    except Exception:
        return False

    if args.topics:
        missing = [t for t in args.topics if t not in topic_specs]
        if missing:
            return False
        topic_specs = {t: topic_specs[t] for t in args.topics}

    if not topic_specs:
        return False

    task_label, task_outcome = extract_task_label_and_outcome(bag_dir, topic_specs)
    try:
        relative_path = bag_dir.relative_to(input_root)
    except ValueError:
        relative_path = Path(bag_name)
    parent_parts = relative_path.parts[:-1]
    bag_output_rel = (
        Path(*parent_parts) / f"{bag_name}_{task_label}_{task_outcome}"
        if parent_parts
        else Path(f"{bag_name}_{task_label}_{task_outcome}")
    )
    bag_output_name = str(bag_output_rel)
    out_bag_dir = output_root / bag_output_rel

    if out_bag_dir.exists() and not args.overwrite:
        return False

    selected_video_topics = set(topic_specs) & VIDEO_TOPICS
    selected_pointcloud_topics = set(topic_specs) & POINTCLOUD_TOPICS
    if selected_video_topics:
        try:
            require_video_dependencies()
        except RuntimeError:
            # Dependency missing; cannot decode this bag; stay silent.
            return False
    if selected_pointcloud_topics:
        try:
            require_numpy_dependency()
        except RuntimeError:
            return False

    expected_topics: Set[str] = set(topic_specs.keys())
    decoder_factory = DecoderFactory()
    decoder_cache: Dict[Tuple[int, str], Callable[[bytes], Any]] = {}

    def get_decoder(schema, channel) -> Callable[[bytes], Any]:
        key = (schema.id, channel.message_encoding)
        decoder = decoder_cache.get(key)
        if decoder is None:
            decoder = decoder_factory.decoder_for(
                message_encoding=channel.message_encoding,
                schema=schema,
            )
            decoder_cache[key] = decoder
        return decoder

    ndjson_paths: Dict[str, Path] = {}
    ndjson_files: Dict[str, Any] = {}
    ndjson_info_paths: Dict[str, Path] = {}
    ndjson_stats: Dict[str, Dict[str, Any]] = {}
    video_writers: Dict[str, VideoTopicWriter] = {}
    pointcloud_writers: Dict[str, PointCloudTopicWriter] = {}
    total_messages = 0
    seen_topics: Set[str] = set()

    for mcap_path in iter_mcap_files(bag_dir):
        logger.info(f"Decoding {bag_name} -> {mcap_path.name} (out: {bag_output_name}) ...")
        is_zstd = mcap_path.suffix == ".zstd"
        uncompressed_tmp: Optional[Path] = None

        if is_zstd:
            if zstd is None:
                logger.error(f"File {mcap_path.name} requires zstandard. Install via: pip install zstandard")
                print(f"File {mcap_path.name} requires zstandard. Install via: pip install zstandard")
                return True
            try:
                dctx = zstd.ZstdDecompressor()
                with mcap_path.open("rb") as compressed_file:
                    with tempfile.NamedTemporaryFile(suffix=".mcap", delete=False) as tmp:
                        uncompressed_tmp = Path(tmp.name)
                        with dctx.stream_reader(compressed_file) as reader:
                            while True:
                                chunk = reader.read(1 << 20)  # 1 MiB chunks to cap memory
                                if not chunk:
                                    break
                                tmp.write(chunk)
                stream = uncompressed_tmp.open("rb")
            except Exception as exc:
                if uncompressed_tmp and uncompressed_tmp.exists():
                    try:
                        uncompressed_tmp.unlink()
                    except Exception:
                        pass
                logger.error(f"Failed to decompress {mcap_path.name}: {exc}")
                print(f"Failed to decompress {mcap_path.name}: {exc}")
                return True
        else:
            stream = mcap_path.open("rb")

        try:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                if topic not in topic_specs:
                    continue

                seen_topics.add(topic)

                try:
                    decoder = get_decoder(schema, channel)
                    decoded_msg = decoder(message.data)
                except Exception as exc:  # pragma: no cover
                    logger.warn(f"Failed to decode {topic} in {mcap_path.name}: {exc}")
                    continue

                if topic in selected_video_topics:
                    if topic not in video_writers:
                        video_writers[topic] = VideoTopicWriter(topic, out_bag_dir, args.overwrite)
                    video_writer = video_writers[topic]
                    try:
                        frame = image_msg_to_bgr(decoded_msg, topic_specs.get(topic))
                    except Exception as exc:  # pragma: no cover
                        logger.warn(f"Skipping frame for {topic} in {mcap_path.name}: {exc}")
                        continue
                    timestamp_ns = message.publish_time or message.log_time or 0
                    video_writer.add_frame(frame, timestamp_ns, mcap_path.name)
                    total_messages += 1
                    continue

                if topic in selected_pointcloud_topics:
                    if topic not in pointcloud_writers:
                        pointcloud_writers[topic] = PointCloudTopicWriter(topic, out_bag_dir, args.overwrite)
                    pc_writer = pointcloud_writers[topic]
                    try:
                        cloud, cloud_meta = pointcloud_msg_to_array(decoded_msg)
                    except Exception as exc:  # pragma: no cover
                        logger.warn(f"Skipping pointcloud for {topic} in {mcap_path.name}: {exc}")
                        continue
                    timestamp_ns = message.publish_time or message.log_time or 0
                    pc_writer.add_cloud(cloud, timestamp_ns, mcap_path.name, cloud_meta)
                    total_messages += 1
                    continue

                if topic not in ndjson_files:
                    sanitized = sanitize_topic_name(topic)
                    topic_dir = out_bag_dir / sanitized
                    output_file = topic_dir / "messages.ndjson"
                    ensure_parent(output_file, args.overwrite)
                    info_file = topic_dir / "messages_info.json"
                    if info_file.exists():
                        if not args.overwrite:
                            raise FileExistsError(
                                f"{info_file} already exists. Use --overwrite to replace it."
                            )
                        info_file.unlink()
                    fh = output_file.open("w", encoding="utf-8")
                    ndjson_files[topic] = fh
                    ndjson_paths[topic] = output_file
                    ndjson_info_paths[topic] = info_file
                    ndjson_stats[topic] = {
                        "frame_count": 0,
                        "start_time_ns": None,
                        "end_time_ns": None,
                        "bag_files": set(),
                        "type": topic_specs[topic],
                    }

                timestamp_ns = message.publish_time or message.log_time or 0
                record = {
                    "bag_file": mcap_path.name,
                    "topic": topic,
                    "type": topic_specs[topic],
                    "log_time_ns": message.log_time,
                    "publish_time_ns": message.publish_time,
                    "data": to_jsonable(decoded_msg),
                }
                ndjson_files[topic].write(json.dumps(record))
                ndjson_files[topic].write("\n")

                stats = ndjson_stats[topic]
                stats["frame_count"] += 1
                stats["bag_files"].add(mcap_path.name)
                if stats["start_time_ns"] is None or timestamp_ns < stats["start_time_ns"]:
                    stats["start_time_ns"] = timestamp_ns
                if stats["end_time_ns"] is None or timestamp_ns > stats["end_time_ns"]:
                    stats["end_time_ns"] = timestamp_ns
                total_messages += 1
        finally:
            stream.close()
            if uncompressed_tmp:
                try:
                    uncompressed_tmp.unlink()
                except:
                    pass

    missing_topics = sorted(list(expected_topics - seen_topics))
    if missing_topics:
        logger.warn(f"Topics listed in metadata but not found in any MCAP: {missing_topics}")

    info_outputs: Dict[str, Path] = {}
    for topic, stats in ndjson_stats.items():
        info_path = ndjson_info_paths[topic]
        frame_count = stats["frame_count"]
        start_ns = stats["start_time_ns"]
        end_ns = stats["end_time_ns"]
        if start_ns is not None and end_ns is not None and end_ns >= start_ns:
            duration_seconds = (end_ns - start_ns) / 1e9
        else:
            duration_seconds = 0.0
        frequency_hz = frame_count / duration_seconds if duration_seconds > 0 else 0.0
        payload = {
            "bag": bag_name,
            "output_bag_dir_name": bag_output_name,
            "topic": topic,
            "type": stats["type"],
            "messages_path": str(ndjson_paths[topic]),
            "frame_count": frame_count,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "duration_seconds": duration_seconds,
            "frequency_hz": frequency_hz,
            "bag_files": sorted(stats["bag_files"]),
        }
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        info_outputs[topic] = info_path

    video_outputs: Dict[str, Tuple[Path, Path]] = {}
    for topic, writer in video_writers.items():
        try:
            result = writer.finalize()
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to finalize video for {topic} in bag {bag_name}: {exc}")
            print(f"Failed to finalize video for {topic} in bag {bag_name}: {exc}")
            return True
        if result:
            video_outputs[topic] = result
        else:
            logger.warn(f"No frames decoded for {topic}; MP4 not created.")

    pointcloud_outputs: Dict[str, Tuple[List[Path], Path]] = {}
    for topic, writer in pointcloud_writers.items():
        try:
            result = writer.finalize()
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to finalize pointclouds for {topic} in bag {bag_name}: {exc}")
            print(f"Failed to finalize pointclouds for {topic} in bag {bag_name}: {exc}")
            return True
        if result:
            pointcloud_outputs[topic] = result
        else:
            logger.warn(f"No pointclouds decoded for {topic}; outputs not created.")

    try:
        info_json_path = write_rosbag_info(metadata_path, out_bag_dir, args.overwrite, mcap_path=None)
        logger.info(f"Wrote rosbag info -> {info_json_path}")
    except Exception as exc:
        logger.warn(f"Failed to write rosbag_info.json: {exc}")

    logger.info(
        f"Finished decoding {bag_output_name}. Processed {total_messages} message(s), "
        f"{len(ndjson_paths)} NDJSON topic(s), {len(video_outputs)} video topic(s), "
        f"{len(pointcloud_outputs)} pointcloud topic(s)."
    )
    if ndjson_paths:
        logger.info("NDJSON outputs:")
        for topic, path in ndjson_paths.items():
            logger.info(f"   {topic} -> {path}")
            info_path = info_outputs.get(topic)
            if info_path:
                logger.info(f"      info -> {info_path}")
    if video_outputs:
        logger.info("Video outputs:")
        for topic, (video_path, info_path) in video_outputs.items():
            logger.info(f"   {topic} -> {video_path}")
            logger.info(f"      info -> {info_path}")
    if pointcloud_outputs:
        logger.info("Pointcloud outputs:")
        for topic, (cloud_paths, info_path) in pointcloud_outputs.items():
            logger.info(f"   {topic} -> {len(cloud_paths)} cloud file(s)")
            logger.info(f"      info -> {info_path}")

    return True


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        return 1
    workers = max(1, int(args.workers))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = BagLogger(args.output_dir / "decode.log")

    interval = float(args.monitoring_int)

    try:
        while True:
            bag_dirs = find_rosbag_folders(args.input_dir)

            if not bag_dirs:
                if interval <= 0:
                    return 1
                else:
                    time.sleep(interval)
                    continue

            decoded_any = False
            if workers == 1:
                for bag_dir in bag_dirs:
                    ok = decode_one_bag(bag_dir, args.input_dir, args.output_dir, args, logger)
                    decoded_any = decoded_any or ok
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_map = {executor.submit(decode_one_bag, bag_dir, args.input_dir, args.output_dir, args, logger): bag_dir for bag_dir in bag_dirs}
                    for future in as_completed(future_map):
                        try:
                            ok = future.result()
                            decoded_any = decoded_any or ok
                        except Exception as exc:
                            logger.error(f"Decoding failed for {future_map[future]}: {exc}")
                            print(f"Decoding failed for {future_map[future]}: {exc}")

            if interval <= 0:
                return 0
            else:
                # silent sleep between scans
                time.sleep(interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

