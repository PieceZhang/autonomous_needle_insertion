#!/usr/bin/env python3
"""
Decode ROS 2 topics (listed in metadata.yaml) from MCAP files without needing a ROS installation.

Changes in this version:
  * Replace --mcap-dir with --input-dir: a root directory containing multiple rosbag folders.
    Each rosbag folder must contain metadata.yaml and *.mcap / *.mcap.zstd files.
  * Scan all rosbag folders under input-dir; if a corresponding output folder under output-dir
    does NOT exist (or --overwrite is given), decode that rosbag.
  * Add --monitoring-int (seconds): if <=0, scan once and exit; if >0, repeatedly scan at
    the given interval and decode newly appearing rosbags.

Enhancements (from prior version):
  * Non-video topics are written to NDJSON.
  * Video topics (/vega_vt/image_raw, /image_raw/compressed,
    /camera/camera/color/image_raw/compressed,
    /camera/camera/depth/image_rect_raw/compressedDepth)
    are converted into MP4 files plus per-topic info JSON files.
  * Video decoding handles both sensor_msgs/msg/Image and sensor_msgs/msg/CompressedImage,
    including Zstd-compressed payloads whose raw width/height/encoding are embedded in the
    CompressedImage.format string.

Output layout (per rosbag, under <output_dir>/<bag_name>/):
    <bag_out>/<sanitized_topic>/messages.ndjson
    <bag_out>/<sanitized_topic>/messages_info.json
    <bag_out>/<sanitized_topic>/video.mp4
    <bag_out>/<sanitized_topic>/video_info.json
    <bag_out>/rosbag_info.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from array import array
from pathlib import Path
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

from datetime import datetime, timezone

import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

try:  # Optional unless video topics are decoded
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # Optional unless video topics are decoded
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:  # Optional, only when Zstd-compressed CompressedImage payloads are present
    import zstandard as zstd  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import numpy as tnp


VIDEO_TOPICS: Set[str] = {
    "/vega_vt/image_raw",
    "/image_raw/compressed",
    "/camera/camera/color/image_raw/compressed",
    "/camera/camera/depth/image_rect_raw/compressedDepth",
    "/visualize/us_imaging/compressed",
    "/visualize/us_imaging_sync/compressed"
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

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
MAGIC_SIGNATURES = (PNG_SIGNATURE, JPEG_SIGNATURE)


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
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory containing multiple rosbag folders (each with metadata.yaml and *.mcap/*.mcap.zstd).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output root directory. A subfolder per rosbag will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs (if the rosbag output folder already exists).",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        help="Optional subset of topics to decode (default: all listed in metadata).",
    )
    parser.add_argument(
        "--monitoring-int",
        type=float,
        default=0.0,
        help="Scan interval in seconds. <=0 scans once and exits; >0 loops with the given interval, decoding new rosbags.",
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


def require_zstd_dependency() -> None:
    if zstd is None:
        raise RuntimeError(
            "Decoding zstd-compressed image topics requires the 'zstandard' Python package. "
            "Install it via `pip install zstandard`."
        )


def image_msg_to_bgr(msg: Any, msg_type: Optional[str] = None) -> "tnp.ndarray":
    """
    Convert either sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage to a BGR uint8 frame.
    """
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
    """
    Try to decode an image buffer with OpenCV; if it fails, trim to known PNG/JPEG signatures and retry.
    """
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
    """
    Decode compressedDepth format (PNG-encoded 16-bit depth data with header).
    """
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

    if img.dtype == np.uint16:
        depth_image = img
    else:
        depth_image = img.astype(np.uint16)

    min_val = np.min(depth_image)
    max_val = np.max(depth_image)
    if max_val > min_val:
        normalized = ((depth_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_image, dtype=np.uint8)

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


class VideoTopicWriter:
    DEFAULT_FPS = 30.0
    MIN_FPS = 1.0
    MAX_FPS = 240.0

    def __init__(self, topic: str, output_dir: Path, overwrite: bool):
        if np is None or cv2 is None:
            require_video_dependencies()
        self.topic = topic
        # output_dir here is the per-bag output directory
        self.output_dir = output_dir / sanitize_topic_name(topic)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.output_dir / "video.mp4"
        self.info_path = self.output_dir / "video_info.json"
        self.overwrite = overwrite
        self._prepare_outputs()
        self.buffer_frames: List["tnp.ndarray"] = []
        self.frame_timestamps: List[int] = []
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

        self.frame_count += 1
        self.frame_timestamps.append(int(timestamp_ns))
        self.bag_files.add(bag_file)

        self.buffer_frames.append(frame)

    def _compute_measured_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return self.DEFAULT_FPS
        duration_ns = self.frame_timestamps[-1] - self.frame_timestamps[0]
        if duration_ns <= 0:
            return self.DEFAULT_FPS
        fps = (self.frame_count - 1) / (duration_ns / 1e9)
        fps = min(max(fps, self.MIN_FPS), self.MAX_FPS)
        return float(fps)

    def finalize(self) -> Optional[Tuple[Path, Path]]:
        if self.frame_count == 0:
            return None
        if cv2 is None:
            require_video_dependencies()
        if self.width is None or self.height is None:
            raise RuntimeError("Cannot write video without frame dimensions.")

        measured_fps = self._compute_measured_fps()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.video_path), fourcc, measured_fps, (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.video_path}")

        for frame in self.buffer_frames:
            writer.write(frame)
        writer.release()

        duration_ns = 0
        if len(self.frame_timestamps) >= 2:
            duration_ns = self.frame_timestamps[-1] - self.frame_timestamps[0]
        duration_seconds = max(duration_ns / 1e9, 0.0)

        info = {
            "topic": self.topic,
            "video_path": str(self.video_path),
            "frame_count": self.frame_count,
            "duration_seconds": duration_seconds,
            "measured_fps": measured_fps,
            "resolution": {"width": self.width, "height": self.height},
            "start_time_ns": self.frame_timestamps[0],
            "end_time_ns": self.frame_timestamps[-1],
            "bag_files": sorted(self.bag_files),
        }
        with self.info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return self.video_path, self.info_path


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
    """
    Save rosbag_info.json under output_dir containing basic information from metadata.yaml.
    Also attempts to include per-topic decoded info (messages_info.json or video_info.json).
    """
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
                for info_fname in ("messages_info.json", "video_info.json"):
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
    """
    Return all child directories under root that contain a metadata.yaml, treated as rosbag folders.
    """
    bags: List[Path] = []
    if not root.is_dir():
        return bags
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.yaml").is_file():
            bags.append(p)
    return bags


def decode_one_bag(bag_dir: Path, output_root: Path, args: argparse.Namespace) -> bool:
    """
    Decode a single rosbag folder. Returns True if decoding occurred, False if skipped or failed.
    """
    bag_name = bag_dir.name
    out_bag_dir = output_root / bag_name

    if out_bag_dir.exists() and not args.overwrite:
        print(f"[INFO] Skip {bag_name}: output already exists (use --overwrite to re-decode).")
        return False

    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        print(f"[WARN] Skip {bag_name}: metadata.yaml not found.")
        return False

    try:
        topic_specs = load_topic_specs(metadata_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read {metadata_path}: {exc}", file=sys.stderr)
        return False

    if args.topics:
        missing = [t for t in args.topics if t not in topic_specs]
        if missing:
            print(f"[ERROR] {bag_name} metadata does not contain requested topics: {missing}", file=sys.stderr)
            return False
        topic_specs = {t: topic_specs[t] for t in args.topics}

    if not topic_specs:
        print(f"[ERROR] {bag_name} has no selected topics; aborting this bag.", file=sys.stderr)
        return False

    expected_topics: Set[str] = set(topic_specs.keys())

    selected_video_topics = set(topic_specs) & VIDEO_TOPICS
    if selected_video_topics:
        try:
            require_video_dependencies()
        except RuntimeError as exc:
            print(f"[ERROR] {bag_name} missing video dependencies: {exc}", file=sys.stderr)
            return False

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
    total_messages = 0

    seen_topics: Set[str] = set()

    try:
        for mcap_path in iter_mcap_files(bag_dir):
            print(f"[INFO] Decoding {bag_name} -> {mcap_path.name} ...")
            is_zstd = mcap_path.suffix == ".zstd"

            if is_zstd:
                if zstd is None:
                    print(f"[ERROR] File {mcap_path.name} requires zstandard. Install via: pip install zstandard", file=sys.stderr)
                    return False
                from io import BytesIO
                try:
                    with mcap_path.open("rb") as compressed_file:
                        dctx = zstd.ZstdDecompressor()
                        decompressed_chunks = []
                        with dctx.stream_reader(compressed_file) as reader:
                            while True:
                                chunk = reader.read(16384)
                                if not chunk:
                                    break
                                decompressed_chunks.append(chunk)
                        decompressed_data = b"".join(decompressed_chunks)
                except Exception as exc:
                    print(f"[ERROR] Failed to decompress {mcap_path.name}: {exc}", file=sys.stderr)
                    return False
                stream = BytesIO(decompressed_data)
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
                        print(f"[WARN] Failed to decode {topic} in {mcap_path.name}: {exc}", file=sys.stderr)
                        continue

                    if topic in selected_video_topics:
                        if topic not in video_writers:
                            video_writers[topic] = VideoTopicWriter(topic, out_bag_dir, args.overwrite)
                        video_writer = video_writers[topic]
                        try:
                            frame = image_msg_to_bgr(decoded_msg, topic_specs.get(topic))
                        except Exception as exc:  # pragma: no cover
                            print(f"[WARN] Skipping frame for {topic} in {mcap_path.name}: {exc}", file=sys.stderr)
                            continue
                        timestamp_ns = message.publish_time or message.log_time or 0
                        video_writer.add_frame(frame, timestamp_ns, mcap_path.name)
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
                if not is_zstd:
                    stream.close()
    finally:
        for fh in ndjson_files.values():
            fh.close()

    missing_topics = sorted(list(expected_topics - seen_topics))
    if missing_topics:
        print(
            f"[WARN] Topics listed in metadata but not found in any MCAP: {missing_topics}",
            file=sys.stderr,
        )

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
            print(f"[ERROR] Failed to finalize video for {topic} in bag {bag_name}: {exc}", file=sys.stderr)
            return False
        if result:
            video_outputs[topic] = result
        else:
            print(f"[WARN] No frames decoded for {topic}; MP4 not created.", file=sys.stderr)

    try:
        info_json_path = write_rosbag_info(metadata_path, out_bag_dir, args.overwrite, mcap_path=None)
        print(f"[INFO] Wrote rosbag info -> {info_json_path}")
    except Exception as exc:
        print(f"[WARN] Failed to write rosbag_info.json: {exc}", file=sys.stderr)

    print(
        f"[INFO] Finished decoding {bag_name}. Processed {total_messages} message(s), "
        f"{len(ndjson_paths)} NDJSON topic(s), {len(video_outputs)} video topic(s)."
    )
    if ndjson_paths:
        print("[INFO] NDJSON outputs:")
        for topic, path in ndjson_paths.items():
            print(f"       {topic} -> {path}")
            info_path = info_outputs.get(topic)
            if info_path:
                print(f"          info -> {info_path}")
    if video_outputs:
        print("[INFO] Video outputs:")
        for topic, (video_path, info_path) in video_outputs.items():
            print(f"       {topic} -> {video_path}")
            print(f"          info -> {info_path}")

    return True


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    interval = float(args.monitoring_int)
    first_loop = True

    try:
        while True:
            bag_dirs = find_rosbag_folders(args.input_dir)

            if not bag_dirs:
                if interval <= 0:
                    print(f"[ERROR] No rosbag folders (with metadata.yaml) found under {args.input_dir}.", file=sys.stderr)
                    return 1
                else:
                    print(f"[INFO] No rosbag found. Sleeping {interval} s before retry ...")
                    time.sleep(interval)
                    continue

            decoded_any = False
            for bag_dir in bag_dirs:
                bag_name = bag_dir.name
                out_bag_dir = args.output_dir / bag_name
                if out_bag_dir.exists() and not args.overwrite:
                    # Already exists and not overwriting: skip
                    continue
                ok = decode_one_bag(bag_dir, args.output_dir, args)
                decoded_any = decoded_any or ok

            if interval <= 0:
                return 0
            else:
                if decoded_any:
                    print(f"[INFO] Scan done. Sleeping {interval} s before next scan ...")
                else:
                    if first_loop:
                        print(f"[INFO] No rosbag decoded this round. Sleeping {interval} s before monitoring ...")
                    else:
                        print(f"[INFO] No new rosbag to decode. Sleeping {interval} s ...")
                first_loop = False
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted, exiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
