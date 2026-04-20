#!/usr/bin/env python3
"""
Convert a ROS 2 image topic stored in an MCAP file (.mcap) to an MP4 video.

Requirements (no ROS 2 installation needed):

    pip install mcap mcap-ros2-support opencv-python numpy

Usage:

    python mcap_to_mp4.py \
        --bag path/to/recording.mcap \
        --topic /vega_vt/image_raw \
        --out output.mp4

If --fps is not provided, the script estimates FPS from message timestamps.
"""

import argparse
import os
import struct
from typing import Optional, Tuple, List

import numpy as np
import cv2

from mcap_ros2.reader import read_ros2_messages


def estimate_fps(
    bag_path: str,
    topic: str,
    max_samples: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Estimate FPS from message timestamps in the MCAP file for the given topic.

    Returns (fps, frame_count).

    If too few frames to estimate, falls back to 30 FPS.
    """
    timestamps: List[float] = []

    for i, msg in enumerate(read_ros2_messages(bag_path, topics=[topic])):
        t = msg.log_time_ns / 1e9  # seconds
        timestamps.append(t)
        if max_samples is not None and i + 1 >= max_samples:
            break

    frame_count = len(timestamps)
    if frame_count < 2:
        # Fallback if not enough samples to estimate
        return 30.0, frame_count

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return 30.0, frame_count

    fps = (frame_count - 1) / duration
    return fps, frame_count


def decode_compressed_image(ros_msg) -> np.ndarray:
    """
    Decode sensor_msgs/msg/CompressedImage into a BGR image (H x W x 3, uint8).
    """
    data = np.frombuffer(ros_msg.data, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed for CompressedImage frame")
    return img  # BGR


def decode_compressed_depth_image(ros_msg) -> np.ndarray:
    """
    Decode sensor_msgs/msg/CompressedImage with compressedDepth format.

    The compressedDepth format stores depth data as PNG with a special header.
    Format: [header][PNG data]
    Header contains depth quantization parameters.
    """
    data = ros_msg.data

    # CompressedDepth format has a header before PNG data
    # Header format: "depth\0" + 4 bytes (float32) depth_quantization + 4 bytes (float32) max_depth
    if len(data) < 12:
        raise RuntimeError("CompressedDepth data too short")

    # Try to find PNG magic bytes
    png_start = data.find(b'\x89PNG')

    if png_start == -1:
        raise RuntimeError("PNG header not found in compressedDepth data")

    # Decode PNG data
    png_data = np.frombuffer(data[png_start:], dtype=np.uint8)
    img = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError("cv2.imdecode failed for compressedDepth frame")

    # The PNG contains 16-bit depth values
    if img.dtype == np.uint16:
        depth_image = img
    else:
        # If it came as uint8, we need to convert
        depth_image = img.astype(np.uint16)

    # Normalize to 0-255 range for visualization
    min_val = np.min(depth_image)
    max_val = np.max(depth_image)

    if max_val > min_val:
        normalized = ((depth_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth_image, dtype=np.uint8)

    # Apply colormap for better visualization
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored

    # Convert grayscale to BGR (3-channel) for video compatibility
    grayscale_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return grayscale_bgr


def decode_raw_image(ros_msg) -> np.ndarray:
    """
    Decode sensor_msgs/msg/Image (raw) into a BGR image (H x W x 3, uint8).

    Supports common encodings:
      - "bgr8"
      - "rgb8"
      - "mono8"
      - "16UC1" (depth images - will be normalized and colorized)
    """
    height = ros_msg.height
    width = ros_msg.width
    encoding = ros_msg.encoding
    step = ros_msg.step  # bytes per row
    data_bytes = np.frombuffer(ros_msg.data, dtype=np.uint8)

    # Handle 16UC1 (depth images)
    if encoding == "16UC1":
        if data_bytes.size < height * step:
            raise ValueError(
                f"Image data too small: expected at least {height * step} bytes, "
                f"got {data_bytes.size}"
            )

        # Reshape to get the raw bytes
        img_bytes = data_bytes[: height * step].reshape((height, step))

        # Convert to 16-bit unsigned integers
        # Each pixel is 2 bytes, so width * 2 bytes per row
        row_pixels_bytes = width * 2
        if step < row_pixels_bytes:
            raise ValueError(
                f"Step smaller than expected row size: step={step}, "
                f"expected at least {row_pixels_bytes}"
            )

        # Extract pixel data and reshape
        img_bytes = img_bytes[:, :row_pixels_bytes]
        depth_image = np.frombuffer(img_bytes.tobytes(), dtype=np.uint16).reshape((height, width))

        # Normalize to 0-255 range for visualization
        # Handle the case where all values are the same
        min_val = np.min(depth_image)
        max_val = np.max(depth_image)

        if max_val > min_val:
            normalized = ((depth_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros((height, width), dtype=np.uint8)

        # Apply colormap for better visualization
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return colored

    encoding_lower = encoding.lower()

    if encoding_lower not in ("bgr8", "rgb8", "mono8"):
        raise ValueError(f"Unsupported encoding: {ros_msg.encoding}")

    if data_bytes.size < height * step:
        raise ValueError(
            f"Image data too small: expected at least {height * step} bytes, "
            f"got {data_bytes.size}"
        )

    # Reshape to (H, row_stride)
    img_bytes = data_bytes[: height * step].reshape((height, step))

    if encoding_lower == "mono8":
        # First width bytes in each row are the pixels; drop any padding
        gray = img_bytes[:, :width]
        # Convert single-channel to BGR for VideoWriter
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr

    # Color encodings
    channels = 3
    row_pixels_bytes = width * channels

    if step < row_pixels_bytes:
        raise ValueError(
            f"Step smaller than expected row size: step={step}, "
            f"expected at least {row_pixels_bytes}"
        )

    # Drop padding beyond width*channels, if any
    img_bytes = img_bytes[:, :row_pixels_bytes]
    img = img_bytes.reshape((height, width, channels))

    if encoding_lower == "rgb8":
        # OpenCV expects BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # If "bgr8", already in BGR
    return img


def convert_mcap_topic_to_mp4(
    bag_path: str,
    topic: str,
    output_path: str,
    fps: Optional[float] = None,
) -> None:
    """
    Main conversion function:
    - reads MCAP file
    - decodes frames from `topic`
    - writes them to `output_path` as MP4
    """

    if fps is None:
        print(f"[INFO] Estimating FPS from timestamps on topic '{topic}'...")
        fps_est, frame_count_sampled = estimate_fps(bag_path, topic)
        fps = fps_est
        print(
            f"[INFO] Estimated FPS: {fps:.2f} (based on {frame_count_sampled} frames)."
        )
    else:
        print(f"[INFO] Using user-specified FPS: {fps:.2f}")

    # Second pass: decode frames and write video
    writer: Optional[cv2.VideoWriter] = None
    frame_count = 0
    schema_name: Optional[str] = None

    print(f"[INFO] Reading MCAP: {bag_path}")
    for msg in read_ros2_messages(bag_path, topics=[topic]):
        ros_msg = msg.ros_msg
        schema_name = msg.schema.name  # e.g. "sensor_msgs/msg/CompressedImage"
        schema_name_lower = schema_name.lower()

        if "compressedimage" in schema_name_lower:
            # Check if it's compressedDepth format
            if hasattr(ros_msg, 'format') and 'compressedDepth' in ros_msg.format:
                frame = decode_compressed_depth_image(ros_msg)
            elif 'depth' in topic.lower():
                # If topic name contains 'depth', assume it's compressed depth
                frame = decode_compressed_depth_image(ros_msg)
            else:
                frame = decode_compressed_image(ros_msg)
        elif "sensor_msgs/msg/image" in schema_name_lower or " image" in schema_name_lower:
            frame = decode_raw_image(ros_msg)
        else:
            raise ValueError(
                f"Unsupported message type for video: {schema_name}. "
                "Expected sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage."
            )

        height, width = frame.shape[:2]

        if writer is None:
            # Create VideoWriter on first frame (we now know size)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

            print(
                f"[INFO] Writing MP4 to: {output_path}\n"
                f"       Topic: {topic}\n"
                f"       Type:  {schema_name}\n"
                f"       Size:  {width}x{height}\n"
                f"       FPS:   {fps:.2f}"
            )

        writer.write(frame)
        frame_count += 1

    if writer is not None:
        writer.release()

    if frame_count == 0:
        print(
            f"[WARN] No messages found on topic '{topic}' in '{bag_path}'. "
            "No video written."
        )
    else:
        print(f"[INFO] Done. Wrote {frame_count} frames to {output_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a ROS 2 image topic in an MCAP file to an MP4 video "
        "without requiring a ROS 2 installation."
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to .mcap file recorded by rosbag2.",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Image topic to extract (e.g. /vega_vt/image_raw).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path. Default: <bagname>_<sanitized_topic>.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS to use for the video. "
        "If not provided, it will be estimated from timestamps.",
    )

    args = parser.parse_args()

    bag_path = args.bag
    topic = args.topic

    if args.out is not None:
        output_path = args.out
    else:
        base = os.path.splitext(os.path.basename(bag_path))[0]
        safe_topic = topic.strip("/").replace("/", "_")
        if not safe_topic:
            safe_topic = "topic"
        output_path = f"{base}_{safe_topic}.mp4"

    convert_mcap_topic_to_mp4(
        bag_path=bag_path,
        topic=topic,
        output_path=output_path,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

"""
python post_processing/_demo_export_video.py \
    --bag rosbag_recording/rosbag2_2025_12_19-05_24_17/rosbag2_2025_12_19-05_24_17_0.mcap \
    --topic /camera/camera/depth/image_rect_raw/compressedDepth \
    --out image_rect_raw_compressedDepth.mp4 
"""
