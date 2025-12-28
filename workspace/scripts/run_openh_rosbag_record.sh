#!/bin/bash
set -e

OUT_DIR="/ani_ws/rosbag_recording"
mkdir -p "$OUT_DIR"

BAG_NAME="rosbag2_$(date +%Y%m%d_%H%M%S)"

echo "Start recording rosbag..."
echo "Output: $OUT_DIR/$BAG_NAME"
echo "Output: $OUT_DIR"
echo "Use Ctrl+C to stop recording."

sleep 0.25s

ros2 bag record --output "$OUT_DIR/$BAG_NAME" --topics \
                /vega_vt/image_raw /vega_vt/camera_info \
                /image_raw/compressed \
                /camera/camera/color/image_raw/compressed /camera/camera/color/camera_info /camera/camera/color/metadata \
                /camera/camera/depth/image_rect_raw/compressedDepth /camera/camera/depth/camera_info /camera/camera/depth/metadata \
                /camera/camera/extrinsics/depth_to_color \
                /ati_ft_broadcaster/wrench \
                /joint_states \
                /tcp_pose_broadcaster/pose \
                /ndi/us_probe_pose /ndi/needle_pose \
                /task_info /task_info_collection_states \
                /keyboard_listener/glyphkey_pressed /keyboard_listener/key_pressed
#                --compression-mode file \
#                --compression-format zstd


# most of the size comes from /vega_vt/image_raw

