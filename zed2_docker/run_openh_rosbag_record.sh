#!/bin/bash
set -e

WORK_DIR="${WORK_DIR:-/work}"

OUT_DIR="$WORK_DIR/rosbag_recording"
mkdir -p "$OUT_DIR"

BAG_NAME="rosbag2_$(date +%Y%m%d_%H%M%S)"

echo "Start recording rosbag..."
echo "Output: $OUT_DIR/$BAG_NAME"
echo "Use Ctrl+C to stop recording."

ros2 bag record --output "$OUT_DIR/$BAG_NAME" --topics \
                /vega_vt/image_raw/compressed /vega_vt/camera_info \
                /image_raw/compressed \
                /ati_ft_broadcaster/wrench \
                /joint_states \
                /tcp_pose_broadcaster/pose \
                /ndi/us_probe_pose /ndi/needle_pose /ndi/stylus_pose \
                /decoded_coor_image/needle_tip /decoded_coor_image/needle_origin \
                /task_info /task_info_collection_states /task_procedure \
                /keyboard_listener/glyphkey_pressed /keyboard_listener/key_pressed \
                /visualize/us_imaging_sync/compressed \
                /zed/zed_node/depth/camera_info \
                /zed/zed_node/depth/depth_registered/compressedDepth \
                /zed/zed_node/pose \
                /zed/zed_node/rgb/color/rect/camera_info \
                /zed/zed_node/rgb/color/rect/image/compressed
#                --compression-mode file \
#                --compression-format zstd
