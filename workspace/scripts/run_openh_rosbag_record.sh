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
                /vega_vt/image_raw/compressed /vega_vt/camera_info \
                /image_raw/compressed \
                /ati_ft_broadcaster/wrench \
                /joint_states \
                /tcp_pose_broadcaster/pose \
                /ndi/us_probe_pose /ndi/needle_pose /ndi/stylus_pose \
                /decoded_coor_image/needle_tip /decoded_coor_image/needle_origin \
                /task_info /task_info_collection_states /task_procedure \
                /keyboard_listener/glyphkey_pressed /keyboard_listener/key_pressed \
                /visualize/us_imaging/compressed /visualize/us_imaging_sync/compressed \
                /zed/zed_node/rgb/color/rect/image/compressed \
                /zed/zed_node/depth/depth_registered/compressedDepth \
                /zed/zed_node/point_cloud/cloud_registered \
                /zed/zed_node/pose
#                --compression-mode file \
#                --compression-format zstd


# most of the size comes from /vega_vt/image_raw

#                /camera/camera/color/image_raw/compressed /camera/camera/color/camera_info /camera/camera/color/metadata \
#                /camera/camera/depth/image_rect_raw/compressedDepth /camera/camera/depth/camera_info /camera/camera/depth/metadata \
#                /camera/camera/extrinsics/depth_to_color \