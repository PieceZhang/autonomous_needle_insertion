#!/bin/bash

echo "Start recording rosbag..."
echo "Use Ctrl+C to stop recording."

sleep 0.25s

ros2 bag record /vega_vt/image_raw /vega_vt/camera_info \
                /image_raw/compressed \
                /camera/camera/color/image_raw/compressed /camera/camera/color/camera_info /camera/camera/color/metadata \
                /camera/camera/depth/image_rect_raw/compressedDepth /camera/camera/depth/camera_info /camera/camera/depth/metadata \
                /camera/camera/extrinsics/depth_to_color \
                /ati_ft_broadcaster/wrench \
                /joint_states \
                /tcp_pose_broadcaster/pose \
                /ndi/us_probe_pose /ndi/needle_pose \
                /task_info \
                --compression-mode file \
                --compression-format zstd

# TODO add topics: keyboard, procedural_phase,...
# TODO must of the size comes from /vega_vt/image_raw

