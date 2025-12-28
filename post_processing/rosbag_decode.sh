#!/bin/bash

python ./post_processing/rosbag_decode.py \
      --mcap-dir /mnt/dataset/rosbag_recording/rosbag2_2025_12_23-08_38_42 \
      --output-dir /mnt/dataset/rosbag_decoding \
      --overwrite
