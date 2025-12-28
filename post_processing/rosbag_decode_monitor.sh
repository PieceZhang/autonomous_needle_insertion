#!/bin/bash

python ./post_processing/rosbag_decode_monitor.py \
      --input-dir /mnt/dataset/rosbag_recording \
      --output-dir /mnt/dataset/rosbag_decoding \
      --monitoring-int 5
