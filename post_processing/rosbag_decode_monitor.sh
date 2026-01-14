#!/bin/bash

python  ./post_processing/rosbag_decode_monitor.py \
      --input-dir /mnt/dataset/storage \
      --output-dir /mnt/dataset/storage_decoding \
      --monitoring-int 5 \
      --workers 4