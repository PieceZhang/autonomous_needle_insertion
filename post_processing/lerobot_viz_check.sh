python lerobot_viz_check.py \
  --root /Users/leo17/Desktop/surgical_robotics/dataset/Rosbag_lerobot/lerobot_out_run001 \
  --start 0 \
  --length 600 \
  --stride 2 \
  --fps 15 \
  --cell_width 640 \
  --out_mp4 ./viz_check/grid.mp4 \
  --write_error_plot \
  --out_err_png ./viz_check/delta_error.png
