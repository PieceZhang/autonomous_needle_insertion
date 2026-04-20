#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

if [ -f "../install/setup.bash" ]; then
  source "../install/setup.bash"
fi

# --- Start the 2nd long-living command in the background ---
python3 /ani_ws/auto_needle_insertion/auto_needle_insertion/rosbag_recorder_control.py &
REC_PID=$!

cleanup() {
  # If the recorder is still alive, terminate it
  if kill -0 "$REC_PID" 2>/dev/null; then
    kill "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Control the end effector with keyboard ..."
echo "Keep this terminal focused to send keystrokes to the node."
echo

# Foreground (keeps stdin/tty for keystrokes)
ros2 launch auto_needle_insertion dataset.launch.py mode:=keyboard_control
