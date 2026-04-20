#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SCRIPT_DIR}"

if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

if [ -f "${REPO_ROOT}/install/setup.bash" ]; then
  source "${REPO_ROOT}/install/setup.bash"
fi

# --- Start the 2nd long-living command in the background ---
python3 "${REPO_ROOT}/auto_needle_insertion/auto_needle_insertion/rosbag_recorder_control.py" &
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
