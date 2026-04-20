#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/install/setup.bash"
ros2 launch auto_needle_insertion dataset.launch.py mode:=task2robot_record_points
pkill -INT -f "ros2 bag record"
bash "${SCRIPT_DIR}/send_rosbag_stop_command.sh"
echo "All ros bag recording processes have been terminated."
