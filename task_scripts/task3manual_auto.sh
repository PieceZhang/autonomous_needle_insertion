#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/install/setup.bash"
python3 "${REPO_ROOT}/auto_needle_insertion/auto_needle_insertion/task3manual_record.py"
pkill -INT -f "ros2 bag record"
bash "${SCRIPT_DIR}/send_rosbag_stop_command.sh"
echo "All ros bag recording processes have been terminated."
