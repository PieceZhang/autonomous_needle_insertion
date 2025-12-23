#!/usr/bin/env bash
set -e

# Change to the directory of this script (optional but often convenient)
cd "$(dirname "$0")"

# Optional: source your ROS 2 and workspace setup files
# Adjust these paths according to your system
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

# Example workspace overlay (change or remove if not needed)
if [ -f "../install/setup.bash" ]; then
  source "../install/setup.bash"
fi

echo "Control the end effector with keyboard ..."
echo "Keep this terminal focused to send keystrokes to the node."
echo

# Run the launch command in the foreground so it can capture keystrokes
ros2 launch auto_needle_insertion dataset.launch.py
