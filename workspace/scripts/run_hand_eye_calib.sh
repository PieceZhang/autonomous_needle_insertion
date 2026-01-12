#!/usr/bin/env bash
set -euo pipefail

target="${1:-us_probe}"

ros2 launch auto_needle_insertion move_robot.launch.py mode:=hand_eye_calib "target:=${target}"
