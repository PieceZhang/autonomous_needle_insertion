source ./install/setup.bash
export TARGET_P=1 && ros2 launch auto_needle_insertion dataset.launch.py mode:=task2robot_exe_points_motion
pkill -INT -f "ros2 bag record"
bash ./scripts/send_rosbag_stop_command.sh
echo "All ros bag recording processes have been terminated."
