source ./install/setup.bash
ros2 launch auto_needle_insertion dataset.launch.py mode:=task2robot_record_points
pkill -INT -f "ros2 bag record"
bash ./scripts/send_rosbag_stop_command.sh
echo "All ros bag recording processes have been terminated."
