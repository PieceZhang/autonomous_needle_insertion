source ./install/setup.bash
export TASK4_SUBTASK=2 && ros2 launch auto_needle_insertion dataset.launch.py mode:=find_needle_task4
pkill -INT -f "ros2 bag record"
bash ./scripts/send_rosbag_stop_command.sh
echo "All ros bag recording processes have been terminated."
