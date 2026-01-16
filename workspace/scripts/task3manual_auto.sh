source ./install/setup.bash
#ros2 launch auto_needle_insertion dataset.launch.py mode:=task3manual_record
python3 src/auto_needle_insertion/auto_needle_insertion/task3manual_record.py
pkill -INT -f "ros2 bag record"
bash ./scripts/send_rosbag_stop_command.sh
echo "All ros bag recording processes have been terminated."
