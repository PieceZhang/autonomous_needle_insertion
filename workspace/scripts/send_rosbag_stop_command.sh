#!/bin/bash
echo "Sending stop command to rosbag node..."
ros2 topic pub --once --wait-matching-subscriptions 0 /rosbag_control std_msgs/msg/String "{data: 'stop'}"