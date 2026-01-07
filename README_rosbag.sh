
docker compose --profile dev up -d dev
docker exec -it autonomous_needle_insertion-dev bash


## 1. record rosbag:
## Start core stack (without rosbag_recorder)
#docker compose --profile dev up -d \
#  ur_driver ur_driver_mock polaris_driver polaris_camera_driver realsense_driver dev
## When you’re ready to record:
## param 1: record all
#RECORD_MODE=all
## param 2: record selected topics
#RECORD_MODE=list
#RECORD_TOPICS=/tf /tf_static /vega_vt/image_raw /vega_vt/camera_info /camera/color/image_raw /camera/color/camera_info
## Start recording rosbag
#docker compose --profile dev up -d rosbag_recorder
## To stop recording:
#docker compose stop rosbag_recorder

ros2 bag record --all
# use ctrl+c to stop recording

# 2. replay rosbag:
# first stop live stream
docker compose stop polaris_camera_driver realsense_driver
# run in two separate terminals:
ros2 bag play BAGNAME
ros2 run rqt_image_view rqt_image_view /vega_vt/image_raw


# stop all containers
docker stop $(docker ps -q)