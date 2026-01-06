docker stop $(docker ps -q)

./desktop_build_dockerfile_from_sdk_ubuntu_and_cuda_version.sh ubuntu-24.04 cuda-12.6.3 zedsdk-5.1.0

docker run -it --rm   --gpus all   --privileged   --network=host   --ipc=host   --pid=host   -e NVIDIA_DRIVER_CAPABILITIES=all   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -v /dev:/dev   -v /dev/shm:/dev/shm   zed_ros2_desktop_u24.04_sdk_5.1.0_cuda_12.6.3:latest   bash

docker exec -it zed_ros2_neural_jazzy bash
