{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\csgray\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\fs22 \cf2 \CocoaLigature0 docker stop $(docker ps -q)\
\
./desktop_build_dockerfile_from_sdk_ubuntu_and_cuda_version.sh ubuntu-24.04 cuda-12.6.3 zedsdk-5.1.0\
\
docker run -it --rm   --gpus all   --privileged   --network=host   --ipc=host   --pid=host   -e NVIDIA_DRIVER_CAPABILITIES=all   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -v /dev:/dev   -v /dev/shm:/dev/shm   zed_ros2_desktop_u24.04_sdk_5.1.0_cuda_12.6.3:latest   bash\
\
docker exec -it zed_ros2_neural_jazzy bash}