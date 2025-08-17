FROM osrf/ros:kilted-desktop-full
SHELL ["/bin/bash","-lc"]

# Ensure interactive shells have ROS, and install essential networking tools
RUN set -euo pipefail \
 && echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> /root/.bashrc \
 && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ros-kilted-rmw-cyclonedds-cpp \
      python3-colcon-common-extensions \
      git \
      build-essential \
      cmake \
      python3-rosdep \
      iproute2 iputils-ping net-tools netcat-openbsd dnsutils traceroute tcpdump \
      ros-$ROS_DISTRO-ros2controlcli \
      ros-$ROS_DISTRO-ros2-control \
      ros-$ROS_DISTRO-ros2-controllers
# && rm -rf /var/lib/apt/lists/* \


# --- Build NDI ROS 2 driver from local source into /opt/ndi_ws ---
ARG WS=/opt/ndi_ws

# Prepare workspace and copy local subtree (ensure .dockerignore does not exclude it)
RUN mkdir -p ${WS}/src
COPY ndi_ros2_driver ${WS}/src/ndi_ros2_driver

RUN set -eo pipefail \
 && rosdep init || true \
 && rosdep update --rosdistro $ROS_DISTRO \
 && source /opt/ros/$ROS_DISTRO/setup.bash \
 && rosdep install --from-paths ${WS}/src -i -y --rosdistro $ROS_DISTRO \
 && colcon build --merge-install --base-paths ${WS}/src --install-base ${WS}/install \
 && echo '[ -f /opt/ndi_ws/install/setup.bash ] && source /opt/ndi_ws/install/setup.bash' >> /root/.bashrc