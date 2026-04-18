###############################################################################
# Stage 1: base – shared ROS 2 Jazzy layer (no driver-specific packages)
###############################################################################
FROM osrf/ros:jazzy-desktop-full-noble@sha256:da0f4fadcc085bd38fc86ad531d3a8d23eab6fd575065521aee0a5dc3236a06a AS base
SHELL ["/bin/bash","-lc"]

# Ensure ALL interactive bash shells source ROS setup
RUN printf '%s\n' \
  'if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then source "/opt/ros/$ROS_DISTRO/setup.bash"; fi' \
  >> /etc/bash.bashrc

ARG UBUNTU_MIRRORS="https://ubuntu-archive.mirrorservice.org/ubuntu https://mirror.ox.ac.uk/sites/archive.ubuntu.com/ubuntu https://archive.ubuntu.com/ubuntu https://ftp.jaist.ac.jp/pub/Linux/ubuntu https://ftp.riken.jp/Linux/ubuntu https://ftp.kaist.ac.kr/ubuntu https://mirror.kakao.com/ubuntu https://free.nchc.org.tw/ubuntu https://mirror.xtom.com.hk/ubuntu https://mirrors.tuna.tsinghua.edu.cn/ubuntu https://mirrors.ustc.edu.cn/ubuntu https://mirrors.bfsu.edu.cn/ubuntu https://mirrors.aliyun.com/ubuntu https://mirrors.sjtug.sjtu.edu.cn/ubuntu"
ENV UBUNTU_MIRRORS="${UBUNTU_MIRRORS}"
RUN set -eux; \
  apt-get install -y --no-install-recommends curl ca-certificates gnupg; \
  . /etc/os-release; CODENAME="${UBUNTU_CODENAME}"; \
  GB_LIST=$(curl -fsSL --max-time 2 https://mirrors.ubuntu.com/GB.txt || true); \
  cand="${UBUNTU_MIRRORS:-https://archive.ubuntu.com/ubuntu}"; \
  if [ -n "$GB_LIST" ]; then cand="$GB_LIST $cand"; fi; \
  fastest=""; best=999999; \
  for m in $cand; do \
    for f in InRelease Release; do \
      url="$m/dists/$CODENAME/$f"; \
      t=$(curl -o /dev/null -s -w '%{time_total}' --max-time 2 "$url" || true); \
      if [ -n "$t" ]; then \
        if awk "BEGIN{exit !($t < $best)}"; then best="$t"; fastest="$m"; fi; \
        break; \
      fi; \
    done; \
  done; \
  : "${fastest:=https://archive.ubuntu.com/ubuntu}"; \
  rm -f /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; \
  printf 'Types: deb\nURIs: %s\nSuites: %s %s-updates %s-backports\nComponents: main restricted universe multiverse\n\nTypes: deb\nURIs: https://security.ubuntu.com/ubuntu\nSuites: %s-security\nComponents: main restricted universe multiverse\n' \
    "$fastest" "$CODENAME" "$CODENAME" "$CODENAME" "$CODENAME" > /etc/apt/sources.list.d/ubuntu.sources; \
  echo "Ubuntu archive mirror chosen: $fastest (best=${best}s)"; \
  cat /etc/apt/sources.list.d/ubuntu.sources

RUN printf '%s\n' \
  'Acquire::Languages "none";' \
  'Acquire::IndexTargets::deb::Contents-deb::DefaultEnabled "false";' \
  'Acquire::PDiffs "false";' \
  'Acquire::Retries "2";' \
  > /etc/apt/apt.conf.d/99lean-apt

ARG ROS2_MIRRORS="https://mirror.umd.edu/packages.ros.org/ros2/ubuntu http://ftp.tudelft.nl/ros2/ubuntu http://packages.ros.org/ros2/ubuntu"
ENV ROS2_MIRRORS="${ROS2_MIRRORS}"
RUN set -eux; \
  rm -f /etc/apt/sources.list.d/ros2*.sources /etc/apt/sources.list.d/*ros2*.list \
        /etc/apt/sources.list.d/*ros*.sources /etc/apt/sources.list.d/*ros*.list \
        /usr/share/ros-apt-source/*ros2*.sources || true; \
  mkdir -p /usr/share/keyrings; \
  curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg; \
  . /etc/os-release; CODENAME="${UBUNTU_CODENAME}"; \
  fastest=""; best=999999; \
  for m in $ROS2_MIRRORS; do \
    for f in InRelease Release; do \
      url="$m/dists/$CODENAME/$f"; \
      resp=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 5 "$url" || echo '000 999999'); \
      code="${resp%% *}"; \
      time="${resp#* }"; \
      if [ "$code" = "200" ]; then \
        if awk "BEGIN{exit !($time < $best)}"; then best="$time"; fastest="$m"; fi; \
        break; \
      fi; \
    done; \
  done; \
  if [ -z "$fastest" ]; then fastest="http://packages.ros.org/ros2/ubuntu"; fi; \
  case "$fastest" in \
    https://packages.ros.org/*) fastest="http://packages.ros.org/ros2/ubuntu" ;; \
  esac; \
  printf 'Types: deb\nURIs: %s\nSuites: %s\nComponents: main\nSigned-By: /usr/share/keyrings/ros-archive-keyring.gpg\n' "$fastest" "$CODENAME" > /etc/apt/sources.list.d/ros2.sources; \
  echo "ROS 2 mirror chosen: $fastest (best=${best}s)"; \
  cat /etc/apt/sources.list.d/ros2.sources

# Common packages shared by ALL images (NO ros2-control, UR, MoveIt, or GStreamer)
RUN apt-get update \
 && apt-get install -y \
      ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
      python3-colcon-common-extensions \
      python3-vcstool \
      python3-pip \
      git \
      build-essential \
      cmake \
      libgtest-dev \
      libgmock-dev \
      python3-rosdep \
      python3-pyqtgraph \
      python3-pynput \
      ros-$ROS_DISTRO-tf2-tools \
      ros-$ROS_DISTRO-realsense2-* \
      # Hardware acceleration / GUI
      mesa-utils \
      x11-apps \
      libgl1 \
      libgl1-mesa-dri \
      # Rosbag
      ros-$ROS_DISTRO-ros2bag \
      ros-$ROS_DISTRO-rosbag2-storage-default-plugins \
      ros-$ROS_DISTRO-rosbag2-transport \
      # USB video grabber
      ros-$ROS_DISTRO-v4l2-camera \
      # OpenCV + cv_bridge
      python3-opencv \
      ros-$ROS_DISTRO-cv-bridge \
      ros-$ROS_DISTRO-image-transport-plugins

RUN rosdep init || true && rosdep update --rosdistro $ROS_DISTRO

# --- ROS-aware Python & Pip wrappers for IDEs (PyCharm) ---
RUN set -Eeuo pipefail; \
  printf '%s\n' '#!/usr/bin/env bash' 'set -Ee -o pipefail' \
  'if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then' \
  '  source "/opt/ros/$ROS_DISTRO/setup.bash"' \
  'fi' \
  'for ws_setup in /opt/ndi_ws/install/setup.bash /opt/franka_ws/install/setup.bash /opt/ati_ws/install/setup.bash /ws/install/setup.bash; do' \
  '  [ -f "$ws_setup" ] && source "$ws_setup"' \
  'done' \
  'exec /usr/bin/python3 "$@"' \
  > /usr/local/bin/python_ros \
  && chmod 0755 /usr/local/bin/python_ros \
  && printf '%s\n' '#!/usr/bin/env bash' 'set -Ee -o pipefail' \
  'if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then' \
  '  source "/opt/ros/$ROS_DISTRO/setup.bash"' \
  'fi' \
  'for ws_setup in /opt/ndi_ws/install/setup.bash /opt/franka_ws/install/setup.bash /opt/ati_ws/install/setup.bash /ws/install/setup.bash; do' \
  '  [ -f "$ws_setup" ] && source "$ws_setup"' \
  'done' \
  'exec /usr/bin/python3 -m pip "$@"' \
  > /usr/local/bin/pip_ros \
  && chmod 0755 /usr/local/bin/pip_ros


###############################################################################
# Stage 2: app – general-purpose image with apt ros2-control, UR, MoveIt, ATI,
#   keyboard.  Used by any service compatible with the apt ros2-control ABI.
#   Serves: ur_driver, ati_ft_driver, keyboard_driver, us_visualizer,
#           rqt_task_ui, ati_ft_nano17_driver, needle_deflection_calculator,
#           usb_video_grabber, dev
###############################################################################
FROM base AS app

RUN apt-get update \
 && apt-get install -y \
      ros-$ROS_DISTRO-ros2controlcli \
      ros-$ROS_DISTRO-ros2-control \
      ros-$ROS_DISTRO-ros2-controllers \
      ros-$ROS_DISTRO-ur \
      ros-$ROS_DISTRO-ur-description \
      ros-$ROS_DISTRO-ur-moveit-config \
      ros-$ROS_DISTRO-moveit \
      ros-$ROS_DISTRO-moveit-py \
      ros-$ROS_DISTRO-moveit-planners-ompl \
      ros-$ROS_DISTRO-moveit-ros-control-interface \
      ros-$ROS_DISTRO-moveit-simple-controller-manager

# --- Patch ur_client_library: increase configuration package timeout from 1s to 10s ---
# The apt-installed ur_client_library 2.x has a hardcoded 1-second timeout in
# UrDriver::init() for receiving the configuration package from the robot's primary
# interface. On some network setups this is too short and causes a fatal crash.
# We rebuild the library from source with the timeout raised to 10 seconds.
RUN set -ex \
 && URCL_TAG="2.9.0" \
 && echo "Patching ur_client_library ${URCL_TAG} configuration timeout..." \
 && cd /tmp && git clone --depth 1 --branch ${URCL_TAG} \
      https://github.com/UniversalRobots/Universal_Robots_Client_Library.git urcl_src \
 && sed -i 's/std::chrono::milliseconds timeout(1000)/std::chrono::milliseconds timeout(10000)/' \
      /tmp/urcl_src/src/ur/ur_driver.cpp \
 && grep -q 'timeout(10000)' /tmp/urcl_src/src/ur/ur_driver.cpp \
 && mkdir /tmp/urcl_build && cd /tmp/urcl_build \
 && cmake /tmp/urcl_src \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/ros/$ROS_DISTRO \
      -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu \
      -DCMAKE_PREFIX_PATH=/opt/ros/$ROS_DISTRO \
      -DBUILD_TESTING=OFF \
 && make -j$(nproc) \
 && make install \
 && rm -rf /tmp/urcl_src /tmp/urcl_build \
 && echo "ur_client_library patched and installed successfully"

# Build ATI + keyboard workspaces (small, stable, no namespace conflict with UR)
ARG ATI_WS=/opt/ati_ws
ARG KB_WS=/opt/kb_ws
RUN mkdir -p ${ATI_WS}/src ${KB_WS}/src
COPY third_party/ros2_net_ft_driver ${ATI_WS}/src/ros2_net_ft_driver
COPY third_party/keystroke          ${KB_WS}/src/keystroke

RUN set -eo pipefail \
 && source /opt/ros/$ROS_DISTRO/setup.bash \
 && rosdep install --from-paths ${ATI_WS}/src -i -y --rosdistro $ROS_DISTRO \
 && colcon build --merge-install --base-paths ${ATI_WS}/src --install-base ${ATI_WS}/install \
 && rosdep install --from-paths ${KB_WS}/src -i -y --rosdistro $ROS_DISTRO \
 && colcon build --merge-install --base-paths ${KB_WS}/src --install-base ${KB_WS}/install

RUN printf '%s\n' \
  '[ -f /opt/ati_ws/install/setup.bash ] && source /opt/ati_ws/install/setup.bash' \
  '[ -f /opt/kb_ws/install/setup.bash ]  && source /opt/kb_ws/install/setup.bash' \
  '[ -f /ws/install/setup.bash ]         && source /ws/install/setup.bash' \
  >> /etc/bash.bashrc


###############################################################################
# Stage 3: ndi – NDI Polaris tracker + GStreamer camera
#   Serves: polaris_driver, polaris_camera_driver, polaris_image_compressed
###############################################################################
FROM base AS ndi

RUN apt-get update \
 && apt-get install -y \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav

ARG NDI_WS=/opt/ndi_ws
RUN mkdir -p ${NDI_WS}/src
COPY ndi_ros2_driver          ${NDI_WS}/src/ndi_ros2_driver
COPY third_party/gscam2       ${NDI_WS}/src/gscam2
COPY third_party/ros2_shared  ${NDI_WS}/src/ros2_shared

RUN set -eo pipefail \
 && source /opt/ros/$ROS_DISTRO/setup.bash \
 && rosdep install --from-paths ${NDI_WS}/src -i -y --rosdistro $ROS_DISTRO \
 && colcon build --merge-install --base-paths ${NDI_WS}/src --install-base ${NDI_WS}/install

RUN printf '%s\n' \
  '[ -f /opt/ndi_ws/install/setup.bash ] && source /opt/ndi_ws/install/setup.bash' \
  >> /etc/bash.bashrc


###############################################################################
# Stage 4: franka – Franka ROS 2 driver with VENDORED ros2-control
#   Serves: franka_driver, franka-dev
#   NOTE: Does NOT install ros-jazzy-ros2-control from apt.  The vendored
#         controller_manager / hardware_interface / realtime_tools from
#         franka_ros2's dependency.repos are built from source here,
#         completely isolated from the UR/apt versions in the ur-app image.
###############################################################################
FROM base AS franka

# MoveIt for Franka planning; ros2-control comes from source via dependency.repos
RUN apt-get update \
 && apt-get install -y \
      ros-$ROS_DISTRO-moveit \
      ros-$ROS_DISTRO-moveit-py \
      ros-$ROS_DISTRO-moveit-planners-ompl \
      ros-$ROS_DISTRO-moveit-ros-control-interface \
      ros-$ROS_DISTRO-moveit-simple-controller-manager \
      ros-$ROS_DISTRO-ros2controlcli

ARG FRANKA_WS=/opt/franka_ws
RUN mkdir -p ${FRANKA_WS}/src
COPY third_party/franka_ros2 ${FRANKA_WS}/src

RUN set -eo pipefail \
 && source /opt/ros/$ROS_DISTRO/setup.bash \
 && vcs import ${FRANKA_WS}/src < ${FRANKA_WS}/src/dependency.repos --recursive --skip-existing \
 && rosdep install --from-paths ${FRANKA_WS}/src -i -y --rosdistro $ROS_DISTRO \
 && colcon build --merge-install --base-paths ${FRANKA_WS}/src --install-base ${FRANKA_WS}/install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF

RUN printf '%s\n' \
  '[ -f /opt/franka_ws/install/setup.bash ] && source /opt/franka_ws/install/setup.bash' \
  >> /etc/bash.bashrc

