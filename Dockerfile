FROM osrf/ros:jazzy-desktop-full
SHELL ["/bin/bash","-lc"]

# Ensure ALL interactive bash shells (for any user) source ROS and local workspaces
RUN printf '%s\n' \
  'if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then source "/opt/ros/$ROS_DISTRO/setup.bash"; fi' \
  '[ -f /opt/ndi_ws/install/setup.bash ] && source /opt/ndi_ws/install/setup.bash' \
  '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' \
  >> /etc/bash.bashrc

ARG UBUNTU_MIRRORS="https://ubuntu-archive.mirrorservice.org/ubuntu https://mirror.ox.ac.uk/sites/archive.ubuntu.com/ubuntu https://archive.ubuntu.com/ubuntu https://ftp.jaist.ac.jp/pub/Linux/ubuntu https://ftp.riken.jp/Linux/ubuntu https://download.nus.edu.sg/mirror/ubuntu https://ftp.kaist.ac.kr/ubuntu https://mirror.kakao.com/ubuntu https://free.nchc.org.tw/ubuntu https://mirror.xtom.com.hk/ubuntu https://mirrors.tuna.tsinghua.edu.cn/ubuntu https://mirrors.ustc.edu.cn/ubuntu https://mirrors.bfsu.edu.cn/ubuntu https://mirrors.aliyun.com/ubuntu https://mirrors.sjtug.sjtu.edu.cn/ubuntu"
ENV UBUNTU_MIRRORS="${UBUNTU_MIRRORS}"
RUN set -eux; \
#  apt-get update; \
  apt-get install -y --no-install-recommends curl ca-certificates gnupg; \
  . /etc/os-release; CODENAME="${UBUNTU_CODENAME}"; \
  # Try country-specific list (UK) briefly; fall back to curated list
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
  # Use deb822-style ubuntu.sources exclusively to avoid duplicates with sources.list
  rm -f /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list; \
  printf 'Types: deb\nURIs: %s\nSuites: %s %s-updates %s-backports\nComponents: main restricted universe multiverse\n\nTypes: deb\nURIs: https://security.ubuntu.com/ubuntu\nSuites: %s-security\nComponents: main restricted universe multiverse\n' \
    "$fastest" "$CODENAME" "$CODENAME" "$CODENAME" "$CODENAME" > /etc/apt/sources.list.d/ubuntu.sources; \
  echo "Ubuntu archive mirror chosen: $fastest (best=${best}s)"; \
  cat /etc/apt/sources.list.d/ubuntu.sources

# Reduce apt index payload to speed up `apt update`
RUN printf '%s\n' \
  'Acquire::Languages "none";' \
  'Acquire::IndexTargets::deb::Contents-deb::DefaultEnabled "false";' \
  'Acquire::PDiffs "false";' \
  'Acquire::Retries "2";' \
  > /etc/apt/apt.conf.d/99lean-apt

# Detect and use the fastest ROS 2 APT mirror (with packages.ros.org as fallback)
# Replace preconfigured ROS 2 source with a mirror (avoid Signed-By conflicts)
ARG ROS2_MIRRORS="https://mirror.umd.edu/packages.ros.org/ros2/ubuntu http://ftp.tudelft.nl/ros2/ubuntu https://mirrors.ustc.edu.cn/ros2/ubuntu https://mirrors.bfsu.edu.cn/ros2/ubuntu https://mirrors.sustech.edu.cn/ros2/ubuntu https://packages.ros.org/ros2/ubuntu"
ENV ROS2_MIRRORS="${ROS2_MIRRORS}"
RUN set -eux; \
#  apt-get update; \
#  apt-get install -y --no-install-recommends curl ca-certificates gnupg; \
  # Remove any preconfigured ROS 2 sources from the base image (deb822 and legacy)
  rm -f /etc/apt/sources.list.d/ros2*.sources /etc/apt/sources.list.d/*ros2*.list \
        /etc/apt/sources.list.d/*ros*.sources /etc/apt/sources.list.d/*ros*.list \
        /usr/share/ros-apt-source/*ros2*.sources || true; \
  # Install ROS GPG key into a keyring file (path-based; used by Signed-By below)
  curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg; \
  . /etc/os-release; CODENAME="${UBUNTU_CODENAME}"; \
  # Pick the fastest reachable mirror by probing InRelease/Release
  fastest=""; best=999999; \
  for m in $ROS2_MIRRORS; do \
    for f in InRelease Release; do \
      url="$m/dists/$CODENAME/$f"; \
      t=$(curl -o /dev/null -s -w '%{time_total}' --max-time 3 "$url" || true); \
      if [ -n "$t" ]; then \
        if awk "BEGIN{exit !($t < $best)}"; then best="$t"; fastest="$m"; fi; \
        break; \
      fi; \
    done; \
  done; \
  if [ -z "$fastest" ]; then fastest="https://packages.ros.org/ros2/ubuntu"; fi; \
  printf 'Types: deb\nURIs: %s\nSuites: %s\nComponents: main\nSigned-By: /usr/share/keyrings/ros-archive-keyring.gpg\n' "$fastest" "$CODENAME" > /etc/apt/sources.list.d/ros2.sources; \
  echo "ROS 2 mirror chosen: $fastest (best=${best}s)"; \
  cat /etc/apt/sources.list.d/ros2.sources

RUN apt-get update \
 && apt-get install -y \
      ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
      python3-colcon-common-extensions \
      git \
      build-essential \
      cmake \
      python3-rosdep \
#      iproute2 iputils-ping net-tools netcat-openbsd dnsutils traceroute tcpdump \
      ros-$ROS_DISTRO-ros2controlcli \
      ros-$ROS_DISTRO-ros2-control \
      ros-$ROS_DISTRO-ros2-controllers \
      ros-$ROS_DISTRO-ur \
      ros-$ROS_DISTRO-tf2-tools \
      ros-$ROS_DISTRO-moveit-py \
      ros-$ROS_DISTRO-ur-moveit-config \
      ros-$ROS_DISTRO-moveit-configs-utils
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
 && colcon build --merge-install --base-paths ${WS}/src --install-base ${WS}/install

# --- ROS-aware Python & Pip wrappers for IDEs (PyCharm) ---
RUN set -Eeuo pipefail; \
  printf '%s\n' '#!/usr/bin/env bash' 'set -Ee -o pipefail' \
  '# Source core ROS env if available' \
  'if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then' \
  '  source "/opt/ros/$ROS_DISTRO/setup.bash"' \
  'fi' \
  '# Source any overlay workspaces if present' \
  'for ws in /opt/ndi_ws /ws /root/ws; do' \
  '  if [ -f "$ws/install/setup.bash" ]; then' \
  '    source "$ws/install/setup.bash"' \
  '  fi' \
  'done' \
  '# Hand off to the real interpreter' \
  'exec /usr/bin/python3 "$@"' \
  > /usr/local/bin/python_ros \
  && chmod 0755 /usr/local/bin/python_ros \
  && printf '%s\n' '#!/usr/bin/env bash' 'set -Ee -o pipefail' \
  'if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then' \
  '  source "/opt/ros/$ROS_DISTRO/setup.bash"' \
  'fi' \
  'for ws in /opt/ndi_ws /ws /root/ws; do' \
  '  if [ -f "$ws/install/setup.bash" ]; then' \
  '    source "$ws/install/setup.bash"' \
  '  fi' \
  'done' \
  'exec /usr/bin/python3 -m pip "$@"' \
  > /usr/local/bin/pip_ros \
  && chmod 0755 /usr/local/bin/pip_ros
