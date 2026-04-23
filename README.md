#  Robotic Ultrasound-Guided Autonomous Needle Insertion (RUGANI)

[![License](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/your_dockerhub_username/your_image_name.svg)](https://hub.docker.com/layers/osrf/ros/jazzy-desktop-full/images/sha256-b706aba86d1be07e9dc2834bf54c9acf1be87c2bad1aea83cd2f49ff738b6f5e)

## `💡This is a static snapshot of the private development repository. It is provided for reference only and is not intended for active development.`

## NEW
The Franka Research 3 driver has been fully integrated and tested on a machine with a real-time kernel.

## TL;DR
End users can jump to [**Quick start**](#quick-start) to clone the repo, generate a `.env`, and bring up the stack.
Administrators can refer to [**For administrators**](#for-administrators) for host setup details.
Interested developers can read [**Under the hood**](#under-the-hood) for architecture and implementation details.
Refer to [**Troubleshooting**](#troubleshooting) for solutions of common problems.

A [video demo](https://www.youtube.com/watch?v=AOPiP3fkReg) is available to show the deployment process.

## Overview
This repository provides a reproducible, Dockerized ROS 2 workspace for autonomous needle insertion R&D guided by robotic ultrasound.
It integrates a Universal Robots (UR) arm (hardware or mock), an NDI Polaris Vega VT optical tracker, and an ultrasound system.
The workspace includes MoveItPy‑based robot motion control, RGB data collection, along with calibration and monitoring utilities.

**Key features**
- Maker-based registration of pre-operative images to intra-operative tracker frame.
- End‑effector profile‑tracing.
- End‑effector mirroring tracker pose in servo mode.
- Keyboard control of the end-effector in local frame.
- Hand–eye calibration (UR + NDI Polaris) using OpenCV with residual checks.
- Lightweight PoseStamped/TF tool‑pose reporter.
- Integrated GStreamer pipeline for RGB data capture.
- Hardware GL for rendering video streams in ROS.
- Docker Compose services:
  - `ur_driver` (UR5e hardware)
  - `ur_driver_mock` (mock hardware)
  - `polaris_driver` (optical tracking with NDI Polaris)
  - `polaris_camera_driver` (RGB camera with NDI Polaris Vega VT)
  - `franka_driver` (Franka Research 3)
  - `ati_ft_driver` (ATI F/T sensor)
  - `dev` (interactive shell with the workspace mounted)

## Key concepts

| Term                       | What it means here                                                                                                             |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Host**                   | The physical machine where Docker runs and this repository is cloned.                                                          |
| **Container**              | An isolated runtime for a packaged app (an image) used for dev and runtime reproducibility.                                    |
| **Docker Compose profile** | A named grouping of services you activate selectively (e.g., `--profile dev`).                                                 |
| **`.env` file**            | Key–value config read by Compose to parameterize services (e.g., `RUNTIME_WS_DIR`, `UR_ROBOT_IP`).                             |
| **ROS 2 workspace**        | The mounted colcon workspace at `$RUNTIME_WS_DIR` containing your packages.                                                    |
| **Motion profile**         | Plan a trajectory with MoveIt and execute it (as opposed to continuous servoing).                                              |
| **Hand–eye calibration**   | The fixed transform between the robot “hand” and the “eye” (camera/tracker), typically via OpenCV’s `calibrateHandEye`.        |
| **Probe calibration**      | The fixed transform between the ultrasound imaging plane and the tracker, typically via the PLUS Toolkit and a N-wire phantom. |
| **UR ROS 2 driver**        | The official ROS 2 driver for Universal Robots arm (e.g., UR5e); `UR_ROBOT_IP` points to the controller.                       |
| **Mock hardware**          | Run the stack without real devices (simulated/mock drivers only) by setting `USE_MOCK_HARDWARE=true`.                          |
| **NDI Polaris**            | Optical tracking system reporting tool poses in real time; `POLARIS_IP` is its host.                                           |
| **GStreamer**              | Streaming service to capture RGB camera data; `gscam2` is a ROS 2 wrapper of it.                                               |
| **Tailscale / tailnet**    | Mesh VPN used to reach lab devices remotely; a **tailnet** is the Tailscale network.                                           |

## Quick start

Run this project with Docker Compose, and no host-side ROS 2 install is required.
It is **tested on Linux** and is expected to work on macOS/Windows via Docker Desktop.

### 0) Access (required for lab devices)
>You are going to work on a host machine on a lab LAN, no matter locally or remotely. Therefore, ensure you have the following access.

**Host access.** Request a personal login on the lab host from the administrator.

**Remote access (optional).** If you will reach devices over the lab tailnet:
  1. Create your own Tailscale account [here](https://tailscale.com) and request a **Tailscale invite link** to the team tailnet from the administrator.
  2. Install Tailscale on your own device and join the tailnet. You can verify connectivity with:
     ```bash
     tailscale status
     tailscale ip -4   # optional: note your Tailscale IPv4 address
     ```

**GitHub access** is needed to clone or contribute to the repo.
  1. If you work locally on the host machine, follow [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to create an SSH key and add it to your GitHub account.
  2. If you connect to the host via Tailscale and work remotely, you can forward your Mac’s existing SSH key to the host machine (SSH agent forwarding) with the following steps:
     ```bash
     # On your Mac (ensure the key is in the agent; stores passphrase in Keychain)
     ssh-add --apple-use-keychain ~/.ssh/your_private_github_key_such_as_id_ed25519

     # Connect to the remote with agent forwarding
     ssh -A youruser@remote.host
     
     # Optionally, make the forwarding persistent by adding to your local ~/.ssh/config:
     Host remote.host
         HostName host_ip_on_tailnet
         User youruser
         IdentityFile ~/.ssh/your_private_github_key_such_as_id_ed25519
         ForwardAgent yes
     # Then connect with:
     ssh remote.host

     # On the remote, verify forwarding and test GitHub
     echo "$SSH_AUTH_SOCK"
     ssh -T [email protected]
     ```
> TODO
> 
> How do you forward SSH keys on Windows?

### 1) Prepare the codebase
Work locally on the host, or connect remotely via SSH to the host.
Choose a workspace location and clone the repository:
```bash
# clone this repo
git clone <THIS_REPO_URL>.git
cd <THIS_REPO_DIRECTORY>
```

### 2) Generate your `.env`
Create a project `.env` with sensible defaults based on your local UID/GID:
```bash
chmod +x gen-dotenv.sh
./gen-dotenv.sh
```
The generated `.env` will include example values such as:
```dotenv
UID=1001
GID=1001
HOST_WS_DIR=/path/to/this/repo
RUNTIME_WS_DIR=/ani_ws
PACKAGE_NAME=auto_needle_insertion
UR_ROBOT_IP=192.168.56.2
USE_MOCK_HARDWARE=true
POLARIS_IP=192.168.56.5
```
- Set `USE_MOCK_HARDWARE=true` to run **without** the physical robot arm.
- For real hardware, set `UR_ROBOT_IP` to your UR controller IP and `POLARIS_IP` to your NDI Polaris IP.

### 3) Build images (first run)
Download and build the images:
```bash
./script/build.sh -f
# It might take a while depending on your internet connection and your machine specification.
```

### 4) Fire up hardware
**Power on devices.** Ensure the **UR arm**, **Franka FR3**, and **NDI Polaris** are all powered on and connected to the lab LAN.

- On the **UR pendant**:

> Click Open -> Installation and select `ani.installation`.
> 
> Click Open -> Program and select `ani_external_control.urp`.
> 
> Enable the robot.

- Ensure the **Franka FR3** is powered on and released from E-stop. On the **Franka Desktop**, unlock the joints and activate FCI.

**Verify network access.** From the host machine, ping all three devices:
```bash
ping -c 3 <UR_ROBOT_IP>
ping -c 3 <FRANKA_ROBOT_IP>
ping -c 3 <POLARIS_IP>
```

### 5) Fire up the dev stack
Bring up the stack and go directly into the dev container bash:
```bash
# If not yet, make the automation scripts executable:
chmod +x ./scripts/*.sh
# Then, just:
./scripts/launch.sh
```
In cases of failed health checks, get container status and logs with:
```bash
# List all containers and their status:
docker compose ps -a
# Then check the log of a container, for example:
docker logs autonomous_needle_insertion-ur5e-driver 
```

### 6) Available scripts and commands in the package
In the dev container shell, you can run standard ROS 2 commands, and control commands or automation scripts from the auto_needle_insertion package.
```bash
# For individual commands other than the automation scripts, first source the package setup file to make bash aware of the package
source install/setup.bash
# Then launch a command
ros2 somecommand ...
```

#### 6.1) Control the robot arm
Run the external control program. On the UR pendant:
> Press the play button to run the `ani_external_control` program.
> 
> If prompted error, check on the pendent in Installation -> URCaps -> External Control if the host IP setting is correct.

```bash
# Some available commands are:
# Run a simple trajectory profile:
ros2 launch auto_needle_insertion move_robot.launch.py mode:=ee_moveit

# Run keyboard control in the end-effector local frame:
ros2 launch auto_needle_insertion move_robot.launch.py mode:=keyboard
```

#### 6.2) Bring up visualization for cameras
```bash
# First, check the available image topics:
ros2 topic list | grep image
# Then, for example, view the raw image from the Polaris Vega VT camera:
ros2 run rqt_image_view rqt_image_view /vega_vt/image_raw
```

#### 6.3) Automation scripts
Begin dataset collection for OpenH:
```bash
./task_scripts/run_openh_collection.sh
```
Run keyboard control of the robotic end effector:
```bash
./task_scripts/run_keyboard_control.sh
```

#### 6.4) Record ROS bag
Start recording a ROS bag:
```bash
ros2 bag record --all
```
Or record specific topics only:
```bash
ros2 bag record <topic1> <topic2> <topic3> ...
```
Check available topics with:
```bash
ros2 topic list
```
Stop recording by pressing `Ctrl+C`.

#### 6.5) Replay ROS bag 
Start recording a ROS bag:
```bash
ros2 bag play <BAGNAME>
```
One may visualize the replay by:
```bash
# first stop the live stream
docker compose stop polaris_camera_driver realsense_driver
# run in the dev container shell
ros2 run rqt_image_view rqt_image_view /vega_vt/image_raw
```

### 7) Stop / Remove
Stop all the containers in one go:
```bash
./scripts/stop.sh
```

## Key ROS 2 topics
| Topic name                 | Data type      | Published by service  | Data source     |
|----------------------------|----------------|-----------------------|-----------------|
| /vega_vt/image_raw         | Video stream   | polaris_camera_driver | Polaris Vega VT |
| /camera/color/image_raw    | Video stream   | realsense_driver      | Realsense       |
| /ati_ft_broadcaster/wrench | Force / Torque | ati_ft_driver         | ATI Axia80-M8   |

## Under the hood
### Repository & runtime layout
- **Docker Compose orchestration.** `docker-compose.yaml` declares several services that you enable via profiles and `.env` flags:
  - **`dev`** – an interactive development container with the ROS 2 workspace mounted at `$RUNTIME_WS_DIR`.
  - **`ur_driver`** – Universal Robots ROS 2 driver for the **physical** UR arm; enabled when `USE_MOCK_HARDWARE=false` and `UR_ROBOT_IP` is reachable.
  - **`ur_driver_mock`** – mock hardware for local development; enabled when `USE_MOCK_HARDWARE=true`.
  - **`polaris_driver`** –  service that publishes tracked tool poses from NDI Polaris; enabled when `POLARIS_IP` is set.
  - **`polaris_camera_driver`** – Service that publishes camera stream from the NDI Polaris Vega VT; enabled when `POLARIS_IP` is set.
  - **`realsense_driver`** – Service that publishes camera stream from Realsense.
  - **`ati_ft_driver`** – Service that publishes force and torque from ATI F/T sensors.
- **Environment configuration.** `gen-dotenv.sh` generates a project `.env` (UID/GID, `HOST_WS_DIR`, `RUNTIME_WS_DIR`, `PACKAGE_NAME`, device IPs, and flags). Docker Compose reads these at launch to parameterize services.
- **ROS 2 workspace.** The workspace mounted at `$RUNTIME_WS_DIR` contains the package `${PACKAGE_NAME}` under development. Within it you will find:
  - **Motion modules** based on *MoveItPy* for trajectory (“profile”) planning/execution, real‑time servo control, and keyboard control.
  - **Calibration utilities** for hand–eye (robot↔tracker) and ultrasound probe/image plane.
  - **Monitoring tools** including a lightweight PoseStamped/TF tool‑pose reporter for downstream consumers.

### How it works (data flow at runtime)
1. **Device interfaces start.** Depending on configuration, the stack brings up either the real **UR driver** (`ur_driver`) or the **mock** (`ur_driver_mock`). If `POLARIS_IP` is provided, the `polaris_driver` connects to the tracker and starts streaming 6‑DoF poses.
2. **Calibrations provide fixed frames.**
   - *Hand–eye calibration* establishes the rigid transform between the robot end effector (or flange) and the optical tracker.
   - *Probe calibration* provides the transform from the optical tracker to the ultrasound **image plane**. These transforms are broadcast as static TF and consumed by planning/servo and visualization nodes.
3. **Motion layers.**
   - *Trajectory (“profile”) mode*: MoveIt 2 plans a collision‑aware path which the UR driver executes.
   - *Keyboard control mode*: Built upon the trajectory mode, get incremental motion in all 6 axis of the end effector local frame.
   - *Real‑time servo mode*: small Cartesian/velocity commands are streamed for compliant, interactive motions.
4. **State reporting.** A pose reporter publishes the tool pose (PoseStamped) and TF so external clients (UI, planners, logging) can subscribe.
5. **Networking.** Drivers talk to hardware over the lab LAN; ROS 2 nodes discover each other inside the Compose network via DDS. No host‑side ROS 2 install is required.

### Platforms and version compatibility
> This project is **tested on Linux**; macOS/Windows are supported via **Docker Desktop**.
- **Host OS**
  - Linux: Ubuntu 22.04 LTS or 24.04 LTS recommended.
  - macOS / Windows: Use Docker Desktop (observed to work in practice; performance depends on your machine).
- **Containerized middleware**
  - **Docker Engine:** 24.x or newer recommended; any recent Engine that supports the Compose v2 plugin works.
  - **Docker Compose v2:** required. The repository uses the modern Compose Specification and CLI (`docker compose ...`).
- **ROS 2 distributions**
  - **Jazzy Jalisco** (primary for Ubuntu 24.04) – supported and recommended for new deployments.
- **Motion stack**
  - **MoveIt 2 / `moveit_py`** for planning and servo from Python.
- **Robot interface**
  - **Universal Robots ROS 2 driver** for CB3/e‑Series controllers (connect using `UR_ROBOT_IP`).
- **Tracking interface**
  - **NDI Polaris (Vega XT/VT/ST)** over the network, publishing 6‑DoF tool frames.
- **Camera interface**
    - **gscam2** for capturing video stream over the network.
- **Ultrasound probe calibration (optional)**
  - Compatible with **PLUS Toolkit** N‑wire phantom–based workflows; the resulting transforms are consumed as static TF.

### Key environment switches
- `USE_MOCK_HARDWARE=true|false` — toggle between mock and physical robot arm.
- `UR_ROBOT_IP` — IP of the UR controller.
- `POLARIS_IP` — IP of the NDI Polaris sensor.
- `HOST_WS_DIR`, `RUNTIME_WS_DIR`, `PACKAGE_NAME` — host repo path, in-container workspace path, and package name used inside the dev container.
 
### Service lifecycle & health
- Bring services up with `docker compose up --profile dev -d`; health checks ensure dependencies are ready.
- If a health check fails, inspect the affected container’s logs to diagnose the problem (for example: `docker compose logs -f <service>`).

## For administrators
This section explains how to prepare a **host machine** to run the Dockerized ROS 2 workspace reliably.
A Linux lab box is recommended; macOS/Windows hosts work via Docker Desktop.

### 1) Choose a host platform
**Linux** is recommended. Choose Ubuntu 22.04 LTS or 24.04 LTS for best compatibility with Docker.
> Project‑level recommendations (not hard requirements): ≥ 4 physical cores (8 threads), 16 GB RAM, and ≥ 50 GB free SSD space for images/logs. Wired Ethernet LAN to your device is strongly preferred.

### 2) Install Git and Docker
Install Git, Docker Engine and the Compose v2 plugin, then check with:
```bash
docker compose version   # should print a v2.x version
```

### 3) Remote access via Tailscale
Add the machine to your Tailscale network.
From the Tailscale admin console, **invite users** (or generate an **invite link**) and share it with the intended end users so they can join the tailnet with their own accounts.

### 4) Create user account
Create a local account for each person who will use the box:
```bash
sudo useradd -m -s /bin/bash <username>
sudo passwd <username>
```
To require the user to change that password at first login:
```bash
# Either of the following works
sudo chage -d 0 <username>
# or
sudo passwd -e <username>
```
Grant Docker access so they can run `docker` without `sudo`:
```bash
sudo usermod -aG docker <username>
```
> The user must **log out and back in** (or reboot) for new group membership to take effect.

> TODO
> 
> Add a bash script to automate user creation and Docker group assignment.

### 5) Grant users in docker group with write permission
The recorded ros2 bags can be stored in a hard drive.
Let's say the drive is mounted as /mnt/dataset.
In that mount point:
```bash
# Create a directory to house the bags
sudo mkdir -p rosbag_recording
# Set permissions so that all users in the docker group can write to it
sudo chgrp docker /mnt/dataset/rosbag_recording
sudo chmod 2775 /mnt/dataset/rosbag_recording
```

### 6) Hardware acceleration for sensor image rendering
Rendering RTSP videos in `rqt_image_raw` may encounter freezing problems.
Using hardware accelerated GL may help with performance.
The current settings are compatible with X11 only,meaning the users should log in with Xorg.
To force Xorg as the default session for all users:
```bash
# Edit this file
sudo vim /etc/gdm3/custom.conf
# Then remove the comment for
WaylandEnable=false
```

## Troubleshooting
- **Polaris camera streaming failure:** Reset Polaris to factory setting, and remember to change the IP back to the value in your .env file.
- **USB video grabber failure:** The device name (/dev/video*) of the usb video grabber is not consistent across reboots. You need to plug & replug the device to figure out the exact device name, then make corresponding changes in .env.
