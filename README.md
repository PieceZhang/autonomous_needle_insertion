# Autonomous Needle Insertion Guided by Robotic Ultrasound

## Overview
This repository provides a reproducible, Dockerized ROS 2 workspace for autonomous needle insertion R&D guided by robotic ultrasound.
It integrates a Universal Robots (UR) arm (hardware or mock), an NDI Polaris optical tracker, and an ultrasound system.
The workspace includes MoveItPy‑based motion control in both trajectory (“profile”) and real‑time servo modes, along with calibration and monitoring utilities.

**Key features**
- End‑effector profile‑tracing with MoveItPy.
- Hand–eye calibration (UR + NDI Polaris) using OpenCV with residual checks.
- Lightweight PoseStamped/TF tool‑pose reporter.
- Docker Compose services:
  - `ur_driver` (UR5e hardware)
  - `ur_driver_mock` (mock hardware)
  - `polaris_driver` (NDI)
  - `dev` (interactive shell with the workspace mounted)

End users can jump to **Quick start** to clone the repo, generate a `.env`, and bring up the stack. Administrators can refer to **For administrators** for host setup details.

## Key concepts

| Term                       | What it means here                                                                                                             |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Host**                   | The physical machine where Docker runs and this repository is cloned.                                                          |
| **Container**              | An isolated runtime for a packaged app (an image) used for dev and runtime reproducibility.                                    |
| **Docker Compose profile** | A named grouping of services you activate selectively (e.g., `--profile dev`).                                                 |
| **`.env` file**            | Key–value config read by Compose to parameterize services (e.g., `WS_DIR`, `UR_ROBOT_IP`).                                     |
| **ROS 2 workspace**        | The mounted colcon workspace at `$WS_DIR` containing your packages.                                                            |
| **Motion profile**         | Plan a trajectory with MoveIt and execute it (as opposed to continuous servoing).                                              |
| **Hand–eye calibration**   | The fixed transform between the robot “hand” and the “eye” (camera/tracker), typically via OpenCV’s `calibrateHandEye`.        |
| **Probe calibration**      | The fixed transform between the ultrasound imaging plane and the tracker, typically via the PLUS Toolkit and a N-wire phantom. |
| **UR ROS 2 driver**        | The official ROS 2 driver for Universal Robots arm (e.g., UR5e); `UR_ROBOT_IP` points to the controller.                       |
| **Mock hardware**          | Run the stack without real devices (simulated/mock drivers only) by setting `USE_MOCK_HARDWARE=true`.                          |
| **NDI Polaris**            | Optical tracking system reporting tool poses in real time; `POLARIS_IP` is its host.                                           |
| **Tailscale / tailnet**    | Mesh VPN used to reach lab devices remotely; a **tailnet** is the Tailscale network.                                           |

## Quick start

Run this project with Docker Compose, and no host-side ROS 2 install is required.
It is **tested on Linux** and is expected to work on macOS/Windows via Docker Desktop.

### 0) Access (required for lab devices)
**Host access.** Request a personal login on the lab host from the administrator.

**Remote access (optional).** If you will reach devices over the lab tailnet:
  1. Create your own Tailscale account and request a **Tailscale invite link** to the team tailnet.
  2. Install Tailscale on your own device and join the tailnet. You can verify connectivity with:
     ```bash
     tailscale status
     tailscale ip -4   # optional: note your Tailscale IPv4 address
     ```

### 1) Prepare the codebase
Choose a workspace location and clone the repository:
```bash
mkdir -p ~/ani_ws && cd ~/ani_ws
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
WS_DIR=/ani_ws
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
docker compose build
# It might take a while depending on your internet connection.
```

### 4) Start services
Bring up the stack in the background:
```bash
docker compose up --profile dev -d
```
and wait for all the container health checks to pass.
Check container status and logs with:
```bash
docker compose ps -a
docker compose logs -f
```

### 5) Open a development shell
Get inside the container shell:
```bash
docker compose exec -it auto_needle_insertion bash
```
Run ROS 2 commands inside the container:
```bash
ros2 topic list
```
Then start your experiments or development.

### 6) Stop / Remove
Stop all the containers in one go:
```bash
TODO update command
docker compose down
```

## Under the hood
### Repository & runtime layout
- **Docker Compose orchestration.** `compose.yaml` declares four services that you enable via profiles and `.env` flags:
  - **`dev`** – an interactive development container with the ROS 2 workspace mounted at `$WS_DIR`.
  - **`ur_driver`** – Universal Robots ROS 2 driver for **physical** UR arms; enabled when `USE_MOCK_HARDWARE=false` and `UR_ROBOT_IP` is reachable.
  - **`ur_driver_mock`** – mock hardware for local development; enabled when `USE_MOCK_HARDWARE=true`.
  - **`polaris_driver`** – NDI Polaris client that publishes tracked tool poses; enabled when `POLARIS_IP` is set.
- **Environment configuration.** `gen-dotenv.sh` generates a project `.env` (UID/GID, `WS_DIR`, `PACKAGE_NAME`, device IPs, and flags). Docker Compose reads these at launch to parameterize services.
- **ROS 2 workspace.** The workspace mounted at `$WS_DIR` contains your package `${PACKAGE_NAME}`. Within it you will find:
  - **Motion modules** based on *MoveItPy* for both trajectory (“profile”) planning/execution and real‑time servo control.
  - **Calibration utilities** for hand–eye (robot↔tracker) and ultrasound probe/image plane.
  - **Monitoring tools** including a lightweight PoseStamped/TF tool‑pose reporter for downstream consumers.

### How it works (data flow at runtime)
1. **Device interfaces start.** Depending on configuration, the stack brings up either the real **UR driver** (`ur_driver`) or the **mock** (`ur_driver_mock`). If `POLARIS_IP` is provided, the `polaris_driver` connects to the tracker and starts streaming 6‑DoF poses.
2. **Calibrations provide fixed frames.**
   - *Hand–eye calibration* establishes the rigid transform between the robot end effector (or flange) and the optical tracker.
   - *Probe calibration* provides the transform from the optical tracker to the ultrasound **image plane**. These transforms are broadcast as static TF and consumed by planning/servo and visualization nodes.
3. **Motion layers.**
   - *Trajectory (“profile”) mode*: MoveIt 2 plans a collision‑aware path which the UR driver executes.
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
- **Ultrasound probe calibration (optional)**
  - Compatible with **PLUS Toolkit** N‑wire phantom–based workflows; the resulting transforms are consumed as static TF.

### Key environment switches
- `USE_MOCK_HARDWARE=true|false` — toggle between mock and physical robot arm.
- `UR_ROBOT_IP` — IP of the UR controller.
- `POLARIS_IP` — IP of the NDI Polaris sensor.
- `WS_DIR`, `PACKAGE_NAME` — ROS 2 workspace mount and package name used inside the dev container.
 
### Service lifecycle & health
- Bring services up with `docker compose up --profile dev -d`; health checks ensure dependencies are ready.
- If a health check fails, inspect the affected container’s logs to diagnose the problem (for example: `docker compose logs -f <service>`).
- Stop and remove containers with `docker compose down` when finished.

## For administrators
This section explains how to prepare a **host machine** to run the Dockerized ROS 2 workspace reliably.
A Linux lab box is recommended; macOS/Windows hosts work via Docker Desktop.

### 1) Choose a host platform
**Linux** is recommended. Choose Ubuntu 22.04 LTS or 24.04 LTS for best compatibility with ROS 2 and Docker.
> Project‑level recommendations (not hard requirements): ≥4 physical cores (8 threads), 16 GB RAM, and ≥50 GB free SSD space for images/logs. Wired Ethernet to your device LAN is strongly preferred.

### 2) Install Git and Docker
Install Git, Docker Engine and the Compose v2 plugin, then allow your user to run Docker without root:
```bash
sudo usermod -aG docker "$USER" && newgrp docker
docker compose version   # should print a v2.x version
```

### 3) Remote access via Tailscale
Add the machine to your Tailscale network.
From the Tailscale admin console, **invite users** (or generate an **invite link**) and share it with the intended end users so they can join the tailnet with their own accounts.

### Troubleshooting
- **Permission denied (Linux):** ensure your user is in the `docker` group (see Tip above), then log out/in or run `newgrp docker`.
- **Compose not found:** verify `docker compose version` prints a v2.x version. If not, install the Compose plugin for your distro.
