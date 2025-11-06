# Autonomous Needle Insertion Guided by Robotic Ultrasound

## Overview
This repository provides a Dockerized ROS 2 workspace for autonomous needle insertion research and development, integrating a UR robot (hardware or mock), an NDI Polaris tracker, and an ultrasound machines.
It includes MoveItPy-based motion controls in profile and servo mode, as well as several calibration routines.

This project requires (a) private network access to the lab resources via **Tailscale**, and (b) a **Docker** runtime with **Docker Compose v2**.
The required actions differ slightly depending on your role.

## Features
- End-effector profile tracing demo with MoveItPy.
- Hand–eye calibration (UR + NDI Polaris) using OpenCV’s calibrateHandEye with residual checks.
- Simple PoseStamped/TF tool pose reporter.
- Docker Compose services for:
  - ur_driver (UR5e hardware)
  - ur_driver_mock (mock hardware)
  - polaris_driver (NDI)
  - dev (interactive shell with the workspace mounted)

## Quick start

The project runs with **Docker + Docker Compose**. This avoids installing ROS 2 and other dependencies on your host. It should work on Linux, macOS, and Windows, but it is currently only tested on Linux.
For **end users**, follow these steps to quickly get started.

### Get access
#### 1) Get host access
Request a personal user account from the administrator on the host machine.

#### 2) Get remote access
If you want remote access to lab devices:
  - Get your own personal Tailscale account.
  - Request a **Tailscale invite link** to the team tailnet so your machine can connect to the lab network.
  - Install Tailscale on your own machine and sign in with your account. Open the invite link and accept it so your device joins the tailnet. If you prefer CLI, run:
    ```bash
    sudo tailscale up
    ```
  - Verify connectivity. Confirm your device is connected and visible:
      ```bash
      tailscale status
      tailscale ip -4   # optional: note your Tailscale IPv4 address
      ```

### Prepare the codebase
#### 1) Clone the repository
  ```bash
  # choose a location for the workspace (any path is fine)
  mkdir -p ~/ani_ws && cd ~/ani_ws
  # clone this repo
  git clone <THIS_REPO_URL>.git
  cd <THIS_REPO_DIRECTORY>
  ```

#### 2) Generate your `.env`
This project uses a `.env` file to configure the docker containers.
Use the helper script to create one with sane defaults based on your local UID/GID:
  ```bash
  chmod +x gen-dotenv.sh
  ./gen-dotenv.sh
  ```

The generated `.env` will include the following example values:
```dotenv
UID=1001
GID=1001
WS_DIR=/ani_ws
PACKAGE_NAME=auto_needle_insertion
UR_ROBOT_IP=192.168.56.2
USE_MOCK_HARDWARE=true
POLARIS_IP=192.168.56.5
```

- Set `USE_MOCK_HARDWARE=true` to run **without** any actual robot arms.
- When using real hardware, set `UR_ROBOT_IP` to your UR controller’s IP and `POLARIS_IP` to your NDI Polaris host IP.

### Start the stack
For the first run ever, download and build images with:
```bash
docker compose build
```
It might take a while depending on your internet connection.

Bring up all services in the background with:
```bash
docker compose up --profile dev -d
```
and wait for all the containers to start. Check the status, health and logs with:
```bash
docker compose ps -a
docker compose logs -f
```

### Get into the container shell
Use this if you want to run ROS 2 commands, build, or inspect the environment.
```bash
# replace <SERVICE> with your dev service name (commonly `dev` or `ros`)
docker compose exec -it auto_needle_insertion bash
# inside the container you can run, for example:
ros2 topic list
```
Then start your experiments or development.

### Stop / Remove
```bash
# stop containers but keep data/volumes
docker compose down
```

## Under the hood
### What gets started?
- A ROS 2 workspace mounted at `$WS_DIR` containing the package `${PACKAGE_NAME}`.
- Interfaces for real devices **only** if `USE_MOCK_HARDWARE=false` and IPs are set (`UR_ROBOT_IP`, `POLARIS_IP`).
- Mock drivers and demos if `USE_MOCK_HARDWARE=true`.

## For administrators

### 0) Prerequisites
- **Docker** (Engine on Linux or Desktop on macOS/Windows)
- **Docker Compose v2** (bundled with Docker Desktop; on Linux install the `docker-compose-plugin`)
- **Git**

> Administration Tip (Linux): after installing Docker, add your user to the `docker` group so you can run Docker without `sudo`:
>
> ```bash
> sudo usermod -aG docker "$USER" && newgrp docker
> docker compose version   # should print a v2.x version
> ```
>  Ignore this step if you use a machine that is already configured for Docker.

#### If you are the **tailnet administrator**
1. Prepare a machine with decent specs for the team.
2. **Add the machine to your Tailscale network first.** Install the Tailscale client, authenticate the device into your tailnet, and confirm visibility:
   ```bash
   sudo tailscale up
   tailscale status
   ```
   From the Tailscale admin console, **invite users** (or generate an **invite link**) and share it with the intended end users so they can join the tailnet with their own accounts.
3. **Then install Docker and Compose v2.** Once the machine is visible in Tailscale, install **Docker** (Docker Desktop on macOS/Windows, or Docker Engine + Compose plugin on Linux). On Linux, ensure the intended local user is in the `docker` group and verify Compose v2:
   ```bash
   sudo usermod -aG docker <username>
   # user should re-login or run `newgrp docker` to apply the new group
   docker compose version
   ```
4. **Ensure Git is available** so the repository can be cloned.

4. **Install Docker and Compose v2.**
    - **macOS / Windows:** Install **Docker Desktop** (includes Compose v2).
    - **Linux:** Install **Docker Engine** and the **Docker Compose plugin**, then allow your user to run Docker without root and verify Compose v2:
      ```bash
      sudo apt-get update
      sudo apt-get install -y docker.io docker-compose-plugin
      sudo usermod -aG docker "$USER" && newgrp docker
      docker compose version
      ```
      *(Adjust the package names/commands for your distribution as needed.)*
5. **Install Git.** Ensure `git` is available on your system to clone this repository.

### Troubleshooting
- **Permission denied (Linux):** ensure your user is in the `docker` group (see Tip above), then log out/in or run `newgrp docker`.
- **Compose not found:** verify `docker compose version` prints a v2.x version. If not, install the Compose plugin for your distro.
