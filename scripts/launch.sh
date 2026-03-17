#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEV_CONTAINER="autonomous_needle_insertion-dev"
STARTUP_TIMEOUT_SEC=180

wait_for_service_ready() {
  local service="$1"
  local timeout="${2:-$STARTUP_TIMEOUT_SEC}"
  local elapsed=0
  local cid=""
  local state=""
  local health=""

  while (( elapsed < timeout )); do
    cid="$(docker compose ps -q "${service}" 2>/dev/null || true)"
    if [[ -n "${cid}" ]]; then
      state="$(docker inspect -f '{{.State.Status}}' "${cid}" 2>/dev/null || true)"
      health="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "${cid}" 2>/dev/null || true)"
      if [[ "${health}" == "healthy" ]]; then
        echo "Service '${service}' is healthy."
        return 0
      fi
      if [[ -z "${health}" && "${state}" == "running" ]]; then
        echo "Service '${service}' is running (no healthcheck configured)."
        return 0
      fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "Warning: timed out waiting for service '${service}' (state='${state}', health='${health}')." >&2
  return 1
}

print_help() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Start (if needed) the ROS2 development stack and attach to the dev container shell.

Behaviour:
  - If container '${DEV_CONTAINER}' is already running:
        Directly attach an interactive bash shell inside the container.
  - If container '${DEV_CONTAINER}' is not running:
        1. Allow local X11 clients via 'xhost +local:'.
        2. Run 'docker compose --profile dev up -d'.
        3. Check container status.
        4. If running, attach an interactive bash shell inside the container.

Options:
  -h, --help    Show this help message and exit.

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} -h

EOF
}

# Argument parsing (currently only supports -h / --help)
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  print_help
  exit 0
fi

# Check whether the dev container is already running
existing_cid=$(docker ps -q --filter "name=^/${DEV_CONTAINER}$" || true)

if [[ -n "$existing_cid" ]]; then
  echo "Container '${DEV_CONTAINER}' is already running. Attaching to shell..."
  exec docker exec -it "${DEV_CONTAINER}" bash
fi

echo "Container '${DEV_CONTAINER}' is not running. Starting dev stack..."

# Allow local X11 clients (ignore failure but warn)
xhost +local: || echo "Warning: xhost +local: failed (X11 may not work inside the container)" >&2
echo

echo "Starting dev stack with docker compose..."
docker compose --profile dev up -d
echo

echo "Waiting for driver services to be ready..."
wait_for_service_ready ur_driver
wait_for_service_ready franka_driver
wait_for_service_ready polaris_driver
wait_for_service_ready polaris_camera_driver
wait_for_service_ready ati_ft_driver
wait_for_service_ready keyboard_driver

echo "docker compose up -d completed. Checking dev container status..."

# After compose up, obtain container ID again
cid=$(docker ps -q --filter "name=^/${DEV_CONTAINER}$" || true)

if [ -z "$cid" ]; then
  echo "Error: container '${DEV_CONTAINER}' is not created or not running."
  exit 1
fi

# Check basic container status
status=$(docker inspect -f '{{.State.Status}}' "$cid")

# Treat 'running' and 'restarting' as acceptable
if [ "$status" != "running" ] && [ "$status" != "restarting" ]; then
  echo "Error: container '${DEV_CONTAINER}' is in unexpected state: ${status}"
  exit 1
fi

echo "Container '${DEV_CONTAINER}' is ${status}. Attaching to shell..."
exec docker exec -it "${DEV_CONTAINER}" bash
