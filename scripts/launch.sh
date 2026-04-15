#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DEV_CONTAINER="autonomous_needle_insertion-dev"
DEV_IMAGE="aniros:jazzy-dev"
DOCKERFILE_PATH="Dockerfile"

image_needs_rebuild() {
  # Rebuild is required when the target image is missing.
  if ! docker image inspect "${DEV_IMAGE}" >/dev/null 2>&1; then
    return 0
  fi

  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    return 1
  fi

  local dockerfile_mtime
  dockerfile_mtime=$(stat -f '%m' "${DOCKERFILE_PATH}")

  local image_created_epoch
  image_created_epoch=$(docker image inspect --format '{{.Created}}' "${DEV_IMAGE}" \
    | python3 -c 'import datetime, sys; s=sys.stdin.read().strip(); print(int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()))')

  [[ "${dockerfile_mtime}" -gt "${image_created_epoch}" ]]
}

confirm_rebuild() {
  local answer
  while true; do
    read -r -p "Container image may be outdated. Rebuild now? [y/N]: " answer
    case "${answer}" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      ""|[Nn]|[Nn][Oo]) return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

print_help() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Start (if needed) the ROS2 development stack and attach to the dev container shell.

Behaviour:
  - If container '${DEV_CONTAINER}' is already running:
        Directly attach an interactive bash shell inside the container.
  - If container '${DEV_CONTAINER}' is not running:
        0. Check whether image '${DEV_IMAGE}' may be outdated (missing or older than Dockerfile).
           If outdated, ask whether to rebuild it first.
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

if image_needs_rebuild; then
  echo "Detected that '${DEV_IMAGE}' may need a rebuild before launch."

  if [[ -t 0 ]]; then
    if confirm_rebuild; then
      echo "Rebuilding '${DEV_IMAGE}'..."
      ./scripts/build.sh
    else
      echo "Skipping rebuild and continuing with existing image."
    fi
  else
    echo "Non-interactive shell detected; skipping rebuild prompt and continuing without rebuild." >&2
  fi

  echo
fi

# Allow local X11 clients (ignore failure but warn)
xhost +local: || echo "Warning: xhost +local: failed (X11 may not work inside the container)" >&2
echo

echo "Starting dev stack with docker compose..."
docker compose --profile dev up -d
echo

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