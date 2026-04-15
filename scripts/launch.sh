#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
STARTUP_TIMEOUT_SEC=180
PROFILE="dev"
ATTACH_SERVICE=""

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

resolve_attach_service() {
  local profile="$1"
  local override="${LAUNCH_ATTACH_SERVICE:-}"
  local raw_services=""
  local service=""
  local -a services=()
  local -a dev_services=()

  if [[ -n "${override}" ]]; then
    printf '%s' "${override}"
    return 0
  fi

  if ! raw_services="$(docker compose --profile "${profile}" config --services 2>/dev/null)"; then
    echo "Error: failed to discover services for profile '${profile}' via docker compose." >&2
    return 1
  fi

  mapfile -t services < <(printf '%s\n' "${raw_services}" | sed '/^[[:space:]]*$/d')
  if [[ ${#services[@]} -eq 0 ]]; then
    echo "Error: no services found for profile '${profile}'." >&2
    return 1
  fi

  for service in "${services[@]}"; do
    if [[ "${service}" == "${profile}" ]]; then
      printf '%s' "${service}"
      return 0
    fi
  done

  for service in "${services[@]}"; do
    if [[ "${service}" == "${profile}-dev" ]]; then
      printf '%s' "${service}"
      return 0
    fi
  done

  for service in "${services[@]}"; do
    if [[ "${service}" == "dev" ]]; then
      printf '%s' "${service}"
      return 0
    fi
  done

  for service in "${services[@]}"; do
    if [[ "${service}" == *-dev ]]; then
      dev_services+=("${service}")
    fi
  done

  if [[ ${#dev_services[@]} -eq 1 ]]; then
    printf '%s' "${dev_services[0]}"
    return 0
  fi

  if [[ ${#services[@]} -eq 1 ]]; then
    printf '%s' "${services[0]}"
    return 0
  fi

  echo "Error: could not determine attach service for profile '${profile}'." >&2
  echo "Set LAUNCH_ATTACH_SERVICE=<service> to override." >&2
  return 1
}
DEV_CONTAINER="autonomous_needle_insertion-dev"
DEV_IMAGE="aniros:jazzy-dev"
DOCKERFILE_PATH="Dockerfile"

get_file_mtime_epoch() {
  local file_path="$1"

  # GNU coreutils (Linux)
  if stat -c '%Y' "${file_path}" >/dev/null 2>&1; then
    stat -c '%Y' "${file_path}"
    return 0
  fi

  # BSD/macOS
  if stat -f '%m' "${file_path}" >/dev/null 2>&1; then
    stat -f '%m' "${file_path}"
    return 0
  fi

  # Portable fallback
  python3 - "${file_path}" <<'PY'
import os
import sys

print(int(os.path.getmtime(sys.argv[1])))
PY
}

image_needs_rebuild() {
  # Rebuild is required when the target image is missing.
  if ! docker image inspect "${DEV_IMAGE}" >/dev/null 2>&1; then
    return 0
  fi

  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    return 1
  fi

  local dockerfile_mtime
  dockerfile_mtime=$(get_file_mtime_epoch "${DOCKERFILE_PATH}" 2>/dev/null || true)
  if [[ -z "${dockerfile_mtime}" || ! "${dockerfile_mtime}" =~ ^[0-9]+$ ]]; then
    echo "Warning: unable to read mtime for '${DOCKERFILE_PATH}', skipping rebuild check." >&2
    return 1
  fi

  local image_created_epoch
  image_created_epoch=$(docker image inspect --format '{{.Created}}' "${DEV_IMAGE}" \
    | python3 -c 'import datetime, sys; s=sys.stdin.read().strip(); print(int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()))' \
    || true)
  if [[ -z "${image_created_epoch}" || ! "${image_created_epoch}" =~ ^[0-9]+$ ]]; then
    echo "Warning: unable to parse image creation time for '${DEV_IMAGE}', skipping rebuild check." >&2
    return 1
  fi

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
Usage: ${SCRIPT_NAME} [OPTIONS] [PROFILE]

Start (if needed) the ROS2 development stack and attach to a shell service.

Behaviour:
  - Resolve an attach service from the selected profile (or use LAUNCH_ATTACH_SERVICE).
  - If that service is already running:
        Directly attach an interactive bash shell.
  - If that service is not running:
        0. Check whether image '${DEV_IMAGE}' may be outdated (missing or older than Dockerfile).
           If outdated, ask whether to rebuild it first.
        1. Allow local X11 clients via 'xhost +local:'.
        2. Run 'docker compose --profile <profile> up -d' (defaults to 'dev').
        3. Check service container status.
        4. If running, attach an interactive bash shell.

Options:
  -h, --help    Show this help message and exit.

Environment:
  LAUNCH_ATTACH_SERVICE   Force the service name to attach to.

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} dev
  ${SCRIPT_NAME} franka-test
  ${SCRIPT_NAME} -h

EOF
}

# Argument parsing: optional PROFILE plus -h / --help
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      print_help
      exit 1
      ;;
    *)
      if [[ "${PROFILE}" != "dev" ]]; then
        echo "Error: only one optional PROFILE argument is supported." >&2
        print_help
        exit 1
      fi
      PROFILE="$1"
      ;;
  esac
  shift
done

if [[ -z "${PROFILE}" ]]; then
  echo "Error: PROFILE cannot be empty." >&2
  exit 1
fi

ATTACH_SERVICE="$(resolve_attach_service "${PROFILE}")"
echo "Using attach service '${ATTACH_SERVICE}' for profile '${PROFILE}'."

# Check whether the attach service container is already running
existing_cid="$(docker compose ps -q "${ATTACH_SERVICE}" 2>/dev/null || true)"

if [[ -n "$existing_cid" ]]; then
  existing_status="$(docker inspect -f '{{.State.Status}}' "$existing_cid" 2>/dev/null || true)"
  if [[ "$existing_status" == "running" || "$existing_status" == "restarting" ]]; then
    echo "Service '${ATTACH_SERVICE}' is already ${existing_status}. Attaching to shell..."
    exec docker compose exec "${ATTACH_SERVICE}" bash
  fi
fi

echo "Service '${ATTACH_SERVICE}' is not running. Starting stack..."

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

echo "Starting stack with docker compose profile '${PROFILE}'..."
docker compose --profile "${PROFILE}" up -d
echo

echo "Waiting for driver services to be ready..."
if [[ "${PROFILE}" == "dev" ]]; then
  echo "PROFILE='${PROFILE}': waiting for full driver set."
  wait_for_service_ready ur_driver
  wait_for_service_ready franka_driver
  wait_for_service_ready polaris_driver
  wait_for_service_ready polaris_camera_driver
  wait_for_service_ready ati_ft_driver
  wait_for_service_ready keyboard_driver
else
  echo "PROFILE='${PROFILE}': waiting only for franka_driver."
  wait_for_service_ready franka_driver
fi

echo "docker compose up -d completed. Checking attach service status..."

# After compose up, obtain container ID again
cid="$(docker compose ps -q "${ATTACH_SERVICE}" 2>/dev/null || true)"

if [ -z "$cid" ]; then
  echo "Error: service '${ATTACH_SERVICE}' is not created or not running."
  exit 1
fi

# Check basic container status
status=$(docker inspect -f '{{.State.Status}}' "$cid")

# Treat 'running' and 'restarting' as acceptable
if [ "$status" != "running" ] && [ "$status" != "restarting" ]; then
  echo "Error: service '${ATTACH_SERVICE}' is in unexpected state: ${status}"
  exit 1
fi

echo "Service '${ATTACH_SERVICE}' is ${status}. Attaching to shell..."
exec docker compose exec "${ATTACH_SERVICE}" bash
