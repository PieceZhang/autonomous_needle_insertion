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
# All images built from the shared Dockerfile (one per stage)
ALL_IMAGES=("aniros-app:jazzy" "aniros-ndi:jazzy" "aniros-franka:jazzy")
# Map profiles to the subset of images they actually need
declare -A PROFILE_IMAGES=(
  [dev]="aniros-app:jazzy aniros-ndi:jazzy"
  [default]="aniros-app:jazzy aniros-ndi:jazzy aniros-franka:jazzy"
  [franka-test]="aniros-franka:jazzy"
  [ur_test]="aniros-app:jazzy"
  [polaris_test]="aniros-ndi:jazzy"
  [gscam2-test]="aniros-ndi:jazzy"
  [ati_test]="aniros-app:jazzy"
)
# Map image name → build.sh target name
declare -A IMAGE_TO_TARGET=(
  [aniros-app:jazzy]="app"
  [aniros-ndi:jazzy]="ndi"
  [aniros-franka:jazzy]="franka"
)
DOCKERFILE_PATH="Dockerfile"
# Directory storing per-target build stamps (one file per target).
# Each stamp file records the Dockerfile mtime at the time that target was
# last successfully built, so we can tell which targets are actually stale.
BUILD_STAMP_DIR=".build_stamps"
# Populated by image_needs_rebuild(); lists build targets that are stale
STALE_TARGETS=()

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

# Check whether a single image needs rebuilding.
# An image is stale if:
#   - it doesn't exist in Docker at all, OR
#   - its build stamp file is missing (never built via build.sh), OR
#   - the Dockerfile mtime recorded in the stamp is older than the current mtime
single_image_needs_rebuild() {
  local image="$1"
  local dockerfile_mtime="$2"
  local target="${IMAGE_TO_TARGET[$image]}"
  local stamp_file="${BUILD_STAMP_DIR}/${target}"

  # Image missing entirely
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "  ${image}: missing"
    return 0
  fi

  # No stamp file → we don't know when this target was last built against
  # the current Dockerfile, so conservatively treat it as stale
  if [[ ! -f "${stamp_file}" ]]; then
    echo "  ${image}: no build stamp (run build.sh to create one)"
    return 0
  fi

  # Compare the Dockerfile mtime recorded in the stamp vs current mtime
  local stamp_mtime
  stamp_mtime=$(<"${stamp_file}")
  if [[ -z "${stamp_mtime}" || ! "${stamp_mtime}" =~ ^[0-9]+$ ]]; then
    echo "  ${image}: invalid build stamp, treating as stale"
    return 0
  fi

  if [[ "${dockerfile_mtime}" -gt "${stamp_mtime}" ]]; then
    echo "  ${image}: outdated (Dockerfile changed since last build)"
    return 0
  fi

  return 1
}

# Check all images relevant to the current profile.
# Returns 0 (needs rebuild) if ANY image is missing or outdated.
# Populates STALE_TARGETS with the build target names that need rebuilding.
image_needs_rebuild() {
  STALE_TARGETS=()

  if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
    return 1
  fi

  local dockerfile_mtime
  dockerfile_mtime=$(get_file_mtime_epoch "${DOCKERFILE_PATH}" 2>/dev/null || true)
  if [[ -z "${dockerfile_mtime}" || ! "${dockerfile_mtime}" =~ ^[0-9]+$ ]]; then
    echo "Warning: unable to read mtime for '${DOCKERFILE_PATH}', skipping rebuild check." >&2
    return 1
  fi

  # Determine which images to check for the selected profile
  local images_str="${PROFILE_IMAGES[${PROFILE}]:-}"
  local -a images_to_check=()
  if [[ -n "${images_str}" ]]; then
    read -ra images_to_check <<< "${images_str}"
  else
    images_to_check=("${ALL_IMAGES[@]}")
  fi

  for img in "${images_to_check[@]}"; do
    if single_image_needs_rebuild "${img}" "${dockerfile_mtime}"; then
      STALE_TARGETS+=("${IMAGE_TO_TARGET[$img]}")
    fi
  done

  [[ ${#STALE_TARGETS[@]} -gt 0 ]]
}

# After a selective rebuild, refresh stamps for images that were NOT rebuilt
# but still exist.  Since they share the same Dockerfile and Docker's layer
# cache guarantees unchanged stages produce identical images, they are valid.
refresh_untouched_stamps() {
  local dockerfile_mtime
  dockerfile_mtime=$(get_file_mtime_epoch "${DOCKERFILE_PATH}" 2>/dev/null || true)
  [[ -z "${dockerfile_mtime}" ]] && return

  mkdir -p "${BUILD_STAMP_DIR}"
  for img in "${ALL_IMAGES[@]}"; do
    local tgt="${IMAGE_TO_TARGET[$img]}"
    # Skip targets that were just rebuilt (build.sh already wrote their stamps)
    local was_rebuilt=false
    for st in "${STALE_TARGETS[@]}"; do
      [[ "${st}" == "${tgt}" ]] && { was_rebuilt=true; break; }
    done
    "${was_rebuilt}" && continue

    # Only refresh if the image actually exists in Docker
    if docker image inspect "${img}" >/dev/null 2>&1; then
      printf '%s\n' "${dockerfile_mtime}" > "${BUILD_STAMP_DIR}/${tgt}"
    fi
  done
}

confirm_rebuild() {
  local answer
  while true; do
    read -r -p "One or more container images may be outdated. Rebuild now? [y/N]: " answer
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
        0. Check whether any required image (aniros-app, aniros-ndi, aniros-franka)
           is missing or older than the Dockerfile.  If so, ask whether to rebuild.
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
  echo "The following target(s) need rebuilding for profile '${PROFILE}': ${STALE_TARGETS[*]}"

  if [[ -t 0 ]]; then
    if confirm_rebuild; then
      # Build only the stale targets
      build_flags=(-p)
      for tgt in "${STALE_TARGETS[@]}"; do
        build_flags+=(-t "${tgt}")
      done
      echo "Rebuilding: ${STALE_TARGETS[*]} …"
      ./scripts/build.sh "${build_flags[@]}"
      # build.sh writes stamps for rebuilt targets.  For non-rebuilt images
      # that still exist, refresh their stamps too — Docker's layer cache
      # guarantees they are still valid (a rebuild would be a no-op).
      refresh_untouched_stamps
    else
      echo "Skipping rebuild and continuing with existing images."
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

echo "Waiting for attach service '${ATTACH_SERVICE}' to be ready..."
# Keep readiness source-of-truth in docker-compose.yml via depends_on: condition: service_healthy.
wait_for_service_ready "${ATTACH_SERVICE}"

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
