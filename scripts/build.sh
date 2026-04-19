#!/usr/bin/env bash
set -euo pipefail

# BuildKit is required for --mount=type=cache in Dockerfile
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

help() {
  cat <<'EOF'
Usage:
  build.sh              Build all three images (app, ndi, franka)
  build.sh -t TARGET    Build only the specified target(s) (app | ndi | franka)
  build.sh -f           Force a clean rebuild (no cache)
  build.sh -p           Build targets in parallel
  build.sh -m           Skip mirror probing (use default mirrors)

Options:
  -t TARGET   Build a Dockerfile stage: app, ndi, or franka.
              Can be repeated: -t app -t ndi
  -f          Force a clean rebuild (no cache)
  -p          Build targets in parallel
  -m          Skip mirror speed-test; use archive.ubuntu.com + packages.ros.org
  -h          Show this help

Examples:
  build.sh                       # sequential build of all three images
  build.sh -t ndi                # rebuild only the NDI/Polaris image
  build.sh -t app -t ndi         # rebuild app and ndi images
  build.sh -t franka -f          # force-rebuild the Franka image
  build.sh -p                    # parallel build of all images
  build.sh -m                    # skip mirror probing for faster builds
  build.sh -t app -t franka -p   # parallel build of app + franka only
EOF
}

force_no_cache=false
targets=()
parallel=false
skip_mirrors=false

while getopts ":t:fpmh" opt; do
  case "${opt}" in
    t) targets+=("${OPTARG}") ;;
    f) force_no_cache=true ;;
    p) parallel=true ;;
    m) skip_mirrors=true ;;
    h) help; exit 0 ;;
    \?) echo "Error: Unknown option -${OPTARG}" >&2; help; exit 2 ;;
    :)  echo "Error: Option -${OPTARG} requires an argument" >&2; help; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ "$#" -ne 0 ]]; then
  echo "Error: Unexpected arguments: $*" >&2
  help
  exit 2
fi

# Map target name → representative compose service
declare -A TARGET_SERVICES=(
  [app]="ur_driver"
  [ndi]="polaris_driver"
  [franka]="franka_driver"
)

DOCKERFILE_PATH="Dockerfile"
BUILD_STAMP_DIR=".build_stamps"

# Record the Dockerfile mtime for a successfully built target
write_build_stamp() {
  local target="$1"
  mkdir -p "${BUILD_STAMP_DIR}"
  # Get Dockerfile mtime (GNU stat, then BSD stat, then Python)
  local mtime=""
  mtime=$(stat -c '%Y' "${DOCKERFILE_PATH}" 2>/dev/null || true)
  if [[ -z "${mtime}" ]]; then
    mtime=$(stat -f '%m' "${DOCKERFILE_PATH}" 2>/dev/null || true)
  fi
  if [[ -z "${mtime}" ]]; then
    mtime=$(python3 -c "import os; print(int(os.path.getmtime('${DOCKERFILE_PATH}')))" 2>/dev/null || true)
  fi
  if [[ -n "${mtime}" ]]; then
    printf '%s\n' "${mtime}" > "${BUILD_STAMP_DIR}/${target}"
  fi
}

# Validate all supplied targets
for t in "${targets[@]}"; do
  if [[ -z "${TARGET_SERVICES[$t]+_}" ]]; then
    echo "Error: Unknown target '${t}'. Choose from: ${!TARGET_SERVICES[*]}" >&2
    exit 2
  fi
done

# Default to all targets if none specified
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(app ndi franka)
fi

cache_flag=()
if [[ "${force_no_cache}" == "true" ]]; then
  cache_flag=(--no-cache)
fi

build_args=()
if [[ "${skip_mirrors}" == "true" ]]; then
  echo "Skipping mirror probing — using default mirrors."
  build_args+=(--build-arg UBUNTU_MIRROR=https://archive.ubuntu.com/ubuntu)
  build_args+=(--build-arg ROS2_MIRROR=http://packages.ros.org/ros2/ubuntu)
fi

if [[ "${parallel}" == "true" && ${#targets[@]} -gt 1 ]]; then
  # ── Parallel build ──
  echo "Building ${#targets[@]} target(s) in parallel: ${targets[*]} …"
  for tgt in "${targets[@]}"; do
    svc="${TARGET_SERVICES[$tgt]}"
    echo "+ docker compose build ${cache_flag[*]:-} ${build_args[*]:-} ${svc} &"
    docker compose build "${cache_flag[@]}" "${build_args[@]}" "${svc}" &
  done
  wait
  # Record stamps for all targets after successful parallel build
  for tgt in "${targets[@]}"; do
    write_build_stamp "${tgt}"
  done
  echo "All parallel builds finished."
else
  # ── Sequential build ──
  for tgt in "${targets[@]}"; do
    svc="${TARGET_SERVICES[$tgt]}"
    cmd=(docker compose build "${cache_flag[@]}" "${build_args[@]}" "${svc}")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
    write_build_stamp "${tgt}"
  done
fi
