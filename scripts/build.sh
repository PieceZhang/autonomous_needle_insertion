#!/usr/bin/env bash
set -euo pipefail

# BuildKit is required for --mount=type=cache in Dockerfile
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

help() {
  cat <<'EOF'
Usage:
  build.sh              Build all three images (app, ndi, franka) for the dev profile
  build.sh -t TARGET    Build only the specified target (app | ndi | franka)
  build.sh -f           Force a clean rebuild (no cache)
  build.sh -p           Build all targets in parallel
  build.sh -m           Skip mirror probing (use default mirrors)

Options:
  -t TARGET   Build a single Dockerfile stage: app, ndi, or franka
  -f          Force a clean rebuild (no cache)
  -p          Build all targets in parallel (docker buildx bake)
  -m          Skip mirror speed-test; use archive.ubuntu.com + packages.ros.org
  -h          Show this help

Examples:
  build.sh                    # sequential build of all three images
  build.sh -t ndi             # rebuild only the NDI/Polaris image
  build.sh -t franka -f       # force-rebuild the Franka image
  build.sh -p                 # parallel build of all images
  build.sh -m                 # skip mirror probing for faster builds
EOF
}

force_no_cache=false
target=""
parallel=false
skip_mirrors=false

while getopts ":t:fpmh" opt; do
  case "${opt}" in
    t) target="${OPTARG}" ;;
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

# Map target name → representative compose service(s)
declare -A TARGET_SERVICES=(
  [app]="ur_driver"
  [ndi]="polaris_driver"
  [franka]="franka_driver"
)

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

if [[ -n "${target}" ]]; then
  # ── Single-target build ──
  if [[ -z "${TARGET_SERVICES[$target]+_}" ]]; then
    echo "Error: Unknown target '${target}'. Choose from: ${!TARGET_SERVICES[*]}" >&2
    exit 2
  fi
  svc="${TARGET_SERVICES[$target]}"
  cmd=(docker compose build "${cache_flag[@]}" "${build_args[@]}" "${svc}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
elif [[ "${parallel}" == "true" ]]; then
  # ── Parallel build of all targets ──
  echo "Building all targets in parallel …"
  for svc in "${TARGET_SERVICES[@]}"; do
    echo "+ docker compose build ${cache_flag[*]} ${build_args[*]} ${svc} &"
    docker compose build "${cache_flag[@]}" "${build_args[@]}" "${svc}" &
  done
  wait
  echo "All parallel builds finished."
else
  # ── Sequential build of all targets ──
  for tgt in app ndi franka; do
    svc="${TARGET_SERVICES[$tgt]}"
    cmd=(docker compose build "${cache_flag[@]}" "${build_args[@]}" "${svc}")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
  done
fi
