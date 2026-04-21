#!/usr/bin/env bash
set -euo pipefail

# BuildKit is required for --mount=type=cache in Dockerfile
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

help() {
  cat <<'EOF'
Usage:
  build.sh              Build all images (app, ndi, franka, console)
  build.sh -t TARGET    Build only the specified target(s) (app | ndi | franka | console)
  build.sh -f           Force a clean rebuild (no cache)
  build.sh -p           Build targets in parallel (uses docker buildx bake)
  build.sh -M           Enable mirror speed-test (slower but may pick a faster mirror)

Options:
  -t TARGET   Build a Dockerfile stage: app, ndi, franka, or console.
              Can be repeated: -t app -t ndi
  -f          Force a clean rebuild (no cache)
  -p          Build targets in parallel via docker buildx bake
              (shares the base stage across targets — much faster than
              running separate docker compose build commands)
  -M          Enable mirror probing; by default mirrors are skipped
              and archive.ubuntu.com + packages.ros.org are used
  -h          Show this help

Examples:
  build.sh                       # sequential build of all images
  build.sh -t ndi                # rebuild only the NDI/Polaris image
  build.sh -t console            # rebuild only the guidance console image
  build.sh -t app -t ndi         # rebuild app and ndi images
  build.sh -t franka -f          # force-rebuild the Franka image
  build.sh -p                    # parallel build of all images (recommended)
  build.sh -M                    # enable mirror probing (slow networks)
  build.sh -t app -t franka -p   # parallel build of app + franka only
EOF
}

force_no_cache=false
targets=()
parallel=false
enable_mirrors=false

while getopts ":t:fpMh" opt; do
  case "${opt}" in
    t) targets+=("${OPTARG}") ;;
    f) force_no_cache=true ;;
    p) parallel=true ;;
    M) enable_mirrors=true ;;
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

VALID_TARGETS=(app ndi franka console)

# Map target name → representative compose service.
# Keep this Bash 3.2-compatible for macOS's default /bin/bash.
target_to_service() {
  case "$1" in
    app) printf '%s\n' "ur_driver" ;;
    ndi) printf '%s\n' "polaris_driver" ;;
    franka) printf '%s\n' "franka_driver" ;;
    console) printf '%s\n' "guidance-console" ;;
    *) return 1 ;;
  esac
}

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
  if ! target_to_service "${t}" >/dev/null; then
    echo "Error: Unknown target '${t}'. Choose from: ${VALID_TARGETS[*]}" >&2
    exit 2
  fi
done

# Default to all targets if none specified
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(app ndi franka console)
fi

if [[ "${enable_mirrors}" != "true" ]]; then
  echo "Mirror probing disabled (default). Use -M to enable."
fi

docker_compose_build() {
  local svc="$1"
  local -a cmd
  cmd=(docker compose build)

  if [[ "${force_no_cache}" == "true" ]]; then
    cmd+=(--no-cache)
  fi

  if [[ "${enable_mirrors}" != "true" ]]; then
    cmd+=(--build-arg UBUNTU_MIRROR=https://archive.ubuntu.com/ubuntu)
    cmd+=(--build-arg ROS2_MIRROR=http://packages.ros.org/ros2/ubuntu)
  fi

  cmd+=("${svc}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}

# Check whether docker buildx bake is available
has_buildx_bake() {
  docker buildx bake --help >/dev/null 2>&1
}

if [[ "${parallel}" == "true" && ${#targets[@]} -gt 1 ]]; then
  # ── Parallel build via docker buildx bake ──
  # Bake builds all targets in one DAG, sharing the base stage automatically.
  if has_buildx_bake; then
    echo "Building ${#targets[@]} target(s) in parallel via buildx bake: ${targets[*]} …"
    bake_args=()
    if [[ "${force_no_cache}" == "true" ]]; then
      bake_args+=(--no-cache)
    fi
    if [[ "${enable_mirrors}" != "true" ]]; then
      bake_args+=(--set "*.args.UBUNTU_MIRROR=https://archive.ubuntu.com/ubuntu")
      bake_args+=(--set "*.args.ROS2_MIRROR=http://packages.ros.org/ros2/ubuntu")
    fi
    bake_args+=(--load)
    echo "+ docker buildx bake ${bake_args[*]} ${targets[*]}"
    docker buildx bake "${bake_args[@]}" "${targets[@]}"
    for tgt in "${targets[@]}"; do
      write_build_stamp "${tgt}"
    done
    echo "All parallel builds finished."
  else
    # Fallback: build base first, then remaining targets in parallel
    echo "docker buildx bake not available — falling back to sequential-base + parallel targets."
    echo "Building shared base stage first…"
    # Build one target to warm the base cache, then the rest in parallel
    first_tgt="${targets[0]}"
    first_svc="$(target_to_service "${first_tgt}")"
    docker_compose_build "${first_svc}"
    write_build_stamp "${first_tgt}"

    remaining=("${targets[@]:1}")
    if [[ ${#remaining[@]} -gt 0 ]]; then
      echo "Building remaining target(s) in parallel: ${remaining[*]} …"
      for tgt in "${remaining[@]}"; do
        svc="$(target_to_service "${tgt}")"
        docker_compose_build "${svc}" &
      done
      wait
      for tgt in "${remaining[@]}"; do
        write_build_stamp "${tgt}"
      done
    fi
    echo "All parallel builds finished."
  fi
else
  # ── Sequential build ──
  for tgt in "${targets[@]}"; do
    svc="$(target_to_service "${tgt}")"
    docker_compose_build "${svc}"
    write_build_stamp "${tgt}"
  done
fi
