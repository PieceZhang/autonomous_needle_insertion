#!/usr/bin/env bash
set -euo pipefail

help() {
  cat <<'EOF'
Usage:
  build.sh              Build all three images (ur-app, ndi, franka) for the dev profile
  build.sh -t TARGET    Build only the specified target (ur-app | ndi | franka)
  build.sh -f           Force a clean rebuild (no cache)
  build.sh -p           Build all targets in parallel

Options:
  -t TARGET   Build a single Dockerfile stage: ur-app, ndi, or franka
  -f          Force a clean rebuild (no cache)
  -p          Build all targets in parallel (docker buildx bake)
  -h          Show this help

Examples:
  build.sh                    # sequential build of all three images
  build.sh -t ndi             # rebuild only the NDI/Polaris image
  build.sh -t franka -f       # force-rebuild the Franka image
  build.sh -p                 # parallel build of all images
EOF
}

force_no_cache=false
target=""
parallel=false

while getopts ":t:fph" opt; do
  case "${opt}" in
    t) target="${OPTARG}" ;;
    f) force_no_cache=true ;;
    p) parallel=true ;;
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
  [ur-app]="ur_driver"
  [ndi]="polaris_driver"
  [franka]="franka_driver"
)

cache_flag=()
if [[ "${force_no_cache}" == "true" ]]; then
  cache_flag=(--no-cache)
fi

if [[ -n "${target}" ]]; then
  # ── Single-target build ──
  if [[ -z "${TARGET_SERVICES[$target]+_}" ]]; then
    echo "Error: Unknown target '${target}'. Choose from: ${!TARGET_SERVICES[*]}" >&2
    exit 2
  fi
  svc="${TARGET_SERVICES[$target]}"
  cmd=(docker compose build "${cache_flag[@]}" "${svc}")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
elif [[ "${parallel}" == "true" ]]; then
  # ── Parallel build of all targets ──
  echo "Building all targets in parallel …"
  for svc in "${TARGET_SERVICES[@]}"; do
    echo "+ docker compose build ${cache_flag[*]} ${svc} &"
    docker compose build "${cache_flag[@]}" "${svc}" &
  done
  wait
  echo "All parallel builds finished."
else
  # ── Sequential build of all targets ──
  for tgt in ur-app ndi franka; do
    svc="${TARGET_SERVICES[$tgt]}"
    cmd=(docker compose build "${cache_flag[@]}" "${svc}")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
  done
fi
