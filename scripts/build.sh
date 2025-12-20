#!/usr/bin/env bash
set -euo pipefail

help() {
  cat <<'EOF'
Usage:
  build_dev.sh          Run: docker compose --profile dev build
  build_dev.sh -f       Run: docker compose --profile dev build --no-cache

Options:
  -f    Force a clean rebuild (no cache)
  -h    Show this help
EOF
}

force_no_cache=false

while getopts ":fh" opt; do
  case "${opt}" in
    f) force_no_cache=true ;;
    h) help; exit 0 ;;
    \?) echo "Error: Unknown option -${OPTARG}" >&2; help; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ "$#" -ne 0 ]]; then
  echo "Error: Unexpected arguments: $*" >&2
  help
  exit 2
fi

cmd=(docker compose --profile dev build)

if [[ "${force_no_cache}" == "true" ]]; then
  cmd+=(--no-cache)
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"
