#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

print_help() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Stop all currently running Docker containers.

Behaviour:
  - If no containers are running, exit with a short message.
  - If containers are running, run:
        docker stop \$(docker ps -q)

Options:
  -h, --help    Show this help message and exit.

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} -h

EOF
}

# Argument parsing (only -h / --help supported for now)
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  print_help
  exit 0
fi

# Stop all running containers (if any)
if [ -z "$(docker ps -q)" ]; then
  echo "No running containers to stop."
  exit 0
fi

echo "Stopping all running containers..."
docker stop $(docker ps -q)
echo "All running containers have been stopped."