#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
ACTIVE_PROFILE_FILE=".active_profile"

print_help() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS] [PROFILE]

Stop containers that were brought up by launch.sh.

Behaviour (default):
  - Read the active profile from ${ACTIVE_PROFILE_FILE} (written by launch.sh),
    then run:  docker compose --profile <PROFILE> down
  - If PROFILE is given as an argument, use that instead of the saved profile.
  - If no saved profile and no argument, fall back to stopping all project
    containers (all known profiles).

Options:
  -f          Force-stop ALL running Docker containers (not just this project).
  -h, --help  Show this help message and exit.

Examples:
  ${SCRIPT_NAME}                # stop the profile that launch.sh started
  ${SCRIPT_NAME} dev            # stop the 'dev' profile explicitly
  ${SCRIPT_NAME} franka-test    # stop the 'franka-test' profile
  ${SCRIPT_NAME} -f             # stop ALL running Docker containers

EOF
}

force_all=false
profile=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -f)
      force_all=true
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      print_help
      exit 1
      ;;
    *)
      if [[ -n "${profile}" ]]; then
        echo "Error: only one optional PROFILE argument is supported." >&2
        print_help
        exit 1
      fi
      profile="$1"
      ;;
  esac
  shift
done

# -f: stop ALL running Docker containers (original behaviour)
if [[ "${force_all}" == "true" ]]; then
  if [ -z "$(docker ps -q)" ]; then
    echo "No running containers to stop."
    exit 0
  fi
  echo "Force-stopping ALL running Docker containers..."
  docker stop $(docker ps -q)
  echo "All running containers have been stopped."
  # Clean up the active profile marker since everything is stopped
  rm -f "${ACTIVE_PROFILE_FILE}"
  exit 0
fi

# Determine which profile to stop
if [[ -z "${profile}" ]]; then
  if [[ -f "${ACTIVE_PROFILE_FILE}" ]]; then
    profile="$(<"${ACTIVE_PROFILE_FILE}")"
    profile="${profile%%[[:space:]]}"  # trim trailing whitespace/newline
  fi
fi

if [[ -n "${profile}" ]]; then
  echo "Stopping compose services for profile '${profile}'..."
  docker compose --profile "${profile}" down
  # Clean up the active profile marker
  rm -f "${ACTIVE_PROFILE_FILE}"
  echo "Profile '${profile}' stopped."
else
  # No saved profile and no argument — stop all known profiles as a safe fallback
  echo "No active profile found. Stopping all compose project services..."
  # Collect all unique profiles from the compose file
  all_profiles=()
  while IFS= read -r p; do
    [[ -n "$p" ]] && all_profiles+=("$p")
  done < <(docker compose config --profiles 2>/dev/null || true)

  if [[ ${#all_profiles[@]} -gt 0 ]]; then
    profile_flags=()
    for p in "${all_profiles[@]}"; do
      profile_flags+=(--profile "$p")
    done
    docker compose "${profile_flags[@]}" down
  else
    docker compose down
  fi
  rm -f "${ACTIVE_PROFILE_FILE}"
  echo "All compose project services stopped."
fi

