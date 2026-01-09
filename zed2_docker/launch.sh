#!/usr/bin/env bash
set -euo pipefail

detect_docker_gid() {
  # Allow override: DOCKER_GID=984 ./gen-dotenv.sh
  if [[ -n "${DOCKER_GID:-}" ]]; then
    printf '%s' "${DOCKER_GID}"
    return 0
  fi

  local gid=""
  if command -v getent >/dev/null 2>&1; then
    # Format: group:password:GID:user(s)
    gid="$(getent group docker 2>/dev/null | cut -d: -f3 || true)"
  elif [[ -r /etc/group ]]; then
    gid="$(awk -F: '$1=="docker"{print $3}' /etc/group || true)"
  fi

  # Non-fatal: some hosts (e.g., macOS) may not have a docker group
  if [[ -z "$gid" ]]; then
    echo "Warning: could not determine docker group GID (group 'docker' not found). DOCKER_GID will be empty." >&2
  fi

  printf '%s' "$gid"
}

uid=""
gid=""

if [[ "${USE_DIR_OWNER}" -eq 1 ]]; then
  # Try GNU stat (-c) first, then BSD/macOS stat (-f).
  if stat -c '%u' "$DIR_PATH" >/dev/null 2>&1; then
    uid="$(stat -c '%u' "$DIR_PATH")"
    gid="$(stat -c '%g' "$DIR_PATH")"
  elif stat -f '%u' "$DIR_PATH" >/dev/null 2>&1; then
    uid="$(stat -f '%u' "$DIR_PATH")"
    gid="$(stat -f '%g' "$DIR_PATH")"
  else
    echo "Error: cannot determine owner of '$DIR_PATH' with stat." >&2
    exit 1
  fi
else
  # Current host user's ids.
  uid="$(id -u)"
  gid="$(id -g)"
fi

docker_gid="$(detect_docker_gid)"

echo "Using UID=${uid} GID=${gid} DOCKER_GID=${docker_gid} for docker compose"

# 3) Run docker compose with the computed environment
docker compose up -d