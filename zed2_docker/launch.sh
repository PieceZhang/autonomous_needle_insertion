#!/usr/bin/env bash
set -euo pipefail

# 1) Host user IDs
UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

# 2) Docker socket group ID (best-effort; useful when mounting /var/run/docker.sock)
DOCKER_GID_VAL=""

# Prefer reading the group ID directly from the docker socket (works even if group isn't named "docker")
if [[ -S /var/run/docker.sock ]]; then
  # Linux (GNU coreutils): stat -c
  if DOCKER_GID_VAL="$(stat -c '%g' /var/run/docker.sock 2>/dev/null)"; then
    :
  # macOS / BSD: stat -f
  elif DOCKER_GID_VAL="$(stat -f '%g' /var/run/docker.sock 2>/dev/null)"; then
    :
  else
    DOCKER_GID_VAL=""
  fi
fi

# Fallback: look up "docker" group GID (common on Linux)
if [[ -z "${DOCKER_GID_VAL}" ]] && command -v getent >/dev/null 2>&1; then
  DOCKER_GID_VAL="$(getent group docker | awk -F: '{print $3}' | head -n1 || true)"
fi

# Last resort: avoid empty interpolation (empty group_add entry can break compose parsing on some setups)
if [[ -z "${DOCKER_GID_VAL}" ]]; then
  DOCKER_GID_VAL="${GID_VAL}"
fi

export UID="${UID_VAL}"
export GID="${GID_VAL}"
export DOCKER_GID="${DOCKER_GID_VAL}"

echo "[run-compose] UID=${UID} GID=${GID} DOCKER_GID=${DOCKER_GID}"

# 3) Run docker compose with the computed environment
docker compose up -d