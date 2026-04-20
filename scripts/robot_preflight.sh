#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
PROFILE=""

as_bool() {
  local v="${1:-}"
  v="$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" || "${v}" == "on" ]]
}

info() {
  printf '[preflight] %s\n' "$*"
}

warn() {
  printf '[preflight][warn] %s\n' "$*" >&2
}

die() {
  printf '[preflight][error] %s\n' "$*" >&2
  exit 1
}

services_for_profile() {
  local profile="$1"
  docker compose --profile "${profile}" config --services 2>/dev/null || true
}

profile_has_service() {
  local profile="$1"
  local service="$2"
  local raw_services
  raw_services="$(services_for_profile "${profile}")"
  if [[ -z "${raw_services}" ]]; then
    return 1
  fi
  printf '%s\n' "${raw_services}" | grep -Fxq "${service}"
}

ur_dashboard_send() {
  local ip="$1"
  local port="$2"
  local timeout_sec="$3"
  local cmd="$4"

  python3 - "$ip" "$port" "$timeout_sec" "$cmd" <<'PY'
import socket
import sys

ip = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
cmd = sys.argv[4]

def recv_all(sock, first_timeout):
    chunks = []
    sock.settimeout(first_timeout)
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
            # Most dashboard replies are single-line; once we got a newline,
            # switch to a very short timeout and drain leftovers.
            if b"\n" in data:
                sock.settimeout(0.1)
    except socket.timeout:
        pass
    return b"".join(chunks).decode("utf-8", errors="replace")

with socket.create_connection((ip, port), timeout=timeout_s) as sock:
    _ = recv_all(sock, timeout_s)
    sock.sendall((cmd + "\n").encode("utf-8"))
    reply = recv_all(sock, timeout_s).strip()
    if reply:
        print(reply)
PY
}

ur_wait_until() {
  local ip="$1"
  local port="$2"
  local timeout_sec="$3"
  local poll_cmd="$4"
  local want_regex="$5"
  local wait_sec="$6"

  local elapsed=0
  local out=""
  while (( elapsed < wait_sec )); do
    out="$(ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "${poll_cmd}" 2>/dev/null || true)"
    if printf '%s' "${out}" | grep -Eiq "${want_regex}"; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  warn "Timed out waiting for UR dashboard condition cmd='${poll_cmd}', expected='${want_regex}', last='${out}'."
  return 1
}

run_ur_preflight() {
  local ip="${UR_ROBOT_IP:-}"
  local port="${UR_DASHBOARD_PORT:-29999}"
  local timeout_sec="${UR_DASHBOARD_TIMEOUT_SEC:-3}"
  local wait_running_sec="${UR_WAIT_RUNNING_SEC:-90}"
  local wait_playing_sec="${UR_WAIT_PLAYING_SEC:-90}"
  local installation_file="${UR_INSTALLATION_FILE:-}"
  local program_file="${UR_EXTERNAL_CONTROL_PROGRAM:-}"

  [[ -n "${ip}" ]] || die "UR preflight enabled but UR_ROBOT_IP is empty."

  info "UR preflight: connecting dashboard ${ip}:${port}."
  ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "robotmode" >/dev/null

  if as_bool "${UR_AUTO_POWER_ON:-true}"; then
    info "UR preflight: power on."
    ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "power on" >/dev/null || die "UR 'power on' failed."
  fi

  if as_bool "${UR_AUTO_BRAKE_RELEASE:-true}"; then
    info "UR preflight: brake release."
    ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "brake release" >/dev/null || die "UR 'brake release' failed."
  fi

  ur_wait_until "${ip}" "${port}" "${timeout_sec}" "robotmode" "running|robotmode:[[:space:]]*7" "${wait_running_sec}" \
    || die "UR did not reach RUNNING mode. Check pendant for safety popups/E-stop/protective stop."

  if [[ -n "${installation_file}" ]]; then
    info "UR preflight: loading installation '${installation_file}'."
    ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "load installation ${installation_file}" >/dev/null \
      || die "UR failed to load installation '${installation_file}'."
  fi

  if [[ -n "${program_file}" ]]; then
    info "UR preflight: loading program '${program_file}'."
    ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "load ${program_file}" >/dev/null \
      || die "UR failed to load program '${program_file}'."
  fi

  if as_bool "${UR_AUTO_PLAY:-true}"; then
    info "UR preflight: play program."
    ur_dashboard_send "${ip}" "${port}" "${timeout_sec}" "play" >/dev/null || die "UR 'play' failed."
    ur_wait_until "${ip}" "${port}" "${timeout_sec}" "programState" "playing|running" "${wait_playing_sec}" \
      || die "UR program did not enter PLAYING state."
  fi

  info "UR preflight: completed."
}

franka_api_call() {
  local method="$1"
  local url="$2"
  local timeout_sec="$3"
  local token="$4"

  local hdr_auth=()
  if [[ -n "${token}" ]]; then
    hdr_auth=(-H "Authorization: Bearer ${token}")
  fi

  local response
  response="$(curl -ksS -m "${timeout_sec}" -X "${method}" "${url}" \
    -H "Accept: application/json" "${hdr_auth[@]}" \
    -w $'\n%{http_code}' 2>/dev/null || true)"

  if [[ -z "${response}" ]]; then
    return 1
  fi

  local http_code
  http_code="$(printf '%s\n' "${response}" | tail -n1)"
  if [[ "${http_code}" =~ ^2[0-9][0-9]$ ]]; then
    return 0
  fi

  return 1
}

run_franka_preflight_api() {
  local robot_ip="${FRANKA_ROBOT_IP:-}"
  local timeout_sec="${FRANKA_DESK_TIMEOUT_SEC:-8}"
  local token="${FRANKA_DESK_TOKEN:-}"
  local base_url="${FRANKA_DESK_BASE_URL:-}"
  local unlock_endpoint="${FRANKA_DESK_UNLOCK_ENDPOINT:-/desk/api/joints/unlock}"
  local fci_on_endpoint="${FRANKA_DESK_FCI_ON_ENDPOINT:-/desk/api/system/fci/enable}"
  local fci_status_endpoint="${FRANKA_DESK_STATUS_ENDPOINT:-}"
  local fci_status_regex="${FRANKA_DESK_STATUS_EXPECT_REGEX:-active[[:space:]]*\"?[[:space:]]*:[[:space:]]*true|fci[[:space:]]*\"?[[:space:]]*:[[:space:]]*true}"
  local wait_sec="${FRANKA_WAIT_FCI_SEC:-60}"

  [[ -n "${robot_ip}" ]] || die "Franka preflight enabled but FRANKA_ROBOT_IP is empty."
  if [[ -z "${base_url}" ]]; then
    base_url="https://${robot_ip}"
  fi

  info "Franka preflight(api): unlock joints."
  franka_api_call "POST" "${base_url}${unlock_endpoint}" "${timeout_sec}" "${token}" \
    || die "Franka Desk unlock API call failed. Verify endpoint/token."

  info "Franka preflight(api): activate FCI."
  franka_api_call "POST" "${base_url}${fci_on_endpoint}" "${timeout_sec}" "${token}" \
    || die "Franka Desk FCI activation API call failed. Verify endpoint/token."

  if [[ -n "${fci_status_endpoint}" ]]; then
    local elapsed=0
    while (( elapsed < wait_sec )); do
      if franka_api_call "GET" "${base_url}${fci_status_endpoint}" "${timeout_sec}" "${token}"; then
        local body
        body="$(curl -ksS -m "${timeout_sec}" -X GET "${base_url}${fci_status_endpoint}" \
          -H "Accept: application/json" \
          ${token:+-H "Authorization: Bearer ${token}"} 2>/dev/null || true)"
        if printf '%s' "${body}" | grep -Eiq "${fci_status_regex}"; then
          info "Franka preflight(api): FCI active."
          return 0
        fi
      fi
      sleep 1
      elapsed=$((elapsed + 1))
    done
    die "Franka FCI status did not become active in time."
  fi

  warn "Franka status endpoint not configured; skipped FCI state verification."
}

run_franka_preflight_playwright() {
  local cmd="${FRANKA_DESK_PLAYWRIGHT_CMD:-node scripts/franka_desk_playwright.mjs}"
  info "Franka preflight(playwright): running '${cmd}'."
  eval "${cmd}" || die "Franka Playwright automation failed."
}

run_franka_preflight() {
  if as_bool "${FRANKA_USE_FAKE_HARDWARE:-true}"; then
    info "Franka preflight: skipped (FRANKA_USE_FAKE_HARDWARE=true)."
    return 0
  fi

  local backend="${FRANKA_DESK_BACKEND:-api}"
  case "${backend}" in
    api)
      run_franka_preflight_api
      ;;
    playwright)
      run_franka_preflight_playwright
      ;;
    none)
      warn "Franka preflight backend is 'none'; skipping Desk automation."
      ;;
    *)
      die "Unknown FRANKA_DESK_BACKEND='${backend}'. Use one of: api, playwright, none."
      ;;
  esac

  info "Franka preflight: completed."
}

print_help() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS]

Automate robot-side preflight steps before docker compose startup.

Options:
  --profile PROFILE   Compose profile name used to decide which robot checks run
  -h, --help          Show help and exit

Environment (high-level):
  AUTO_ROBOT_PREFLIGHT=true|false   Master switch (default: true)
  AUTO_UR_DASHBOARD=true|false      Enable UR dashboard preflight (default: true)
  AUTO_FRANKA_DESK=true|false       Enable Franka Desk preflight (default: true)

UR key vars:
  UR_ROBOT_IP
  UR_DASHBOARD_PORT (default: 29999)
  UR_INSTALLATION_FILE (optional)
  UR_EXTERNAL_CONTROL_PROGRAM (optional)
  UR_AUTO_POWER_ON, UR_AUTO_BRAKE_RELEASE, UR_AUTO_PLAY (default: true)

Franka key vars:
  FRANKA_ROBOT_IP
  FRANKA_USE_FAKE_HARDWARE (if true, Franka preflight is skipped)
  FRANKA_DESK_BACKEND=api|playwright|none (default: api)
  FRANKA_DESK_TOKEN, FRANKA_DESK_BASE_URL, FRANKA_DESK_*_ENDPOINT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires an argument."
      PROFILE="$2"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

if ! as_bool "${AUTO_ROBOT_PREFLIGHT:-true}"; then
  info "Robot preflight disabled (AUTO_ROBOT_PREFLIGHT=false)."
  exit 0
fi

run_ur=false
run_franka=false

if [[ -n "${PROFILE}" ]]; then
  profile_has_service "${PROFILE}" "ur_driver" && run_ur=true || true
  profile_has_service "${PROFILE}" "franka_driver" && run_franka=true || true
else
  run_ur=true
  run_franka=true
fi

if "${run_ur}" && as_bool "${AUTO_UR_DASHBOARD:-true}"; then
  run_ur_preflight
fi

if "${run_franka}" && as_bool "${AUTO_FRANKA_DESK:-true}"; then
  run_franka_preflight
fi

if ! "${run_ur}" && ! "${run_franka}"; then
  info "No robot driver services in profile '${PROFILE:-<none>}' requiring preflight."
fi

info "Robot preflight done."

