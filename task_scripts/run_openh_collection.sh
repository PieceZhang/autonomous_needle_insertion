#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSPECTIVE_FILE="${SCRIPT_DIR}/rqt_perspective_file/openh.perspective"
LOG_DIR="${SCRIPT_DIR}/../log"
LOG_FILE="${LOG_DIR}/rqt_dataset_collection.log"

echo "Launching rqt with a perspective file for OpenH dataset collection ..."
echo

# Run rqt with the specified perspective file
mkdir -p "${LOG_DIR}"
rqt --perspective-file "${PERSPECTIVE_FILE}" \
  > "${LOG_FILE}" 2>&1 &
