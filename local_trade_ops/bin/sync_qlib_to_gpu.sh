#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

if [ -z "${QLIB_PROVIDER_URI:-}" ]; then
  echo "QLIB_PROVIDER_URI is required."
  exit 1
fi

if [ -z "${GPU_QLIB_PROVIDER_URI:-}" ]; then
  echo "GPU_QLIB_PROVIDER_URI is required."
  exit 1
fi

if [ ! -d "${QLIB_PROVIDER_URI}/calendars" ] || [ ! -d "${QLIB_PROVIDER_URI}/features" ] || [ ! -d "${QLIB_PROVIDER_URI}/instruments" ]; then
  echo "Local QLIB_PROVIDER_URI does not look like a Qlib cn_data directory: ${QLIB_PROVIDER_URI}"
  exit 1
fi

ssh -p "${GPU_SSH_PORT}" "${GPU_SSH}" "mkdir -p '${GPU_QLIB_PROVIDER_URI}'"

rsync -avz --delete -e "ssh -p ${GPU_SSH_PORT}" \
  "${QLIB_PROVIDER_URI}/" \
  "${GPU_SSH}:${GPU_QLIB_PROVIDER_URI}/"

echo "Synced local Qlib data to GPU:"
echo "  local: ${QLIB_PROVIDER_URI}"
echo "  gpu:   ${GPU_SSH}:${GPU_QLIB_PROVIDER_URI}"
if [ -f "${QLIB_PROVIDER_URI}/calendars/day.txt" ]; then
  echo "Local calendar tail:"
  tail -5 "${QLIB_PROVIDER_URI}/calendars/day.txt"
fi
