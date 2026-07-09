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

DATE_ARG="${1:-$(date +%F)}"
"${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/short_hold_daily_snapshot.py" --date "${DATE_ARG}"
