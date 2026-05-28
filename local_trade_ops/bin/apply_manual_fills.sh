#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash bin/apply_manual_fills.sh YYYY-MM-DD [--fills-csv path] [--allow-negative-cash]"
  exit 2
fi

EXECUTION_DATE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

"${PYTHON_BIN}" "${OPS_HOME}/bin/apply_manual_fills.py" \
  --execution-date "${EXECUTION_DATE}" \
  "$@"
