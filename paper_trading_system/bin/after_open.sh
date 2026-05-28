#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash bin/after_open.sh YYYY-MM-DD"
  exit 2
fi

EXECUTION_DATE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

set -a
source "${ENV_FILE}"
set +a

if [ -n "${QLIB_SRC:-}" ]; then
  export PYTHONPATH="${QLIB_SRC}:${PAPER_HOME}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PAPER_HOME}/src:${PYTHONPATH:-}"
fi
mkdir -p "${PAPER_STATE_DIR}" "${PAPER_REPORT_DIR}" "${PAPER_LOG_DIR}"

"${PYTHON_BIN}" "${PAPER_HOME}/src/paper_trading_daily.py" after-open \
  --provider-uri "${QLIB_PROVIDER_URI}" \
  --state-dir "${PAPER_STATE_DIR}" \
  --report-dir "${PAPER_REPORT_DIR}" \
  --execution-date "${EXECUTION_DATE}" \
  --capital "${CAPITAL}" \
  --topk "${TOPK}" \
  --n-drop "${N_DROP}" \
  --lot-size "${LOT_SIZE}" \
  --buy-cost-rate "${BUY_COST_RATE}" \
  --sell-cost-rate "${SELL_COST_RATE}" \
  --min-cost "${MIN_COST}"
