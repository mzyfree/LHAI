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

if [ -n "${QLIB_SRC:-}" ]; then
  export PYTHONPATH="${QLIB_SRC}:${PAPER_ENGINE_HOME}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PAPER_ENGINE_HOME}/src:${PYTHONPATH:-}"
fi

mkdir -p "${PAPER_STATE_DIR}" "${PAPER_REPORT_DIR}" "${PAPER_LOG_DIR}"

if [ "${ALLOW_STALE_PRED:-0}" = "1" ] || [ "${ALLOW_STALE_PRED:-0}" = "true" ]; then
  "${PYTHON_BIN}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${PRED_A}" \
    --pred "${PRED_B}" \
    --pred "${PRED_C}" \
    --allow-stale
else
  "${PYTHON_BIN}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${PRED_A}" \
    --pred "${PRED_B}" \
    --pred "${PRED_C}"
fi

ONE_LOT_FLAG="--no-allow-one-lot-over-target"
if [ "${ALLOW_ONE_LOT_OVER_TARGET:-1}" = "1" ] || [ "${ALLOW_ONE_LOT_OVER_TARGET:-1}" = "true" ]; then
  ONE_LOT_FLAG="--allow-one-lot-over-target"
fi

"${PYTHON_BIN}" "${PAPER_ENGINE_HOME}/src/paper_trading_daily.py" after-close \
  --provider-uri "${QLIB_PROVIDER_URI}" \
  --state-dir "${PAPER_STATE_DIR}" \
  --report-dir "${PAPER_REPORT_DIR}" \
  --pred-a "${PRED_A}" \
  --pred-b "${PRED_B}" \
  --pred-c "${PRED_C}" \
  --weights "${WEIGHTS}" \
  --capital "${CAPITAL}" \
  --topk "${TOPK}" \
  --n-drop "${N_DROP}" \
  --buy-scan-topk "${BUY_SCAN_TOPK:-150}" \
  --lot-size "${LOT_SIZE}" \
  --reserve-cash-pct "${RESERVE_CASH_PCT}" \
  --max-position-pct "${MAX_POSITION_PCT}" \
  --filter-mode "${FILTER_MODE}" \
  --min-price "${MIN_PRICE}" \
  --max-price "${MAX_PRICE}" \
  --min-amount "${MIN_AMOUNT}" \
  --lookback "${LOOKBACK}" \
  "${ONE_LOT_FLAG}"
