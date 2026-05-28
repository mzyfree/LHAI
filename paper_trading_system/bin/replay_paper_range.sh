#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/replay_paper_range.sh START_SIGNAL_DATE END_SIGNAL_DATE

Example:
  bash bin/replay_paper_range.sh 2026-05-11 2026-05-18

For each signal date, this runs:
  1. after-close with that signal date
  2. after-open on the next trading day, if that day exists in the local Qlib calendar
EOF
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

START_DATE="$1"
END_DATE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

OVERRIDE_CAPITAL="${CAPITAL:-}"
OVERRIDE_TOPK="${TOPK:-}"
OVERRIDE_N_DROP="${N_DROP:-}"
OVERRIDE_BUY_SCAN_TOPK="${BUY_SCAN_TOPK:-}"
OVERRIDE_ALLOW_ONE_LOT_OVER_TARGET="${ALLOW_ONE_LOT_OVER_TARGET:-}"
OVERRIDE_PAPER_STATE_DIR="${PAPER_STATE_DIR:-}"
OVERRIDE_PAPER_REPORT_DIR="${PAPER_REPORT_DIR:-}"
OVERRIDE_FILTER_MODE="${FILTER_MODE:-}"
OVERRIDE_MIN_PRICE="${MIN_PRICE:-}"
OVERRIDE_MAX_PRICE="${MAX_PRICE:-}"
OVERRIDE_MIN_AMOUNT="${MIN_AMOUNT:-}"
OVERRIDE_LOOKBACK="${LOOKBACK:-}"
OVERRIDE_QLIB_PROVIDER_URI="${QLIB_PROVIDER_URI:-}"
OVERRIDE_PRED_A="${PRED_A:-}"
OVERRIDE_PRED_B="${PRED_B:-}"
OVERRIDE_PRED_C="${PRED_C:-}"
OVERRIDE_FUSED_PRED="${FUSED_PRED:-}"
OVERRIDE_LOT_SIZE="${LOT_SIZE:-}"
OVERRIDE_RESERVE_CASH_PCT="${RESERVE_CASH_PCT:-}"
OVERRIDE_MAX_POSITION_PCT="${MAX_POSITION_PCT:-}"
OVERRIDE_BUY_COST_RATE="${BUY_COST_RATE:-}"
OVERRIDE_SELL_COST_RATE="${SELL_COST_RATE:-}"
OVERRIDE_MIN_COST="${MIN_COST:-}"
OVERRIDE_PYTHON_BIN="${PYTHON_BIN:-}"

set -a
source "${ENV_FILE}"
set +a

CAPITAL="${OVERRIDE_CAPITAL:-${CAPITAL}}"
TOPK="${OVERRIDE_TOPK:-${TOPK}}"
N_DROP="${OVERRIDE_N_DROP:-${N_DROP}}"
BUY_SCAN_TOPK="${OVERRIDE_BUY_SCAN_TOPK:-${BUY_SCAN_TOPK:-150}}"
ALLOW_ONE_LOT_OVER_TARGET="${OVERRIDE_ALLOW_ONE_LOT_OVER_TARGET:-${ALLOW_ONE_LOT_OVER_TARGET:-1}}"
PAPER_STATE_DIR="${OVERRIDE_PAPER_STATE_DIR:-${PAPER_STATE_DIR}}"
PAPER_REPORT_DIR="${OVERRIDE_PAPER_REPORT_DIR:-${PAPER_REPORT_DIR}}"
FILTER_MODE="${OVERRIDE_FILTER_MODE:-${FILTER_MODE:-jq_filter}}"
MIN_PRICE="${OVERRIDE_MIN_PRICE:-${MIN_PRICE:-5}}"
MAX_PRICE="${OVERRIDE_MAX_PRICE:-${MAX_PRICE:-80}}"
MIN_AMOUNT="${OVERRIDE_MIN_AMOUNT:-${MIN_AMOUNT:-20000000}}"
LOOKBACK="${OVERRIDE_LOOKBACK:-${LOOKBACK:-20}}"
QLIB_PROVIDER_URI="${OVERRIDE_QLIB_PROVIDER_URI:-${QLIB_PROVIDER_URI}}"
PRED_A="${OVERRIDE_PRED_A:-${PRED_A:-}}"
PRED_B="${OVERRIDE_PRED_B:-${PRED_B:-}}"
PRED_C="${OVERRIDE_PRED_C:-${PRED_C:-}}"
FUSED_PRED="${OVERRIDE_FUSED_PRED:-${FUSED_PRED:-}}"
LOT_SIZE="${OVERRIDE_LOT_SIZE:-${LOT_SIZE:-100}}"
RESERVE_CASH_PCT="${OVERRIDE_RESERVE_CASH_PCT:-${RESERVE_CASH_PCT:-0.02}}"
MAX_POSITION_PCT="${OVERRIDE_MAX_POSITION_PCT:-${MAX_POSITION_PCT:-0.12}}"
BUY_COST_RATE="${OVERRIDE_BUY_COST_RATE:-${BUY_COST_RATE:-0.0003}}"
SELL_COST_RATE="${OVERRIDE_SELL_COST_RATE:-${SELL_COST_RATE:-0.0008}}"
MIN_COST="${OVERRIDE_MIN_COST:-${MIN_COST:-5}}"
PYTHON_BIN="${OVERRIDE_PYTHON_BIN:-${PYTHON_BIN:-python}}"

ONE_LOT_FLAG="--no-allow-one-lot-over-target"
if [ "${ALLOW_ONE_LOT_OVER_TARGET}" = "1" ] || [ "${ALLOW_ONE_LOT_OVER_TARGET}" = "true" ]; then
  ONE_LOT_FLAG="--allow-one-lot-over-target"
fi

if [ -n "${QLIB_SRC:-}" ]; then
  export PYTHONPATH="${QLIB_SRC}:${PAPER_HOME}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PAPER_HOME}/src:${PYTHONPATH:-}"
fi

mkdir -p "${PAPER_STATE_DIR}" "${PAPER_REPORT_DIR}" "${PAPER_LOG_DIR}"

pairs="$("${PYTHON_BIN}" - "${QLIB_PROVIDER_URI}" "${START_DATE}" "${END_DATE}" <<'PY'
from __future__ import annotations

import sys

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

provider_uri, start, end = sys.argv[1:4]
qlib.init(provider_uri=provider_uri, region=REG_CN)
cal = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
all_cal = pd.DatetimeIndex(D.calendar(start_time=start, freq="day"))
for signal_date in cal:
    future = all_cal[all_cal > signal_date]
    execution_date = future[0] if len(future) else pd.NaT
    print(f"{signal_date.date()} {execution_date.date() if not pd.isna(execution_date) else 'NA'}")
PY
)"

while read -r signal_date execution_date; do
  [ -n "${signal_date}" ] || continue

  echo "===== AFTER CLOSE signal=${signal_date} execution=${execution_date} ====="
  if [ "${execution_date}" = "NA" ]; then
    echo "No execution date found. Skipping after-close and after-open."
    continue
  fi

  if [ -n "${FUSED_PRED:-}" ] && [ -f "${FUSED_PRED}" ]; then
    "${PYTHON_BIN}" "${PAPER_HOME}/src/paper_trading_daily.py" after-close \
      --provider-uri "${QLIB_PROVIDER_URI}" \
      --state-dir "${PAPER_STATE_DIR}" \
      --report-dir "${PAPER_REPORT_DIR}" \
      --fused-pred "${FUSED_PRED}" \
      --capital "${CAPITAL}" \
      --topk "${TOPK}" \
      --n-drop "${N_DROP}" \
      --buy-scan-topk "${BUY_SCAN_TOPK}" \
      --lot-size "${LOT_SIZE}" \
      --reserve-cash-pct "${RESERVE_CASH_PCT}" \
      --max-position-pct "${MAX_POSITION_PCT}" \
      --filter-mode "${FILTER_MODE}" \
      --min-price "${MIN_PRICE}" \
      --max-price "${MAX_PRICE}" \
      --min-amount "${MIN_AMOUNT}" \
      --lookback "${LOOKBACK}" \
      "${ONE_LOT_FLAG}" \
      --signal-date "${signal_date}" \
      --execution-date "${execution_date}"
  else
    "${PYTHON_BIN}" "${PAPER_HOME}/src/paper_trading_daily.py" after-close \
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
      --buy-scan-topk "${BUY_SCAN_TOPK}" \
      --lot-size "${LOT_SIZE}" \
      --reserve-cash-pct "${RESERVE_CASH_PCT}" \
      --max-position-pct "${MAX_POSITION_PCT}" \
      --filter-mode "${FILTER_MODE}" \
      --min-price "${MIN_PRICE}" \
      --max-price "${MAX_PRICE}" \
      --min-amount "${MIN_AMOUNT}" \
      --lookback "${LOOKBACK}" \
      "${ONE_LOT_FLAG}" \
      --signal-date "${signal_date}" \
      --execution-date "${execution_date}"
  fi

  echo "===== AFTER OPEN execution=${execution_date} ====="
  "${PYTHON_BIN}" "${PAPER_HOME}/src/paper_trading_daily.py" after-open \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --state-dir "${PAPER_STATE_DIR}" \
    --report-dir "${PAPER_REPORT_DIR}" \
    --execution-date "${execution_date}" \
    --capital "${CAPITAL}" \
    --topk "${TOPK}" \
    --n-drop "${N_DROP}" \
    --lot-size "${LOT_SIZE}" \
    --buy-cost-rate "${BUY_COST_RATE}" \
    --sell-cost-rate "${SELL_COST_RATE}" \
    --min-cost "${MIN_COST}"
done <<< "${pairs}"
