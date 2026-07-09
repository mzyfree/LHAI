#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_HOME="$(cd "${OPS_HOME}/.." && pwd)"
PAPER_HOME="${PROJECT_HOME}/paper_trading_system"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

echo "===== STEP 1/3 update or confirm local Qlib data ====="
bash "${PAPER_HOME}/bin/update_data.sh"

echo
echo "===== STEP 2/3 ensure short-hold v2 predictions ====="
bash "${OPS_HOME}/bin/ensure_short_hold_prediction_ready.sh"

echo
echo "===== STEP 3/3 generate short-hold single-peak ticket ====="
echo "Short-hold scoring mode: ${SHORT_HOLD_SCORING_MODE:-v2}"
SHORT_HOLD_PRED_DIR="${SHORT_HOLD_PRED_DIR:-${OPS_HOME}/preds/csi1000_short_hold_v2}"
export PRED_A="${SHORT_HOLD_PRED_A:-${SHORT_HOLD_PRED_DIR}/xgb_csi1000_long_prod2026.pkl}"
export PRED_B="${SHORT_HOLD_PRED_B:-${SHORT_HOLD_PRED_DIR}/doubleensemble_csi1000_short_prod2026.pkl}"
export PRED_C="${SHORT_HOLD_PRED_C:-${SHORT_HOLD_PRED_DIR}/catboost_csi1000_long_prod2026.pkl}"
"${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/short_hold_generate_orders.py"
