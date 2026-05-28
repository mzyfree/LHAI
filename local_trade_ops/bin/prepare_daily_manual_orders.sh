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

echo "===== STEP 1/5 update local Qlib data ====="
bash "${PAPER_HOME}/bin/update_data.sh"

echo
echo "===== STEP 2/5 validate or generate predictions ====="
bash "${OPS_HOME}/bin/generate_predictions.sh"

echo
echo "===== STEP 3/5 generate after-close order candidates ====="
bash "${OPS_HOME}/bin/after_close_local.sh"

EXECUTION_DATE="$(
  "${PYTHON_BIN:-python3}" - <<'PY'
from pathlib import Path
import os

state_dir = Path(os.environ["PAPER_STATE_DIR"]).expanduser().resolve()
files = sorted(state_dir.glob("pending_orders_*.csv"), key=lambda p: p.stat().st_mtime)
if not files:
    raise SystemExit(f"No pending_orders_*.csv found in {state_dir}")
name = files[-1].name
print(name.removeprefix("pending_orders_").removesuffix(".csv"))
PY
)"

echo
echo "===== STEP 4/5 auto approve execution date ${EXECUTION_DATE} ====="
"${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/approve_orders.py" \
  --execution-date "${EXECUTION_DATE}" \
  --status approved \
  --reviewer auto \
  --note "auto-approved by prepare_daily_manual_orders"

echo
echo "===== STEP 5/5 export Eastmoney manual order ticket ====="
bash "${OPS_HOME}/bin/export_manual_order_ticket.sh" "${EXECUTION_DATE}" --force

echo
echo "Prepared manual order ticket for execution date: ${EXECUTION_DATE}"
