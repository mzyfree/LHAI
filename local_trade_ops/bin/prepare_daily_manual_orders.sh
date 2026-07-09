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

echo "===== STEP 1/4 ensure shared data/prediction layer ====="
bash "${OPS_HOME}/bin/ensure_daily_prediction_ready.sh"

echo
echo "===== STEP 2/4 generate after-close order candidates ====="
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

TODAY_LOCAL="$(
  "${PYTHON_BIN:-python3}" - <<'PY'
from datetime import datetime

print(datetime.now().strftime("%Y-%m-%d"))
PY
)"

LATEST_QLIB_DATE="$(
  "${PYTHON_BIN:-python3}" - <<'PY'
from pathlib import Path
import os

provider = Path(os.environ.get("QLIB_PROVIDER_URI", "")).expanduser()
calendar = provider / "calendars" / "day.txt"
if not calendar.exists():
    print("unknown")
else:
    dates = [line.strip() for line in calendar.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(dates[-1] if dates else "unknown")
PY
)"

if [ "${EXECUTION_DATE}" != "${TODAY_LOCAL}" ]; then
  echo
  echo "===== STOP stale manual ticket ====="
  echo "Refusing to export a non-today manual order ticket."
  echo "Today: ${TODAY_LOCAL}"
  echo "Latest local Qlib calendar date: ${LATEST_QLIB_DATE}"
  echo "Generated execution date: ${EXECUTION_DATE}"
  echo
  echo "This usually means the local Qlib data release has not updated through the previous trading day yet."
  echo "Do not trade this stale ticket. Wait for a newer Qlib data release, then run 生成当天清单 again."
  exit 1
fi

echo
echo "===== STEP 3/4 auto approve execution date ${EXECUTION_DATE} ====="
"${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/approve_orders.py" \
  --execution-date "${EXECUTION_DATE}" \
  --status approved \
  --reviewer auto \
  --note "auto-approved by prepare_daily_manual_orders"

echo
echo "===== STEP 4/4 export Eastmoney manual order ticket ====="
bash "${OPS_HOME}/bin/export_manual_order_ticket.sh" "${EXECUTION_DATE}" --force

echo
echo "Prepared manual order ticket for execution date: ${EXECUTION_DATE}"
