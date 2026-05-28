#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash bin/export_manual_order_ticket.sh YYYY-MM-DD [--force]"
  exit 2
fi

EXECUTION_DATE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "===== STEP 1/2 create live order task ====="
bash "${OPS_HOME}/bin/trigger_live_task.sh" "${EXECUTION_DATE}" --force

echo
echo "===== STEP 2/2 export manual broker ticket ====="
bash "${OPS_HOME}/bin/submit_live_orders.sh" "${EXECUTION_DATE}" --force "$@"
