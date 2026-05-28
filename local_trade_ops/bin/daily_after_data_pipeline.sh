#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_HOME="$(cd "${OPS_HOME}/.." && pwd)"
PAPER_HOME="${PROJECT_HOME}/paper_trading_system"

echo "===== STEP 1/3 update local Qlib data ====="
bash "${PAPER_HOME}/bin/update_data.sh"

echo "===== STEP 2/3 validate or generate predictions ====="
bash "${OPS_HOME}/bin/generate_predictions.sh"

echo "===== STEP 3/3 generate after-close review package ====="
bash "${OPS_HOME}/bin/after_close_local.sh"

echo "Daily pipeline finished. Review pending orders in the UI before any execution task is generated."
