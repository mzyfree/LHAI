#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/run_capital_scenarios.sh START_SIGNAL_DATE END_SIGNAL_DATE

Example:
  bash bin/run_capital_scenarios.sh 2026-01-05 2026-05-18

Runs top20/drop2 paper replay for:
  100000, 200000, 1000000

Execution rule:
  BUY_SCAN_TOPK=20, ALLOW_ONE_LOT_OVER_TARGET=1
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

set -a
source "${ENV_FILE}"
set +a

for capital in 100000 200000 1000000; do
  label="$((capital / 10000))w"
  state_dir="${PAPER_HOME}/state_top20_drop2_cap${label}_ytd"
  report_dir="${PAPER_HOME}/reports_top20_drop2_cap${label}_ytd"

  echo "===== RUN capital=${capital} state=${state_dir} ====="
  rm -rf "${state_dir}" "${report_dir}"

  PAPER_STATE_DIR="${state_dir}" \
  PAPER_REPORT_DIR="${report_dir}" \
  CAPITAL="${capital}" \
  TOPK=20 \
  N_DROP=2 \
  BUY_SCAN_TOPK=20 \
  ALLOW_ONE_LOT_OVER_TARGET=1 \
    bash "${PAPER_HOME}/bin/replay_paper_range.sh" "${START_DATE}" "${END_DATE}"
done

"${PYTHON_BIN}" "${PAPER_HOME}/bin/summarize_capital_scenarios.py" "${PAPER_HOME}"
