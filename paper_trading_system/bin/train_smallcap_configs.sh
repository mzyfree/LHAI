#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/train_smallcap_configs.sh MARKET WAVE [WINDOWS]

Examples:
  bash bin/train_smallcap_configs.sh csi1000 core short,mid,long
  bash bin/train_smallcap_configs.sh csi2000 tree short,mid
  bash bin/train_smallcap_configs.sh csi1000 deep short

Waves:
  core -> lgb xgb linear
  tree -> catboost doubleensemble
  deep -> alstm adarnn
  all  -> lgb xgb linear catboost doubleensemble alstm adarnn
EOF
}

if [ "$#" -lt 2 ]; then
  usage
  exit 2
fi

MARKET="$1"
WAVE="$2"
WINDOWS="${3:-short,mid,long}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

QRUN_BIN="${QRUN_BIN:-qrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QLIB_SRC="${QLIB_SRC:-}"
MLRUNS_DIR="${MLRUNS_DIR:-${PAPER_HOME}/mlruns}"
PAPER_LOG_DIR="${PAPER_LOG_DIR:-${PAPER_HOME}/logs}"
PRED_DIR="${PRED_DIR:-${PAPER_HOME}/preds_smallcap}"
CONFIG_ROOT="${CONFIG_ROOT:-${PAPER_HOME}/config/smallcap}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

if [ -n "${QLIB_SRC}" ]; then
  export PYTHONPATH="${QLIB_SRC}:${PAPER_HOME}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PAPER_HOME}/src:${PYTHONPATH:-}"
fi

mkdir -p "${PAPER_LOG_DIR}" "${PRED_DIR}/${MARKET}"

case "${WAVE}" in
  core)
    MODELS=(lgb xgb linear)
    ;;
  tree)
    MODELS=(catboost doubleensemble)
    ;;
  deep)
    MODELS=(alstm adarnn)
    ;;
  all)
    MODELS=(lgb xgb linear catboost doubleensemble alstm adarnn)
    ;;
  *)
    echo "Unknown wave: ${WAVE}"
    usage
    exit 2
    ;;
esac

IFS=',' read -r -a WINDOW_LIST <<< "${WINDOWS}"

snapshot_preds() {
  "${PYTHON_BIN}" - "$MLRUNS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
for p in sorted(root.glob("**/artifacts/pred.pkl")):
    print(p)
PY
}

new_pred_since() {
  local before_file="$1"
  "${PYTHON_BIN}" - "$MLRUNS_DIR" "$before_file" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
before_path = Path(sys.argv[2])
before = set(before_path.read_text().splitlines()) if before_path.exists() else set()
after = sorted(root.glob("**/artifacts/pred.pkl"), key=lambda p: p.stat().st_mtime)
new = [p for p in after if str(p) not in before]
if not new:
    raise SystemExit(0)
print(new[-1])
PY
}

run_one() {
  local model="$1"
  local window="$2"
  local name="${model}_${MARKET}_${window}_prod2026"
  local config="${CONFIG_ROOT}/${MARKET}/workflow_config_${name}.yaml"
  local pred="${PRED_DIR}/${MARKET}/${name}.pkl"
  local log="${PAPER_LOG_DIR}/train_${name}_$(date +%Y%m%d_%H%M%S).log"

  if [ "${FORCE_TRAIN}" != "1" ] && [ -f "${pred}" ]; then
    echo "SKIP exists: ${pred}"
    return 0
  fi

  if [ ! -f "${config}" ]; then
    echo "Missing config: ${config}"
    return 1
  fi

  echo "===== TRAIN ${name} ====="
  echo "config=${config}"
  echo "pred=${pred}"
  echo "log=${log}"

  local before_file
  before_file="$(mktemp)"
  snapshot_preds > "${before_file}"

  "${QRUN_BIN}" "${config}" 2>&1 | tee "${log}"

  local matched
  matched="$(new_pred_since "${before_file}")"
  rm -f "${before_file}"
  if [ -z "${matched}" ] || [ ! -f "${matched}" ]; then
    echo "No new pred.pkl found under ${MLRUNS_DIR} for ${name}"
    return 1
  fi

  cp "${matched}" "${pred}"
  ls -lh "${pred}"
}

for window in "${WINDOW_LIST[@]}"; do
  for model in "${MODELS[@]}"; do
    run_one "${model}" "${window}"
  done
done
