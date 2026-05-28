#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/train_models.sh daily [cpu|gpu|auto]
  bash bin/train_models.sh weekly [cpu|gpu|auto]
  bash bin/train_models.sh monthly [cpu|gpu|auto]
  bash bin/train_models.sh full [cpu|gpu|auto]
  bash bin/train_models.sh all [cpu|gpu|auto]

Schedule:
  daily   -> XGB short
  weekly  -> XGB short + ADARNN short
  monthly -> XGB short + XGB long + ADARNN short
  full    -> XGB short + XGB long + ADARNN short
  all     -> alias of full
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

SCHEDULE="$1"
DEVICE_OVERRIDE="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

set -a
source "${ENV_FILE}"
set +a

DEVICE="${DEVICE_OVERRIDE:-${DEVICE:-auto}}"

case "${DEVICE}" in
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-${PAPER_HOME}/tmp/joblib}"
    mkdir -p "${JOBLIB_TEMP_FOLDER}"
    echo "DEVICE=cpu, CUDA disabled"
    ;;
  gpu)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    echo "DEVICE=gpu, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    ;;
  auto)
    echo "DEVICE=auto, leaving CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    ;;
  *)
    echo "Unknown DEVICE: ${DEVICE}"
    exit 2
    ;;
esac

if [ -n "${QLIB_SRC:-}" ]; then
  export PYTHONPATH="${QLIB_SRC}:${PAPER_HOME}/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PAPER_HOME}/src:${PYTHONPATH:-}"
fi
mkdir -p "${PAPER_LOG_DIR}" "$(dirname "${XGB_SHORT_PRED}")"

latest_pred() {
  "${PYTHON_BIN}" - "$MLRUNS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
matches = list(root.glob("**/artifacts/pred.pkl"))
if not matches:
    raise SystemExit(0)
latest = max(matches, key=lambda p: p.stat().st_mtime)
print(latest)
PY
}

run_one() {
  local name="$1"
  local config="$2"
  local pred="$3"
  local log="${PAPER_LOG_DIR}/train_${name}_$(date +%Y%m%d_%H%M%S).log"

  if [ "${FORCE_TRAIN:-0}" != "1" ] && [ -f "${pred}" ]; then
    echo "SKIP exists: ${pred}"
    return 0
  fi

  if [ ! -f "${config}" ]; then
    echo "Missing config for ${name}: ${config}"
    return 1
  fi

  echo "===== TRAIN ${name} ====="
  echo "config=${config}"
  echo "pred=${pred}"
  echo "log=${log}"

  "${QRUN_BIN}" "${config}" 2>&1 | tee "${log}"

  local matched
  matched="$(latest_pred)"
  if [ -z "${matched}" ] || [ ! -f "${matched}" ]; then
    echo "No pred.pkl found under ${MLRUNS_DIR}"
    return 1
  fi

  mkdir -p "$(dirname "${pred}")"
  cp "${matched}" "${pred}"
  ls -lh "${pred}"
}

case "${SCHEDULE}" in
  daily)
    run_one "${XGB_SHORT_NAME}" "${XGB_SHORT_CONFIG}" "${XGB_SHORT_PRED}"
    ;;
  weekly)
    run_one "${XGB_SHORT_NAME}" "${XGB_SHORT_CONFIG}" "${XGB_SHORT_PRED}"
    run_one "${ADARNN_SHORT_NAME}" "${ADARNN_SHORT_CONFIG}" "${ADARNN_SHORT_PRED}"
    ;;
  monthly|full|all)
    run_one "${XGB_SHORT_NAME}" "${XGB_SHORT_CONFIG}" "${XGB_SHORT_PRED}"
    run_one "${XGB_LONG_NAME}" "${XGB_LONG_CONFIG}" "${XGB_LONG_PRED}"
    run_one "${ADARNN_SHORT_NAME}" "${ADARNN_SHORT_CONFIG}" "${ADARNN_SHORT_PRED}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
