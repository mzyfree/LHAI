#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/train_weekly_main_models.sh RUN_ID PROVIDER_URI TRAIN_END VALID_START VALID_END INFER_DATE [LABEL_MODE] [MODEL_FLAVOR]

Example:
  bash bin/train_weekly_main_models.sh 20260531_20260529 \
    /root/autodl-tmp/llhh/reference_cn_data/releases/20260531_20260529/qlib_bin \
    2025-12-31 2026-01-05 2026-05-22 2026-05-29
EOF
}

if [ "$#" -lt 6 ]; then
  usage
  exit 2
fi

RUN_ID="$1"
PROVIDER_URI="$2"
TRAIN_END="$3"
VALID_START="$4"
VALID_END="$5"
INFER_DATE="$6"
LABEL_MODE="${7:-default}"
MODEL_FLAVOR="${8:-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
QRUN_BIN="${QRUN_BIN:-/root/miniconda3/bin/qrun}"
QLIB_SRC="${QLIB_SRC:-/root/autodl-tmp/llhh/qlib}"
MLRUNS_DIR="${MLRUNS_DIR:-${PAPER_HOME}/mlruns}"
CONFIG_BASE="${CONFIG_BASE:-${PAPER_HOME}/config/smallcap/csi1000}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/llhh/weekly_train/${RUN_ID}}"
CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${RUN_ROOT}/logs"
PRED_DIR="${RUN_ROOT}/preds/csi1000"
PACKAGE_ROOT_NAME="csi1000_${MODEL_FLAVOR}_${RUN_ID}"
PACKAGE_ROOT="${RUN_ROOT}/model_package/${PACKAGE_ROOT_NAME}"
PACKAGE_PATH="${RUN_ROOT}/csi1000_${MODEL_FLAVOR}_model_packages_${RUN_ID}.tar.gz"

export PYTHONPATH="${QLIB_SRC}:${PAPER_HOME}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${PRED_DIR}" "${PACKAGE_ROOT}"

if [ ! -x "${QRUN_BIN}" ]; then
  if command -v qrun >/dev/null 2>&1; then
    QRUN_BIN="$(command -v qrun)"
  else
    echo "qrun not found. Set QRUN_BIN or install qrun in the active Python environment."
    exit 1
  fi
fi

echo "===== weekly train context ====="
echo "RUN_ID=${RUN_ID}"
echo "PAPER_HOME=${PAPER_HOME}"
echo "PROVIDER_URI=${PROVIDER_URI}"
echo "TRAIN_END=${TRAIN_END}"
echo "VALID=${VALID_START} -> ${VALID_END}"
echo "INFER_DATE=${INFER_DATE}"
echo "LABEL_MODE=${LABEL_MODE}"
echo "MODEL_FLAVOR=${MODEL_FLAVOR}"
echo "RUN_ROOT=${RUN_ROOT}"
echo

"${PYTHON_BIN}" "${PAPER_HOME}/bin/gen_weekly_prod_configs.py" \
  --base-dir "${CONFIG_BASE}" \
  --output-dir "${CONFIG_DIR}" \
  --provider-uri "${PROVIDER_URI}" \
  --train-end "${TRAIN_END}" \
  --valid-start "${VALID_START}" \
  --valid-end "${VALID_END}" \
  --infer-date "${INFER_DATE}" \
  --label-mode "${LABEL_MODE}"

snapshot_runs() {
  "${PYTHON_BIN}" - "${MLRUNS_DIR}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()
for p in sorted(root.glob("**/artifacts/pred.pkl")):
    print(p.parents[1])
PY
}

find_new_run() {
  local before_file="$1"
  "${PYTHON_BIN}" - "${MLRUNS_DIR}" "${before_file}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser()
before = set(Path(sys.argv[2]).read_text().splitlines())
runs = sorted(
    {p.parents[1] for p in root.glob("**/artifacts/pred.pkl")},
    key=lambda p: p.stat().st_mtime,
)
for run in reversed(runs):
    if str(run) in before:
        continue
    artifacts = run / "artifacts"
    if (artifacts / "params.pkl").exists() and (artifacts / "pred.pkl").exists() and (artifacts / "task").exists():
        print(run)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

train_one() {
  local model_name="$1"
  local config="${CONFIG_DIR}/workflow_config_${model_name}.yaml"
  local log="${LOG_DIR}/train_${model_name}.log"
  local model_dir="${PACKAGE_ROOT}/${model_name}"

  if [ ! -f "${config}" ]; then
    echo "Missing generated config: ${config}"
    return 1
  fi

  echo "===== TRAIN ${model_name} ====="
  local before_file
  before_file="$(mktemp)"
  snapshot_runs > "${before_file}"

  "${QRUN_BIN}" "${config}" 2>&1 | tee "${log}"

  local run_dir
  run_dir="$(find_new_run "${before_file}")"
  rm -f "${before_file}"

  mkdir -p "${model_dir}/artifacts"
  cp "${run_dir}/artifacts/params.pkl" "${model_dir}/artifacts/params.pkl"
  cp "${run_dir}/artifacts/pred.pkl" "${model_dir}/artifacts/pred.pkl"
  cp "${run_dir}/artifacts/task" "${model_dir}/artifacts/task"
  cp "${config}" "${model_dir}/workflow_config.yaml"
  cp "${run_dir}/artifacts/pred.pkl" "${PRED_DIR}/${model_name}.pkl"

  "${PYTHON_BIN}" - "${PRED_DIR}/${model_name}.pkl" "${INFER_DATE}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
expected = sys.argv[2]
df = pd.read_pickle(path)
dates = pd.DatetimeIndex(sorted(df.index.get_level_values("datetime").unique()))
print(f"{path.name}: {dates[0].date()} -> {dates[-1].date()} n={len(dates)}")
if str(dates[-1].date()) != expected:
    raise SystemExit(f"Prediction latest date {dates[-1].date()} != expected {expected}")
PY
}

train_one "xgb_csi1000_long_prod2026"
train_one "doubleensemble_csi1000_short_prod2026"
train_one "catboost_csi1000_long_prod2026"

tar -czf "${PACKAGE_PATH}" -C "${RUN_ROOT}/model_package" "${PACKAGE_ROOT_NAME}"

echo
echo "===== weekly training package ready ====="
ls -lh "${PACKAGE_PATH}"
find "${PRED_DIR}" -maxdepth 1 -type f -name '*.pkl' -print | sort
