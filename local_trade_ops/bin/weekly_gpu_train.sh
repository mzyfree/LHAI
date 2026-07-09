#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/weekly_gpu_train.sh start
  bash bin/weekly_gpu_train.sh collect RUN_ID

Environment:
  GPU_SSH                 default from config/env.local, e.g. root@connect.westc.seetacloud.com
  GPU_SSH_PORT            default from config/env.local, e.g. 37172
  GPU_SSH_PASSWORD        optional; if set, sshpass is used via SSHPASS
  WEEKLY_TRAIN_ARCHIVE    local qlib tar.gz, defaults to data/archives/latest_qlib_bin.tar.gz
  WEEKLY_TRAIN_INFER_DATE defaults to latest local Qlib calendar date
  WEEKLY_VALID_END        optional; defaults to infer date minus 2 trading days
  WEEKLY_LABEL_MODE       default; use short_hold_v2 for T+1 open -> T+2 close target
  WEEKLY_MODEL_FLAVOR     main by default; use short_hold_v2 to keep packages separate

Training split defaults:
  train: original config start -> 2025-12-31
  valid: 2026-01-05 -> infer date minus 2 trading days
  infer: latest local Qlib date
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
LHAI_HOME="$(cd "${OPS_HOME}/.." && pwd)"
PAPER_HOME="${LHAI_HOME}/paper_trading_system"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

ACTION="${1:-}"
if [ -z "${ACTION}" ]; then
  usage
  exit 2
fi

GPU_SSH="${GPU_SSH:-root@connect.westc.seetacloud.com}"
GPU_SSH_PORT="${GPU_SSH_PORT:-37172}"
GPU_PROJECT_ROOT="${GPU_PROJECT_ROOT:-/root/autodl-tmp/llhh}"
GPU_PAPER_HOME="${GPU_PAPER_HOME:-${GPU_PROJECT_ROOT}/paper_trading_system}"
GPU_QDATA_ROOT="${GPU_QDATA_ROOT:-${GPU_PROJECT_ROOT}/reference_cn_data}"
LOCAL_ARCHIVE="${WEEKLY_TRAIN_ARCHIVE:-${PAPER_HOME}/data/archives/latest_qlib_bin.tar.gz}"
LOCAL_MODEL_PACKAGE_DIR="${LOCAL_MODEL_PACKAGE_DIR:-${OPS_HOME}/model_packages}"
LOCAL_WEEKLY_ARCHIVE_DIR="${LOCAL_WEEKLY_ARCHIVE_DIR:-${OPS_HOME}/archive/weekly_train}"
WEEKLY_LABEL_MODE="${WEEKLY_LABEL_MODE:-default}"
WEEKLY_MODEL_FLAVOR_INPUT="${WEEKLY_MODEL_FLAVOR:-}"
WEEKLY_MODEL_FLAVOR="${WEEKLY_MODEL_FLAVOR:-main}"
PACKAGE_STEM="csi1000_${WEEKLY_MODEL_FLAVOR}_model_packages"
if [ -z "${LOCAL_LATEST_PACKAGE:-}" ]; then
  if [ "${WEEKLY_MODEL_FLAVOR}" = "main" ]; then
    LOCAL_LATEST_PACKAGE="${LOCAL_MODEL_PACKAGE_DIR}/latest_csi1000_main_model_package.tar.gz"
  else
    LOCAL_LATEST_PACKAGE="${LOCAL_MODEL_PACKAGE_DIR}/latest_csi1000_${WEEKLY_MODEL_FLAVOR}_model_package.tar.gz"
  fi
fi
if [ -z "${LOCAL_WEEKLY_PRED_DIR:-}" ]; then
  if [ "${WEEKLY_MODEL_FLAVOR}" = "main" ]; then
    LOCAL_WEEKLY_PRED_DIR="${OPS_HOME}/preds/csi1000"
  else
    LOCAL_WEEKLY_PRED_DIR="${OPS_HOME}/preds/csi1000_${WEEKLY_MODEL_FLAVOR}"
  fi
fi
if [ "${WEEKLY_MODEL_FLAVOR}" != "main" ] && [ "${LOCAL_WEEKLY_PRED_DIR}" = "${OPS_HOME}/preds/csi1000" ]; then
  LOCAL_WEEKLY_PRED_DIR="${OPS_HOME}/preds/csi1000_${WEEKLY_MODEL_FLAVOR}"
fi
if [ "${WEEKLY_MODEL_FLAVOR}" != "main" ] && [ "${LOCAL_LATEST_PACKAGE}" = "${LOCAL_MODEL_PACKAGE_DIR}/latest_csi1000_main_model_package.tar.gz" ]; then
  LOCAL_LATEST_PACKAGE="${LOCAL_MODEL_PACKAGE_DIR}/latest_csi1000_${WEEKLY_MODEL_FLAVOR}_model_package.tar.gz"
fi

TRAIN_END="${WEEKLY_TRAIN_END:-2025-12-31}"
VALID_START="${WEEKLY_VALID_START:-2026-01-05}"
VALID_END="${WEEKLY_VALID_END:-}"

ssh_base=(ssh -p "${GPU_SSH_PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
scp_base=(scp -P "${GPU_SSH_PORT}")
if [ -n "${GPU_SSH_PASSWORD:-}" ]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "GPU_SSH_PASSWORD is set, but sshpass is not installed. Install sshpass or use SSH key login."
    exit 1
  fi
  export SSHPASS="${GPU_SSH_PASSWORD}"
  ssh_base=(sshpass -e "${ssh_base[@]}")
  scp_base=(sshpass -e "${scp_base[@]}")
fi

remote() {
  "${ssh_base[@]}" "${GPU_SSH}" "$@"
}

copy_to_gpu() {
  "${scp_base[@]}" "$1" "${GPU_SSH}:$2"
}

copy_from_gpu() {
  "${scp_base[@]}" "${GPU_SSH}:$1" "$2"
}

latest_calendar_date() {
  local cal="${QLIB_PROVIDER_URI}/calendars/day.txt"
  if [ ! -f "${cal}" ]; then
    echo "Missing local calendar: ${cal}" >&2
    return 1
  fi
  tail -1 "${cal}"
}

validate_local_archive() {
  if [ ! -f "${LOCAL_ARCHIVE}" ]; then
    echo "Missing local qlib archive: ${LOCAL_ARCHIVE}"
    exit 1
  fi
  local latest
  latest="${WEEKLY_TRAIN_INFER_DATE:-$(latest_calendar_date)}"
  echo "Local archive: ${LOCAL_ARCHIVE}"
  echo "Expected infer date: ${latest}"
  tar -xOf "${LOCAL_ARCHIVE}" qlib_bin/calendars/day.txt | tail -5
  if ! tar -xOf "${LOCAL_ARCHIVE}" qlib_bin/calendars/day.txt | tail -1 | grep -qx "${latest}"; then
    echo "Archive calendar latest does not match infer date ${latest}."
    exit 1
  fi
  echo "${latest}"
}

derive_valid_end() {
  local infer_date="$1"
  local offset="${WEEKLY_VALID_END_OFFSET:-2}"
  tar -xOf "${LOCAL_ARCHIVE}" qlib_bin/calendars/day.txt | awk -v infer="${infer_date}" -v offset="${offset}" '
    $0 <= infer { dates[++n] = $0 }
    END {
      idx = n - offset
      if (idx < 1) {
        exit 1
      }
      print dates[idx]
    }
  '
}

start_train() {
  local infer_date
  infer_date="$(validate_local_archive | tail -1)"
  local valid_end="${VALID_END:-$(derive_valid_end "${infer_date}")}"
  local run_id="${WEEKLY_TRAIN_RUN_ID:-$(date +%Y%m%d)_${infer_date//-/}}"
  local remote_incoming="${GPU_PROJECT_ROOT}/incoming"
  local remote_archive="${remote_incoming}/qlib_bin_${run_id}.tar.gz"
  local remote_release="${GPU_QDATA_ROOT}/releases/${run_id}"
  local remote_provider="${remote_release}/qlib_bin"
  local remote_log="${GPU_PROJECT_ROOT}/logs/weekly_train_${run_id}.log"

  echo "RUN_ID=${run_id}"
  echo "Training split:"
  echo "  train: config start -> ${TRAIN_END}"
  echo "  valid: ${VALID_START} -> ${valid_end}"
  echo "  infer: ${infer_date}"
  echo "  label_mode: ${WEEKLY_LABEL_MODE}"
  echo "  model_flavor: ${WEEKLY_MODEL_FLAVOR}"
  echo "Uploading scripts and data to GPU..."
  remote "mkdir -p '${remote_incoming}' '${GPU_PROJECT_ROOT}/logs' '${GPU_PAPER_HOME}/bin'"
  copy_to_gpu "${LOCAL_ARCHIVE}" "${remote_archive}"
  copy_to_gpu "${PAPER_HOME}/bin/gen_weekly_prod_configs.py" "${GPU_PAPER_HOME}/bin/gen_weekly_prod_configs.py"
  copy_to_gpu "${PAPER_HOME}/bin/train_weekly_main_models.sh" "${GPU_PAPER_HOME}/bin/train_weekly_main_models.sh"

  echo "Preparing remote Qlib release..."
  remote "rm -rf '${remote_release}' && mkdir -p '${remote_release}' && tar -xzf '${remote_archive}' -C '${remote_release}' && ln -sfn '${remote_provider}' '${GPU_QDATA_ROOT}/cn_data' && mkdir -p '${GPU_PAPER_HOME}/data' && ln -sfn '${remote_provider}' '${GPU_PAPER_HOME}/data/current' && tail -5 '${remote_provider}/calendars/day.txt'"

  echo "Starting remote weekly training with nohup..."
  remote "cd '${GPU_PAPER_HOME}' && chmod +x bin/train_weekly_main_models.sh bin/gen_weekly_prod_configs.py && nohup bash bin/train_weekly_main_models.sh '${run_id}' '${remote_provider}' '${TRAIN_END}' '${VALID_START}' '${valid_end}' '${infer_date}' '${WEEKLY_LABEL_MODE}' '${WEEKLY_MODEL_FLAVOR}' > '${remote_log}' 2>&1 & echo \$! > '${GPU_PROJECT_ROOT}/logs/weekly_train_${run_id}.pid'"

  echo
  echo "Started weekly training."
  echo "RUN_ID: ${run_id}"
  echo "Remote log: ${remote_log}"
  echo "Check progress:"
  echo "  ssh -p ${GPU_SSH_PORT} ${GPU_SSH} 'tail -f ${remote_log}'"
  echo "Collect after it finishes:"
  echo "  bash bin/weekly_gpu_train.sh collect ${run_id}"
}

collect_train() {
  local run_id="${1:-}"
  if [ -z "${run_id}" ]; then
    echo "collect requires RUN_ID"
    exit 2
  fi
  local remote_run_root="${GPU_PROJECT_ROOT}/weekly_train/${run_id}"
  if [ -z "${WEEKLY_MODEL_FLAVOR_INPUT}" ]; then
    local detected_package
    detected_package="$(remote "find '${remote_run_root}' -maxdepth 1 -type f -name 'csi1000_*_model_packages_${run_id}.tar.gz' -print | head -1" || true)"
    if [ -n "${detected_package}" ]; then
      local detected_file detected_flavor
      detected_file="$(basename "${detected_package}")"
      detected_flavor="${detected_file#csi1000_}"
      detected_flavor="${detected_flavor%_model_packages_${run_id}.tar.gz}"
      WEEKLY_MODEL_FLAVOR="${detected_flavor}"
      PACKAGE_STEM="csi1000_${WEEKLY_MODEL_FLAVOR}_model_packages"
      if [ "${WEEKLY_MODEL_FLAVOR}" != "main" ]; then
        LOCAL_WEEKLY_PRED_DIR="${OPS_HOME}/preds/csi1000_${WEEKLY_MODEL_FLAVOR}"
        LOCAL_LATEST_PACKAGE="${LOCAL_MODEL_PACKAGE_DIR}/latest_csi1000_${WEEKLY_MODEL_FLAVOR}_model_package.tar.gz"
      fi
    fi
  fi
  local remote_package="${remote_run_root}/${PACKAGE_STEM}_${run_id}.tar.gz"
  local local_archive_dir="${LOCAL_WEEKLY_ARCHIVE_DIR}/${run_id}"
  mkdir -p "${LOCAL_MODEL_PACKAGE_DIR}" "${LOCAL_WEEKLY_PRED_DIR}" "${local_archive_dir}"

  echo "Checking remote outputs..."
  remote "ls -lh '${remote_package}' && find '${remote_run_root}/preds/csi1000' -maxdepth 1 -type f -name '*.pkl' -print"

  copy_from_gpu "${remote_package}" "${LOCAL_MODEL_PACKAGE_DIR}/"
  copy_from_gpu "${remote_run_root}/preds/csi1000/xgb_csi1000_long_prod2026.pkl" "${LOCAL_WEEKLY_PRED_DIR}/"
  copy_from_gpu "${remote_run_root}/preds/csi1000/doubleensemble_csi1000_short_prod2026.pkl" "${LOCAL_WEEKLY_PRED_DIR}/"
  copy_from_gpu "${remote_run_root}/preds/csi1000/catboost_csi1000_long_prod2026.pkl" "${LOCAL_WEEKLY_PRED_DIR}/"
  copy_from_gpu "${remote_run_root}/logs/train_xgb_csi1000_long_prod2026.log" "${local_archive_dir}/" || true
  copy_from_gpu "${remote_run_root}/logs/train_doubleensemble_csi1000_short_prod2026.log" "${local_archive_dir}/" || true
  copy_from_gpu "${remote_run_root}/logs/train_catboost_csi1000_long_prod2026.log" "${local_archive_dir}/" || true

  ln -sfn "${LOCAL_MODEL_PACKAGE_DIR}/${PACKAGE_STEM}_${run_id}.tar.gz" "${LOCAL_LATEST_PACKAGE}"

  echo "Validating local prediction pkls..."
  "${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${LOCAL_WEEKLY_PRED_DIR}/xgb_csi1000_long_prod2026.pkl" \
    --pred "${LOCAL_WEEKLY_PRED_DIR}/doubleensemble_csi1000_short_prod2026.pkl" \
    --pred "${LOCAL_WEEKLY_PRED_DIR}/catboost_csi1000_long_prod2026.pkl"

  echo
  echo "Collected model package:"
  ls -lh "${LOCAL_MODEL_PACKAGE_DIR}/${PACKAGE_STEM}_${run_id}.tar.gz"
  echo "Latest model package symlink:"
  ls -lh "${LOCAL_LATEST_PACKAGE}"
  echo "Collected prediction pkls:"
  ls -lh "${LOCAL_WEEKLY_PRED_DIR}"/{xgb_csi1000_long_prod2026.pkl,doubleensemble_csi1000_short_prod2026.pkl,catboost_csi1000_long_prod2026.pkl}
}

case "${ACTION}" in
  start)
    start_train
    ;;
  collect)
    collect_train "${2:-}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
