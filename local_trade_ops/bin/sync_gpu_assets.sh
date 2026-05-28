#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

GPU_PROJECT_ROOT="${GPU_PROJECT_ROOT:-/root/autodl-tmp/llhh}"
LOCAL_ARCHIVE_DIR="${LOCAL_ARCHIVE_DIR:-${OPS_HOME}/archive/gpu}"
LOCAL_REPORT_ARCHIVE="${LOCAL_REPORT_ARCHIVE:-${LOCAL_ARCHIVE_DIR}/reports_smallcap/csi1000}"
LOCAL_STATE_ARCHIVE="${LOCAL_STATE_ARCHIVE:-${LOCAL_ARCHIVE_DIR}/paper_states}"

mkdir -p \
  "${LOCAL_PRED_DIR}" \
  "${PAPER_LOG_DIR}" \
  "${LOCAL_REPORT_ARCHIVE}" \
  "${LOCAL_STATE_ARCHIVE}"

echo "Sync main strategy prediction pkls..."
for pred in \
  xgb_csi1000_long_prod2026.pkl \
  doubleensemble_csi1000_short_prod2026.pkl \
  catboost_csi1000_long_prod2026.pkl
do
  rsync -avz -e "ssh -p ${GPU_SSH_PORT}" \
    "${GPU_SSH}:${GPU_PRED_DIR}/${pred}" \
    "${LOCAL_PRED_DIR}/"
done

echo "Sync key grid reports..."
for report in \
  focus_xgb_de_cat_topdrop.csv \
  focus_xgb_de_cat_capital.csv \
  focus_xgb_de_cat_summary.csv \
  focus_xgb_de_xgbmid_topdrop.csv \
  focus_xgb_de_xgbmid_capital.csv \
  focus_xgb_de_xgbmid_summary.csv \
  focus_xgb_de_topdrop.csv \
  focus_xgb_de_capital.csv \
  focus_xgb_de_summary.csv \
  focus_all5_topdrop.csv \
  focus_all5_capital.csv \
  focus_all5_summary.csv
do
  rsync -avz -e "ssh -p ${GPU_SSH_PORT}" \
    "${GPU_SSH}:${GPU_PROJECT_ROOT}/reports_smallcap/csi1000/${report}" \
    "${LOCAL_REPORT_ARCHIVE}/" || true
done

echo "Sync paper replay states and reports for the promoted main line..."
for name in \
  state_defensive_xgb_de_cat_3_1_1_10w \
  state_defensive_xgb_de_cat_3_1_1_20w \
  state_defensive_xgb_de_cat_3_1_1_100w \
  reports_defensive_xgb_de_cat_3_1_1_10w \
  reports_defensive_xgb_de_cat_3_1_1_20w \
  reports_defensive_xgb_de_cat_3_1_1_100w
do
  rsync -avz -e "ssh -p ${GPU_SSH_PORT}" \
    "${GPU_SSH}:${GPU_PROJECT_ROOT}/paper_trading_system/${name}/" \
    "${LOCAL_STATE_ARCHIVE}/${name}/" || true
done

echo "Local prediction files:"
ls -lh "${LOCAL_PRED_DIR}" || true

echo "Local report archive:"
find "${LOCAL_REPORT_ARCHIVE}" -maxdepth 1 -type f | sort || true

echo "Local state archive:"
find "${LOCAL_STATE_ARCHIVE}" -maxdepth 2 -type f | sort | head -80 || true
