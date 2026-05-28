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

mkdir -p "${LOCAL_PRED_DIR}" "${PAPER_LOG_DIR}"

for pred in \
  xgb_csi1000_long_prod2026.pkl \
  doubleensemble_csi1000_short_prod2026.pkl \
  catboost_csi1000_long_prod2026.pkl
do
  rsync -avz -e "ssh -p ${GPU_SSH_PORT}" \
    "${GPU_SSH}:${GPU_PRED_DIR}/${pred}" \
    "${LOCAL_PRED_DIR}/"
done

ls -lh "${LOCAL_PRED_DIR}"
