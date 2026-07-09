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

SHORT_HOLD_PRED_DIR="${SHORT_HOLD_PRED_DIR:-${OPS_HOME}/preds/csi1000_short_hold_v2}"
SHORT_HOLD_MODEL_PACKAGE="${SHORT_HOLD_MODEL_PACKAGE:-${OPS_HOME}/model_packages/latest_csi1000_short_hold_v2_model_package.tar.gz}"
SHORT_HOLD_MODEL_EXTRACT_DIR="${SHORT_HOLD_MODEL_EXTRACT_DIR:-${OPS_HOME}/model_packages/extracted}"
SHORT_HOLD_AUTO_PREDICT="${SHORT_HOLD_AUTO_PREDICT:-1}"

SHORT_HOLD_PRED_A="${SHORT_HOLD_PRED_A:-${SHORT_HOLD_PRED_DIR}/xgb_csi1000_long_prod2026.pkl}"
SHORT_HOLD_PRED_B="${SHORT_HOLD_PRED_B:-${SHORT_HOLD_PRED_DIR}/doubleensemble_csi1000_short_prod2026.pkl}"
SHORT_HOLD_PRED_C="${SHORT_HOLD_PRED_C:-${SHORT_HOLD_PRED_DIR}/catboost_csi1000_long_prod2026.pkl}"

validate_short_hold_predictions() {
  "${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${SHORT_HOLD_PRED_A}" \
    --pred "${SHORT_HOLD_PRED_B}" \
    --pred "${SHORT_HOLD_PRED_C}"
}

echo "===== short-hold v2 prediction preparation ====="
echo "Qlib provider: ${QLIB_PROVIDER_URI}"
echo "Short-hold model package: ${SHORT_HOLD_MODEL_PACKAGE}"
echo "Short-hold prediction dir: ${SHORT_HOLD_PRED_DIR}"

if validate_short_hold_predictions; then
  echo "Short-hold v2 predictions already align with current Qlib data."
  exit 0
fi

if [ "${SHORT_HOLD_AUTO_PREDICT}" = "0" ] || [ "${SHORT_HOLD_AUTO_PREDICT}" = "false" ]; then
  echo "Short-hold v2 predictions are stale, and SHORT_HOLD_AUTO_PREDICT=${SHORT_HOLD_AUTO_PREDICT}." >&2
  echo "Run short-hold inference first or set SHORT_HOLD_AUTO_PREDICT=1." >&2
  exit 2
fi

if [ ! -s "${SHORT_HOLD_MODEL_PACKAGE}" ]; then
  echo "Missing short-hold v2 model package: ${SHORT_HOLD_MODEL_PACKAGE}" >&2
  exit 3
fi

echo
echo "Short-hold v2 predictions are stale or missing; run local inference now."
mkdir -p "${SHORT_HOLD_PRED_DIR}"
"${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/predict_from_model_package.py" \
  --provider-uri "${QLIB_PROVIDER_URI}" \
  --package "${SHORT_HOLD_MODEL_PACKAGE}" \
  --extract-dir "${SHORT_HOLD_MODEL_EXTRACT_DIR}" \
  --output-dir "${SHORT_HOLD_PRED_DIR}"

echo
echo "Re-validate refreshed short-hold v2 predictions."
validate_short_hold_predictions
