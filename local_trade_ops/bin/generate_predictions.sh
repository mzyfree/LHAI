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

MODE="${PREDICTION_MODE:-skip}"
case "${MODE}" in
  sync)
    echo "Prediction mode: sync latest pkls from GPU."
    bash "${OPS_HOME}/bin/sync_predictions.sh"
    ;;
  command)
    if [ -z "${PREDICTION_CMD:-}" ]; then
      echo "PREDICTION_MODE=command requires PREDICTION_CMD." >&2
      exit 2
    fi
    echo "Prediction mode: command."
    bash -lc "${PREDICTION_CMD}"
    ;;
  skip)
    echo "Prediction mode: skip generation; validate existing pkls only."
    ;;
  *)
    echo "Unknown PREDICTION_MODE=${MODE}. Use sync, command, or skip." >&2
    exit 2
    ;;
esac

for pred in "${PRED_A}" "${PRED_B}" "${PRED_C}"; do
  if [ ! -s "${pred}" ]; then
    echo "Missing or empty prediction pkl: ${pred}" >&2
    exit 3
  fi
done

echo "Prediction pkls ready:"
ls -lh "${PRED_A}" "${PRED_B}" "${PRED_C}"

if [ "${STRICT_PREDICTION_DATE_CHECK:-1}" = "0" ] || [ "${STRICT_PREDICTION_DATE_CHECK:-1}" = "false" ]; then
  "${PYTHON_BIN}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${PRED_A}" \
    --pred "${PRED_B}" \
    --pred "${PRED_C}" \
    --allow-stale
else
  "${PYTHON_BIN}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
    --provider-uri "${QLIB_PROVIDER_URI}" \
    --pred "${PRED_A}" \
    --pred "${PRED_B}" \
    --pred "${PRED_C}"
fi
