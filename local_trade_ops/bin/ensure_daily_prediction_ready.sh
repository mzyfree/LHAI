#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_HOME="$(cd "${OPS_HOME}/.." && pwd)"
PAPER_HOME="${PROJECT_HOME}/paper_trading_system"
ENV_FILE="${OPS_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${OPS_HOME}/config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

MARKER_PATH="${DAILY_PREP_STATUS_PATH:-${OPS_HOME}/state/daily_prediction_ready.json}"
TODAY_LOCAL="$("${PYTHON_BIN:-python3}" - <<'PY'
from datetime import datetime

print(datetime.now().strftime("%Y-%m-%d"))
PY
)"

marker_matches_today() {
  "${PYTHON_BIN:-python3}" - "${MARKER_PATH}" "${TODAY_LOCAL}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
today = sys.argv[2]
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("run_date") == today else 1)
PY
}

validate_predictions() {
  if [ "${STRICT_PREDICTION_DATE_CHECK:-1}" = "0" ] || [ "${STRICT_PREDICTION_DATE_CHECK:-1}" = "false" ]; then
    "${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
      --provider-uri "${QLIB_PROVIDER_URI}" \
      --pred "${PRED_A}" \
      --pred "${PRED_B}" \
      --pred "${PRED_C}" \
      --allow-stale
  else
    "${PYTHON_BIN:-python3}" "${OPS_HOME}/bin/validate_prediction_dates.py" \
      --provider-uri "${QLIB_PROVIDER_URI}" \
      --pred "${PRED_A}" \
      --pred "${PRED_B}" \
      --pred "${PRED_C}"
  fi
}

write_marker() {
  mkdir -p "$(dirname "${MARKER_PATH}")"
  "${PYTHON_BIN:-python3}" - "${MARKER_PATH}" "${TODAY_LOCAL}" "${QLIB_PROVIDER_URI}" "${PRED_A}" "${PRED_B}" "${PRED_C}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

marker, run_date, provider, *preds = sys.argv[1:]
calendar = Path(provider) / "calendars" / "day.txt"
dates = [line.strip() for line in calendar.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "run_date": run_date,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "provider_uri": provider,
    "latest_calendar_date": dates[-1] if dates else "",
    "predictions": preds,
}
Path(marker).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

if marker_matches_today; then
  echo "===== shared data/prediction preparation ====="
  echo "Found today's preparation marker: ${MARKER_PATH}"
  if validate_predictions; then
    echo "Shared data/prediction layer already ready; skip update and inference."
    exit 0
  fi
  echo "Today's marker exists but validation failed; rerun preparation."
fi

echo "===== STEP 1/2 update local Qlib data ====="
bash "${PAPER_HOME}/bin/update_data.sh"

echo
echo "===== STEP 2/2 validate or generate predictions ====="
bash "${OPS_HOME}/bin/generate_predictions.sh"

write_marker
echo "Wrote shared preparation marker: ${MARKER_PATH}"
