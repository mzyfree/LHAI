#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/import_pred.sh adarnn SOURCE
  bash bin/import_pred.sh xgb_short SOURCE
  bash bin/import_pred.sh xgb_long SOURCE

SOURCE can be a local file path or an scp-style remote path:
  /path/to/pred.pkl
  user@host:/root/autodl-tmp/llhh/preds/adarnn_short_2026_2026_ytd_prod_like.pkl
EOF
}

if [ "$#" -ne 2 ]; then
  usage
  exit 2
fi

MODEL="$1"
SOURCE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

set -a
source "${ENV_FILE}"
set +a

case "${MODEL}" in
  adarnn)
    TARGET="${ADARNN_SHORT_PRED}"
    ;;
  xgb_short)
    TARGET="${XGB_SHORT_PRED}"
    ;;
  xgb_long)
    TARGET="${XGB_LONG_PRED}"
    ;;
  *)
    echo "Unknown model: ${MODEL}"
    usage
    exit 2
    ;;
esac

mkdir -p "$(dirname "${TARGET}")"

if [[ "${SOURCE}" == *:* && "${SOURCE}" != /* ]]; then
  scp "${SOURCE}" "${TARGET}"
else
  cp "${SOURCE}" "${TARGET}"
fi

ls -lh "${TARGET}"
echo "Imported ${MODEL} prediction -> ${TARGET}"
