#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

execution_date="${1:-}"
if [[ -z "$execution_date" ]]; then
  echo "usage: bash bin/generate_and_send_short_hold_snapshot.sh YYYY-MM-DD" >&2
  exit 2
fi

ENV_FILE="config/env.local"
if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="config/env.local.example"
fi

set -a
source "${ENV_FILE}"
set +a

echo "Generate short-hold snapshot: ${execution_date}"
bash bin/short_hold_daily_snapshot.sh "${execution_date}"

snapshot_dir="${SHORT_HOLD_SNAPSHOT_DIR:-${OPS_HOME:-$(pwd)}/short_hold_daily_snapshots}"
snapshot_path="${snapshot_dir}/short_hold_snapshot_${execution_date}.json"

echo
echo "Send WeChat short-hold report..."
bash bin/send_wechat_snapshot.sh --snapshot "${snapshot_path}"
