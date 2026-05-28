#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

execution_date="${1:-}"
if [[ -z "$execution_date" ]]; then
  echo "usage: bash bin/generate_and_send_wechat_snapshot.sh YYYY-MM-DD" >&2
  exit 2
fi

echo "Generate daily snapshot: ${execution_date}"
bash bin/generate_daily_snapshot.sh "$execution_date"

echo
echo "Send WeChat daily report..."
bash bin/send_wechat_snapshot.sh
