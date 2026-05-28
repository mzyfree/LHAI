#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash bin/setup_env.sh [cpu|gpu]

Installs runtime dependencies into the Python environment configured by
config/env.local, or config/env.example if env.local does not exist.

Modes:
  cpu  Install standard CPU-friendly dependencies.
  gpu  Install the same Python dependencies. CUDA-specific torch wheels should
       be managed by the target machine image or installed manually if needed.
EOF
}

MODE="${1:-cpu}"
case "${MODE}" in
  cpu|gpu) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

set -a
source "${ENV_FILE}"
set +a

echo "Using PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${PAPER_HOME}/requirements.txt"

echo
echo "Running environment check..."
"${PYTHON_BIN}" "${PAPER_HOME}/bin/check_env.py"
