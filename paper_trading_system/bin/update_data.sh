#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_HOME="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PAPER_HOME}/config/env.local"
if [ ! -f "${ENV_FILE}" ]; then
  ENV_FILE="${PAPER_HOME}/config/env.example"
fi

set -a
source "${ENV_FILE}"
set +a

DATA_ROOT="${DATA_ROOT:-${PAPER_HOME}/data}"
DATA_RELEASES_DIR="${DATA_RELEASES_DIR:-${DATA_ROOT}/releases}"
DATA_CURRENT_LINK="${DATA_CURRENT_LINK:-${DATA_ROOT}/current}"
DATA_DOWNLOAD_URL="${DATA_DOWNLOAD_URL:-https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz}"
DATA_KEEP_RELEASES="${DATA_KEEP_RELEASES:-3}"
DATA_ARCHIVES_DIR="${DATA_ARCHIVES_DIR:-${DATA_ROOT}/archives}"
DATA_KEEP_ARCHIVES="${DATA_KEEP_ARCHIVES:-5}"
DATA_DOLTHUB_PATCH="${DATA_DOLTHUB_PATCH:-1}"
DATA_DOLTHUB_PATCH_STRICT="${DATA_DOLTHUB_PATCH_STRICT:-0}"
DATA_DOLTHUB_API_URL="${DATA_DOLTHUB_API_URL:-https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master}"
DATA_REQUIRED_DATE="${DATA_REQUIRED_DATE:-auto}"
DATA_MARKET_HOLIDAYS="${DATA_MARKET_HOLIDAYS:-2026-06-19}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_DIR="${DATA_RELEASES_DIR}/${STAMP}"
ARCHIVE="${DATA_ARCHIVES_DIR}/qlib_bin_${STAMP}.tar.gz"
TMP_DIR="${DATA_ROOT}/tmp_${STAMP}"

latest_calendar_date() {
  local qlib_dir="$1"
  local calendar="${qlib_dir}/calendars/day.txt"
  if [ ! -f "${calendar}" ]; then
    echo ""
    return 0
  fi
  awk 'NF {last=$0} END {print last}' "${calendar}"
}

resolve_required_date() {
  if [ "${DATA_REQUIRED_DATE}" != "auto" ] && [ -n "${DATA_REQUIRED_DATE}" ]; then
    echo "${DATA_REQUIRED_DATE}"
    return 0
  fi
  "${PYTHON_BIN:-python3}" "${PAPER_HOME}/bin/resolve_required_date.py" \
    --required-date "${DATA_REQUIRED_DATE}" \
    --holidays "${DATA_MARKET_HOLIDAYS}"
}

calendar_reaches_required() {
  local qlib_dir="$1"
  local required="$2"
  local latest
  latest="$(latest_calendar_date "${qlib_dir}")"
  [ -n "${latest}" ] && [ "${latest}" \> "${required}" -o "${latest}" = "${required}" ]
}

cleanup_old_data() {
  echo "Keeping latest ${DATA_KEEP_RELEASES} releases."
  find "${DATA_RELEASES_DIR}" -mindepth 1 -maxdepth 1 -type d | sort -r | tail -n +"$((DATA_KEEP_RELEASES + 1))" | while read -r old; do
    echo "Remove old release: ${old}"
    rm -rf "${old}"
  done

  echo "Keeping latest ${DATA_KEEP_ARCHIVES} archives."
  find "${DATA_ARCHIVES_DIR}" -mindepth 1 -maxdepth 1 -type f -name 'qlib_bin_*.tar.gz' | sort -r | tail -n +"$((DATA_KEEP_ARCHIVES + 1))" | while read -r old; do
    echo "Remove old archive: ${old}"
    rm -f "${old}"
  done
}

REQUIRED_DATE="$(resolve_required_date)"

mkdir -p "${DATA_ROOT}" "${DATA_RELEASES_DIR}" "${DATA_ARCHIVES_DIR}"

echo "Required Qlib latest date: ${REQUIRED_DATE}"
echo "Current data: ${DATA_CURRENT_LINK}"
CURRENT_LATEST="$(latest_calendar_date "${DATA_CURRENT_LINK}")"
echo "Current latest date: ${CURRENT_LATEST:-missing}"
if calendar_reaches_required "${DATA_CURRENT_LINK}" "${REQUIRED_DATE}"; then
  echo "Local Qlib data already reaches ${REQUIRED_DATE}. Skip download."
  echo "Calendar tail:"
  tail -5 "${DATA_CURRENT_LINK}/calendars/day.txt"
  exit 0
fi

mkdir -p "${TMP_DIR}"

echo "Download: ${DATA_DOWNLOAD_URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "${ARCHIVE}" "${DATA_DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE}" "${DATA_DOWNLOAD_URL}"
else
  echo "Neither curl nor wget is available."
  exit 1
fi

echo "Extract to temp: ${TMP_DIR}"
mkdir -p "${TMP_DIR}/cn_data"
tar -zxf "${ARCHIVE}" -C "${TMP_DIR}/cn_data" --strip-components=1

if [ ! -d "${TMP_DIR}/cn_data/calendars" ] || [ ! -d "${TMP_DIR}/cn_data/features" ] || [ ! -d "${TMP_DIR}/cn_data/instruments" ]; then
  echo "Extracted data does not look like a Qlib cn_data directory."
  exit 1
fi

RELEASE_LATEST="$(latest_calendar_date "${TMP_DIR}/cn_data")"
echo "Release latest date: ${RELEASE_LATEST:-missing}"

if calendar_reaches_required "${TMP_DIR}/cn_data" "${REQUIRED_DATE}"; then
  echo "GitHub release already reaches ${REQUIRED_DATE}. DoltHub patch is not needed."
elif [ "${DATA_DOLTHUB_PATCH}" = "1" ]; then
  echo "GitHub release is behind ${REQUIRED_DATE}; patch release data from DoltHub..."
  PATCH_CMD=(
    "${PYTHON_BIN:-python3}"
    "${PAPER_HOME}/bin/patch_qlib_from_dolthub.py"
    --qlib-dir "${TMP_DIR}/cn_data"
    --work-dir "${DATA_ROOT}/dolthub_patch/${STAMP}"
    --api-url "${DATA_DOLTHUB_API_URL}"
    --python-bin "${PYTHON_BIN:-python3}"
  )
  if [ -n "${QLIB_SRC:-}" ]; then
    PATCH_CMD+=(--qlib-src "${QLIB_SRC}")
  fi
  if ! "${PATCH_CMD[@]}"; then
    echo "DoltHub patch failed."
    exit 1
  fi
else
  echo "GitHub release is behind ${REQUIRED_DATE}, and DoltHub patch is disabled by DATA_DOLTHUB_PATCH=${DATA_DOLTHUB_PATCH}."
  exit 1
fi

FINAL_LATEST="$(latest_calendar_date "${TMP_DIR}/cn_data")"
echo "Final candidate latest date: ${FINAL_LATEST:-missing}"
if ! calendar_reaches_required "${TMP_DIR}/cn_data" "${REQUIRED_DATE}"; then
  echo "Candidate Qlib data still does not reach required date ${REQUIRED_DATE}."
  echo "Do not trade stale data. Try again after the upstream data source updates."
  exit 1
fi

echo "Normalize index instrument end dates to ${FINAL_LATEST}..."
"${PYTHON_BIN:-python3}" "${PAPER_HOME}/bin/patch_qlib_from_dolthub.py" \
  --qlib-dir "${TMP_DIR}/cn_data" \
  --work-dir "${DATA_ROOT}/dolthub_patch/${STAMP}" \
  --normalize-instruments-only

mv "${TMP_DIR}/cn_data" "${RELEASE_DIR}"
rmdir "${TMP_DIR}" 2>/dev/null || true

echo "Switch current link: ${DATA_CURRENT_LINK} -> ${RELEASE_DIR}"
ln -sfn "${RELEASE_DIR}" "${DATA_CURRENT_LINK}"

echo "${STAMP}" > "${DATA_ROOT}/CURRENT_VERSION"
ln -sfn "${ARCHIVE}" "${DATA_ARCHIVES_DIR}/latest_qlib_bin.tar.gz"

echo "Archive kept: ${ARCHIVE}"
echo "Latest archive link: ${DATA_ARCHIVES_DIR}/latest_qlib_bin.tar.gz"

cleanup_old_data

echo "Data update complete."
echo "Current data: ${DATA_CURRENT_LINK}"
if [ -f "${DATA_CURRENT_LINK}/calendars/day.txt" ]; then
  echo "Calendar tail:"
  tail -5 "${DATA_CURRENT_LINK}/calendars/day.txt"
fi
