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

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_DIR="${DATA_RELEASES_DIR}/${STAMP}"
ARCHIVE="${DATA_ARCHIVES_DIR}/qlib_bin_${STAMP}.tar.gz"
TMP_DIR="${DATA_ROOT}/tmp_${STAMP}"

mkdir -p "${DATA_ROOT}" "${DATA_RELEASES_DIR}" "${DATA_ARCHIVES_DIR}" "${TMP_DIR}"

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

mv "${TMP_DIR}/cn_data" "${RELEASE_DIR}"
rmdir "${TMP_DIR}" 2>/dev/null || true

echo "Switch current link: ${DATA_CURRENT_LINK} -> ${RELEASE_DIR}"
ln -sfn "${RELEASE_DIR}" "${DATA_CURRENT_LINK}"

echo "${STAMP}" > "${DATA_ROOT}/CURRENT_VERSION"
ln -sfn "${ARCHIVE}" "${DATA_ARCHIVES_DIR}/latest_qlib_bin.tar.gz"

echo "Archive kept: ${ARCHIVE}"
echo "Latest archive link: ${DATA_ARCHIVES_DIR}/latest_qlib_bin.tar.gz"

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

echo "Data update complete."
echo "Current data: ${DATA_CURRENT_LINK}"
if [ -f "${DATA_CURRENT_LINK}/calendars/day.txt" ]; then
  echo "Calendar tail:"
  tail -5 "${DATA_CURRENT_LINK}/calendars/day.txt"
fi
