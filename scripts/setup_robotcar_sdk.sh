#!/usr/bin/env bash
set -euo pipefail
V2_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${V2_ROOT}"
DEST="${ROOT}/third_party/robotcar-dataset-sdk"
if [[ -d "${DEST}/models" ]]; then
  echo "Already present: ${DEST}"
  exit 0
fi
mkdir -p "${ROOT}/third_party"
git clone --depth 1 https://github.com/ori-mrg/robotcar-dataset-sdk.git "${DEST}"
echo "Cloned to ${DEST}"
echo "Set ROBOTCAR_SDK=${DEST} if using a custom location."
