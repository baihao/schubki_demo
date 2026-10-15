#!/usr/bin/env bash
set -euo pipefail
V2_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${V2_ROOT}/third_party/Hierarchical-Localization"
if [[ -d "${DEST}/hloc" ]]; then
  echo "Already present: ${DEST}"
else
  mkdir -p "${V2_ROOT}/third_party"
  git clone --depth 1 https://github.com/cvg/Hierarchical-Localization.git "${DEST}"
fi
bash "${V2_ROOT}/scripts/apply_schubki_hloc_patches.sh"
