#!/usr/bin/env bash
# Apply SchubKI-specific hloc patches (MPS device + calibrated two-view triangulation).
set -euo pipefail
V2_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_HLOC="${V2_ROOT}/third_party/Hierarchical-Localization"
PATCH="${V2_ROOT}/patches/hloc"

# Ignore stale HLOC_ROOT from an old monorepo layout (../hloc/...).
if [[ -n "${HLOC_ROOT:-}" && -d "${HLOC_ROOT}/hloc" ]]; then
  HLOC="${HLOC_ROOT}"
elif [[ -d "${DEFAULT_HLOC}/hloc" ]]; then
  HLOC="${DEFAULT_HLOC}"
else
  echo "hloc not found under ${DEFAULT_HLOC} (HLOC_ROOT=${HLOC_ROOT:-<unset>})." >&2
  echo "Run: git submodule update --init --recursive  # or: bash scripts/setup_hloc.sh" >&2
  exit 1
fi
cp "${PATCH}/utils/device.py" "${HLOC}/hloc/utils/device.py"
cp "${PATCH}/extract_features.py" "${HLOC}/hloc/extract_features.py"
cp "${PATCH}/match_features.py" "${HLOC}/hloc/match_features.py"
cp "${PATCH}/pairs_from_retrieval.py" "${HLOC}/hloc/pairs_from_retrieval.py"
cp "${PATCH}/triangulation.py" "${HLOC}/hloc/triangulation.py"
echo "Applied SchubKI hloc patches to ${HLOC}"
