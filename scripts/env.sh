#!/usr/bin/env bash
# Source from v2.0/:  source scripts/env.sh
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_VENV="${V2_ROOT}/.venv-schubki-v2"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="${V2_ROOT}/src:${PYTHONPATH}"
export HLOC_ROOT="${V2_ROOT}/third_party/Hierarchical-Localization"
export TORCH_HOME="${V2_ROOT}/model_cache/torch_hub"
if [[ -f "${V2_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${V2_VENV}/bin/activate"
elif [[ -f "${V2_ROOT}/.venv/bin/activate" ]]; then
  # legacy name
  source "${V2_ROOT}/.venv/bin/activate"
fi

if [[ -x "${V2_VENV}/bin/python" ]]; then
  _py="$("${V2_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${VIRTUAL_ENV:-}" != "${V2_VENV}" ]]; then
    echo "Note: v2.0 venv not active (VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>})." >&2
    echo "  Use: source ${V2_VENV}/bin/activate  &&  python -m pip install …" >&2
  elif [[ "${_py}" != "3.12" ]]; then
    echo "Warning: expected Python 3.12 in ${V2_VENV}, got ${_py}." >&2
  fi
  unset _py
fi
