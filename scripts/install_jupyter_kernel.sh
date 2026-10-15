#!/usr/bin/env bash
# Register Jupyter kernel for v2.0 notebooks (run once per machine).
set -euo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${V2_ROOT}/.venv-schubki-v2/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing ${V2_ROOT}/.venv-schubki-v2 — create with: python3.12 -m venv .venv-schubki-v2 && pip install -r requirements.txt"
  exit 1
fi
"$PY" -m pip install -q ipykernel jupyter
DISPLAY_NAME="SchubKI v2.0"
"$PY" -m ipykernel install --user --name schubki-v2 --display-name "$DISPLAY_NAME"
"$PY" -m ipykernel install --name schubki-v2 --display-name "$DISPLAY_NAME" --prefix="$V2_ROOT/.venv-schubki-v2"
echo "Jupyter kernelspec: name=schubki-v2  display_name=\"${DISPLAY_NAME}\""
echo "  user:    ~/Library/Jupyter/kernels/schubki-v2"
echo "  venv:    ${V2_ROOT}/.venv-schubki-v2/share/jupyter/kernels/schubki-v2"
echo ""
echo "Cursor / VS Code 若列表里没有该 kernel："
echo "  1. 命令面板 → Notebook: Select Notebook Kernel"
echo "  2. 选 Python Environments → Enter interpreter path →"
echo "     ${V2_ROOT}/.venv-schubki-v2/bin/python"
echo "  或选 Jupyter Kernel → SchubKI v2.0"
