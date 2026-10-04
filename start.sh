#!/usr/bin/env bash
# LetterEye AI – macOS / Linux launcher.
# The first start installs Python and all packages (a few minutes); later starts are fast.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "Installing uv, the Python package manager – one time only …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Note: Ollama was not found. LetterEye needs it for the local AI models."
  echo "The setup assistant will help you – or get it now from https://ollama.com/download"
fi

extras=()
# A native window on Linux needs Qt (WebKitGTK via python-gi also works if installed system-wide).
if [ "$(uname -s)" = "Linux" ] && [ "${LETTEREYE_QT:-0}" = "1" ]; then
  extras=(--extra qt)
fi

exec uv run --python 3.12 ${extras[@]+"${extras[@]}"} lettereye "$@"
