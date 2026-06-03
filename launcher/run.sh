#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    BOOTSTRAP_PYTHON="python"
else
    echo "Error: Python 3.10+ is not installed or not in PATH."
    exit 1
fi

if [ ! -x "venv/bin/python" ]; then
    echo "Creating virtual environment..."
    "$BOOTSTRAP_PYTHON" -m venv venv
fi

VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"

echo "Preparing runtime dependencies..."
"$VENV_PYTHON" setup_env.py --torch auto

exec "$VENV_PYTHON" main.py "$@"
