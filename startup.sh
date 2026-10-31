#!/usr/bin/env bash
# Sets up a virtual environment (first run only) and launches SolarSmart AI.
set -e
cd "$(dirname "$0")"

# Pick a Python interpreter
PY=$(command -v python3 || command -v python || command -v py || true)
if [ -z "$PY" ]; then
    echo "[ERROR] Python not found. Install Python 3.10+ and retry." >&2
    exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi

# Locate venv python (Windows: Scripts, Linux/macOS: bin)
if [ -x ".venv/Scripts/python.exe" ]; then
    VPY=".venv/Scripts/python.exe"
else
    VPY=".venv/bin/python"
fi

# Install dependencies only when requirements change
if [ ! -f ".venv/.deps_installed" ] || [ requirements.txt -nt ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    "$VPY" -m pip install --upgrade pip
    "$VPY" -m pip install -r requirements.txt
    touch .venv/.deps_installed
fi

echo "Starting SolarSmart AI at http://localhost:8501 (Ctrl+C to stop)"
exec "$VPY" -m streamlit run app.py
