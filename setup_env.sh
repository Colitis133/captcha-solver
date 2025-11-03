#!/usr/bin/env bash
set -euo pipefail

# Lightweight, idempotent environment setup for the TrOCR-based CAPTCHA solver.
# Usage:
#   ./setup_env.sh                 # create venv, install deps, generate data, build manifests
#   ./setup_env.sh --no-data       # skip dataset generation
#   ./setup_env.sh --no-manifests  # skip TSV manifest creation

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

DO_DATA=true
DO_MANIFESTS=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-data) DO_DATA=false; shift ;;
    --no-manifests) DO_MANIFESTS=false; shift ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

PY=python3

if ! command -v $PY >/dev/null 2>&1; then
  echo "$PY not found; please install Python 3.8+ and retry." >&2
  exit 1
fi

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv in $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found in $ROOT_DIR" >&2
  exit 1
fi

if $DO_DATA; then
  echo "Generating synthetic dataset (default 2000 train / 500 val images under ./data)..."
  $PY generate_dataset.py --out data
else
  echo "Skipping dataset generation (--no-data)"
fi

if $DO_MANIFESTS; then
  echo "Building TSV manifests under ./annotations..."
  $PY build_annotations.py --data-root data --out-dir annotations
else
  echo "Skipping manifest generation (--no-manifests)"
fi

echo "Environment setup complete. Activate with: source $VENV_DIR/bin/activate"
