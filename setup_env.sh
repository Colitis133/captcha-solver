#!/usr/bin/env bash
set -euo pipefail

# Lightweight, idempotent environment setup for training the CAPTCHA solver.
# Usage:
#   ./setup_env.sh            # create venv, install deps, generate data, smoke test, short train
#   ./setup_env.sh --no-data  # skip dataset generation
#   ./setup_env.sh --no-smoke # skip smoke test
#   ./setup_env.sh --no-train # skip the short training run

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

DO_DATA=true
DO_SMOKE=true
DO_TRAIN=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-data) DO_DATA=false; shift ;;
    --no-smoke) DO_SMOKE=false; shift ;;
    --no-train) DO_TRAIN=false; shift ;;
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
  echo "Generating synthetic dataset (default 2000 images under ./data)..."
  $PY generate_dataset.py --total 2000 --out data
else
  echo "Skipping dataset generation (--no-data)"
fi

if $DO_SMOKE; then
  echo "Running smoke test..."
  $PY -m captcha_solver.smoke_test
else
  echo "Skipping smoke test (--no-smoke)"
fi

if $DO_TRAIN; then
  echo "Starting a short training run (2 epochs, batch 8) to validate training setup..."
  $PY -m captcha_solver.train --train-dir data/train --val-dir data/val --epochs 2 --batch-size 8 || {
    echo "Short training run failed — environment may still be usable for data generation/smoke tests." >&2
  }
else
  echo "Skipping training run (--no-train)"
fi

echo "Environment setup complete. Activate with: source $VENV_DIR/bin/activate"
