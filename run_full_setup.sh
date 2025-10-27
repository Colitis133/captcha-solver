#!/usr/bin/env bash
set -euo pipefail

# Create venv, install deps, generate dataset, run smoke test, and short training run
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate synthetic dataset (2000 images default)
python generate_dataset.py --total 2000 --out data

# Smoke test
python -m captcha_solver.smoke_test

# Short training run
python -m captcha_solver.train --train-dir data/train --val-dir data/val --epochs 2 --batch-size 8

echo "Done"
