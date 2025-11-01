#!/usr/bin/env bash
set -euo pipefail

# Create venv, install deps, generate dataset, and prepare PaddleOCR labels.
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate synthetic dataset (2,000 train / 500 val by default)
python generate_dataset.py --out data

# Prepare PaddleOCR manifests
python prepare_paddle_labels.py --train-dir data/train --val-dir data/val --output-dir paddle_ocr/labels

echo "Done"
