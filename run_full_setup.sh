#!/usr/bin/env bash
set -euo pipefail

# Create venv, install deps, generate paired dataset, and build manifests.
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate synthetic dataset (paired noisy/clean images)
python generate_dataset.py --out data

# Build tab-separated manifests for the cleaner and OCR stages
python build_annotations.py --data-root data --out-dir annotations

echo "Done"
