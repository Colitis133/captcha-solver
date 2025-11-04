#!/usr/bin/env bash
set -euo pipefail

# Environment bootstrap for the cleaner + TrOCR pipeline.
# Usage:
#   ./setup_env.sh                       # create venv, install deps, build paired dataset + manifests
#   ./setup_env.sh --no-pairs            # skip paired dataset generation (just set up env)
#   ./setup_env.sh --train-count 40000   # override default pair counts
#   ./setup_env.sh --val-count 8000
#   ./setup_env.sh --no-manifests        # alias for --no-pairs (backwards compatibility)

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

DO_PAIRS=true
TRAIN_COUNT=50000
VAL_COUNT=10000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pairs|--no-manifests) DO_PAIRS=false; shift ;;
    --train-count) TRAIN_COUNT="$2"; shift 2 ;;
    --val-count) VAL_COUNT="$2"; shift 2 ;;
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

if $DO_PAIRS; then
  if [[ ! -d data/train || ! -d data/val ]]; then
    echo "Expected Kaggle dataset under ./data/train and ./data/val. Please extract it before running this script." >&2
    exit 1
  fi
  echo "Preparing paired cleaner targets (train=$TRAIN_COUNT, val=$VAL_COUNT)..."
  $PY prepare_clean_pairs.py \
    --source-root data \
    --out-root data/paired \
    --manifest-dir annotations \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --overwrite
else
  echo "Skipping paired dataset generation (--no-pairs)"
fi

echo "Environment setup complete. Activate with: source $VENV_DIR/bin/activate"
