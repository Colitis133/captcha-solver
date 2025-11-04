#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the environment and prepare paired targets from the Kaggle dataset.
# Usage: ./run_full_setup.sh [--train-count N] [--val-count M]

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

TRAIN_COUNT=50000
VAL_COUNT=10000

while [[ $# -gt 0 ]]; do
	case "$1" in
		--train-count)
			TRAIN_COUNT="$2"; shift 2 ;;
		--val-count)
			VAL_COUNT="$2"; shift 2 ;;
		-h|--help)
			echo "Usage: $0 [--train-count N] [--val-count M]"
			echo "Requires the Huge Captcha Dataset extracted under ./data/train and ./data/val."
			exit 0 ;;
		*)
			echo "Unknown argument: $1" >&2
			exit 1 ;;
	esac
done

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -d data/train || ! -d data/val ]]; then
	echo "Expected Kaggle dataset under ./data/train and ./data/val. Please extract it before running this script." >&2
	exit 1
fi

python prepare_clean_pairs.py \
	--source-root data \
	--out-root data/paired \
	--manifest-dir annotations \
	--train-count "$TRAIN_COUNT" \
	--val-count "$VAL_COUNT" \
	--overwrite

echo "Environment ready. Paired dataset written to data/paired and manifests to annotations/."
