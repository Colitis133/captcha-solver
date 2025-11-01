#!/usr/bin/env bash
set -euo pipefail

# Helper that runs the finetune wrapper once PADDLE_OCR_ROOT is set.
# Use on the GPU host after installing PaddlePaddle and PaddleOCR deps.

if [ -z "${PADDLE_OCR_ROOT:-}" ]; then
  echo "Please set PADDLE_OCR_ROOT to your PaddleOCR checkout, e.g.:"
  echo "  export PADDLE_OCR_ROOT=$(pwd)/paddle_ocr_repo"
  exit 1
fi

# Ensure manifests exist
if [ ! -f paddle_ocr/labels/train_label.txt ]; then
  echo "Train manifest missing; run prepare_paddle_labels.py first."
  exit 1
fi

# Launch the wrapper
echo "Launching PaddleOCR finetune using $PADDLE_OCR_ROOT"
./paddle_ocr/run_finetune.sh
