#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run PaddleOCR recognition finetuning. This script does not install
# PaddleOCR; run it inside a PaddleOCR environment where the repo is checked out.

if [ -z "${PADDLE_OCR_ROOT:-}" ]; then
  echo "Please set PADDLE_OCR_ROOT to your PaddleOCR checkout, e.g."
  echo "  export PADDLE_OCR_ROOT=~/code/PaddleOCR"
  exit 1
fi

# Edit these if you want different training options
PRETRAINED_MODEL="${PADDLE_OCR_ROOT}/pretrained/rec_mv3_none_bilstm_ctc/best_accuracy"
SAVE_DIR="$(pwd)/output/rec_mv3_finetune"
CHARSET="$(pwd)/paddle_ocr/charset.txt"
TRAIN_LABELS="$(pwd)/paddle_ocr/labels/train_label.txt"
VAL_LABELS="$(pwd)/paddle_ocr/labels/val_label.txt"

python "$PADDLE_OCR_ROOT/tools/train.py" \
  -c "$PADDLE_OCR_ROOT/configs/rec/rec_mv3_none_bilstm_ctc.yml" \
  -o Global.pretrained_model="$PRETRAINED_MODEL" \
     Global.save_model_dir="$SAVE_DIR" \
     Global.character_dict_path="$CHARSET" \
     Train.dataset.data_dir="$(pwd)" \
     Train.dataset.label_file_list[0]="$TRAIN_LABELS" \
     Eval.dataset.data_dir="$(pwd)" \
     Eval.dataset.label_file_list[0]="$VAL_LABELS" \
     Global.max_text_length=8 \
     Optimizer.lr.learning_rate=0.0003
