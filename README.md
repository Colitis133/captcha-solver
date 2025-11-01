# CAPTCHA Dataset + PaddleOCR Finetuning Guide

This repository now focuses on generating synthetic CAPTCHA data and preparing label
manifests so you can fine-tune an open-source OCR stack such as PaddleOCR.

The previous in-house CRNN training code has been archived on the
`backup-crnn-legacy` branch should you need it in the future.

## Synthetic dataset

Use `generate_dataset.py` to produce the train/validation splits. By default it
creates 2,000 training images spread evenly across 20 styles and 500 validation
images of the "Mutant Hybrid" style.

```bash
python generate_dataset.py --out data
```

Fonts live under `fonts/` so generation works offline. Feel free to drop in
additional `.ttf` files—they are picked up automatically.

## Prepare PaddleOCR label files

PaddleOCR expects manifest files where each line contains the relative image
path and the corresponding label separated by a tab. Generate these manifests
after you create the dataset:

```bash
python prepare_paddle_labels.py \
  --train-dir data/train \
  --val-dir data/val \
  --base-dir . \
  --output-dir paddle_ocr/labels
```

This writes `paddle_ocr/labels/train_label.txt` and
`paddle_ocr/labels/val_label.txt`.

## Fine-tune PaddleOCR

1. Clone the [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)
	and follow their installation guide for your platform (GPU strongly
	recommended).
2. Copy the character dictionary provided here (`paddle_ocr/charset.txt`) to
	the PaddleOCR workspace or reference it via an absolute path.
3. Launch fine-tuning using their recognition config, overriding the data paths
	and charset. Example command:

	```bash
	export PADDLE_OCR_ROOT=/path/to/PaddleOCR
	python ${PADDLE_OCR_ROOT}/tools/train.py \
	  -c ${PADDLE_OCR_ROOT}/configs/rec/rec_mv3_none_bilstm_ctc.yml \
	  -o Global.pretrained_model=${PADDLE_OCR_ROOT}/pretrained/rec_mv3_none_bilstm_ctc/best_accuracy \
		  Global.save_model_dir=output/rec_mv3_finetune \
		  Global.character_dict_path=$(pwd)/paddle_ocr/charset.txt \
		  Train.dataset.data_dir=$(pwd) \
		  Train.dataset.label_file_list[0]=$(pwd)/paddle_ocr/labels/train_label.txt \
		  Eval.dataset.data_dir=$(pwd) \
		  Eval.dataset.label_file_list[0]=$(pwd)/paddle_ocr/labels/val_label.txt \
		  Global.max_text_length=8 \
		  Optimizer.lr.learning_rate=0.0003
	```

	Tweak the learning rate, batch size, and schedule to suit your hardware and
	convergence.

4. Export the recognition model with `tools/export_model.py` once validation
	performance is stable.

## Requirements

The Python dependencies required for dataset generation and label preparation
are minimal:

```
pip install -r requirements.txt
```

Install PaddleOCR and PaddlePaddle separately according to their official
instructions—they are not pulled in by the default requirements file.
