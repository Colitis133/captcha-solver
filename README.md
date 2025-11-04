# CAPTCHA Cleaner + OCR Pipeline

A two-stage stack for solving real-world CAPTCHAs:

1. **Cleaner** – a ResNet34-based ResUNet that removes noise and normalises glyphs.
2. **OCR** – Microsoft TrOCR fine-tuned on the cleaner outputs.

The current focus is producing the best possible cleaner so TrOCR receives crisp, consistent text.

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Get the Kaggle Dataset

We rely entirely on the [Huge Captcha Dataset](https://www.kaggle.com/datasets/fournierp/huge-captcha-dataset). Download it into `data/real_captchas/` (see `run_full_setup.sh` for an automated flow). The archive already provides `train/`, `val/`, and `test/` splits with filenames encoding the ground-truth label (e.g. `abcdE.png`).

Counts after download (full set):

- `data/train`: ~157 k images
- `data/val`: ~19.6 k images
- `data/test`: ~19.6 k images

## 3. Create Paired Training Targets

The cleaner expects paired noisy/clean images. Use `prepare_clean_pairs.py` to subset the Kaggle data and render canonical targets with a consistent font, white background, and the same dimensions as the source image.

```bash
python prepare_clean_pairs.py \
  --source-root data \
  --out-root data/paired \
  --manifest-dir annotations \
  --train-count 50000 \
  --val-count 10000 \
  --overwrite
```

What this script does:

- Copies the requested number of raw CAPTCHAs into `data/paired/<split>/inputs/`.
- Re-renders each label (from the filename) with `render_clean_label`, producing a perfectly typeset target under `data/paired/<split>/targets/`.
- Writes manifests `annotations/cleaner_{train,val}.tsv` in the format `inputs/path.png<TAB>targets/path.png<TAB>LABEL`.

Adjust `--train-count`, `--val-count`, or add `--no-shuffle` / `--seed` to control sampling. Set `--overwrite` when regenerating pairs.

## 4. Train the Cleaner

Launch training with the generated manifests:

```bash
python train_cleaner.py \
  --train-manifest annotations/cleaner_train.tsv \
  --val-manifest annotations/cleaner_val.tsv \
  --image-root data/paired \
  --output-dir outputs/cleaner \
  --epochs 100 \
  --patience 20 \
  --encoder-pretrained \
  --perceptual-weight 0.1
```

Key details:

- **Architecture** – ResUNet with a ResNet34 encoder (optionally ImageNet-initialised via `--encoder-pretrained`) and a lightweight decoder. Use `--freeze-encoder` if you want to fine-tune only the decoder head.
- **Loss** – Composite `L = α·L1 + β·(1 - SSIM) + γ·L_perceptual`. The perceptual term leverages a frozen VGG16 feature extractor and focuses on structural fidelity rather than raw pixels.
- **Mixed precision** – Enabled automatically when CUDA is available.
- **Optimisation** – AdamW, ReduceLROnPlateau (patience 3), gradient clipping at 1.0, early stopping via `--patience`.

Progress and checkpoints:

- `outputs/cleaner/best_cleaner.pt` – best validation loss.
- `outputs/cleaner/last_cleaner.pt` – latest epoch.
- `outputs/cleaner/history.json` – per-epoch metrics (`train_loss`, `val_loss`, `val_psnr`, `val_ssim`, `train/val_perceptual` when enabled).

Resume with:

```bash
python train_cleaner.py \
  --train-manifest annotations/cleaner_train.tsv \
  --val-manifest annotations/cleaner_val.tsv \
  --image-root data/paired \
  --output-dir outputs/cleaner \
  --resume-checkpoint outputs/cleaner/best_cleaner.pt \
  --resume-optimizer
```

## 5. Inspect Outputs

- Visualise predictions on held-out inputs from `data/paired/val/inputs/` to confirm the cleaner produces uniform, readable text.
- Review `history.json` to ensure validation curves shadow training curves—large gaps flag overfitting or data leakage.

## 6. Next Steps

1. Run the trained cleaner across the raw Kaggle splits (train/val/test) to generate cleaned inputs for OCR.
2. Fine-tune TrOCR with `trocr_train.py` using the cleaned outputs and original labels.
3. Evaluate end-to-end with `trocr_eval.py`; package the pipeline (`cleaner -> TrOCR`) for inference.

## Troubleshooting

- **Perceptual loss downloads weights** – The first run pulls the VGG16 and ResNet34 weights from torchvision. Cache them locally if needed.
- **Cleaner overfits** – Increase sample counts, keep data augmentations on (handled inside `CleanerDataset`), or freeze the encoder for early epochs.
- **Validation stalls** – Ensure you actually generated the paired subset (check counts in `annotations/cleaner_*.tsv`) and optionally bump `--perceptual-weight` for more structural guidance.
- **Data loader bottlenecks** – Tune `--batch-size`, `--num-workers`, `--prefetch-factor`, and `--no-pin-memory`. Defaults target mid-tier GPUs (batch 16, workers 2, prefetch 2).
