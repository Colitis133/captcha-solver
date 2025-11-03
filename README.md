# CAPTCHA Cleaner + OCR Pipeline

This project builds a two-stage CAPTCHA solver:

1. **Cleaner** – a U-Net that removes noise, normalises fonts, and reconstructs
	 broken characters.
2. **OCR** – Microsoft’s TrOCR, which reads the cleaned images once the cleaner
	 is trained.

The first milestone is training a rock-solid cleaner so TrOCR receives
consistent inputs. Follow every step in order—skipping stages invites
overfitting or brittle models.

## 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Synthesize Paired Captchas

Generate noisy/clean pairs for all 20 training styles plus the held-out
validation style. To reach roughly 10 000 pairs (recommended for the cleaner),
set `--train-per-style 500`:

```bash
python generate_dataset.py --out data --train-per-style 500 --val-total 1000
```

The script writes aligned images under:

- `data/train/noisy/<style>/LABEL__style_XXXX.png`
- `data/train/clean/<style>/LABEL__style_XXXX.png`
- `data/val/noisy/...`
- `data/val/clean/...`

The clean target is always rendered with the same sans-serif font so the U-Net
has a stable goal.

## 3. Build Manifests

Create TSVs describing both training pairs and OCR-friendly targets:

```bash
python build_annotations.py --data-root data --out-dir annotations
```

You will obtain:

- `annotations/cleaner_train.tsv` – `noisy_path<TAB>clean_path<TAB>label`
- `annotations/cleaner_val.tsv`
- `annotations/train.tsv` / `annotations/val.tsv` – clean images + labels for
	later TrOCR fine-tuning

## 4. Train the Cleaner

Launch the default training run (mixed precision on CUDA, early stopping, L1
reconstruction loss):

```bash
python train_cleaner.py \
	--train-manifest annotations/cleaner_train.tsv \
	--val-manifest annotations/cleaner_val.tsv \
	--image-root data \
	--output-dir outputs/cleaner \
	--epochs 60 \
	--batch-size 16
```

During training the script reports training loss, validation loss, and PSNR. It
keeps the best checkpoint (`outputs/cleaner/best_cleaner.pt`) and records the
full history in `outputs/cleaner/history.json`. Early stopping kicks in after 10
epochs without improvement—no manual intervention needed.

### Cleaner Architecture Highlights

- Moderate-depth U-Net with dropout and batch norm to fight overfitting
- On-the-fly augmentations (blur, contrast, Gaussian noise, JPEG artifacts) on
	the noisy input only
- AdamW + ReduceLROnPlateau scheduler; gradient clipping at 1.0

## 5. Inspect Outputs

- Load the checkpoint in a notebook and visualise predictions on
	`data/val/noisy`. Expect crisp Arial-like text.
- Check `history.json` to ensure validation loss tracks training loss closely;
	divergence flags underfitting or data leakage.

## 6. Next Steps (After Cleaner Converges)

1. Generate cleaned datasets by running the trained U-Net across raw captchas.
2. Fine-tune TrOCR on those cleaned images using the existing
	 `trocr_train.py`/`trocr_eval.py` utilities.
3. Export the end-to-end pipeline (`cleaner -> TrOCR`) for inference.

## Troubleshooting

- **Cleaner overfits quickly** – Increase `--train-per-style`, keep augmentations
	enabled (they are baked into the dataset class), and monitor PSNR.
- **Validation loss stalls** – Ensure `generate_dataset.py` ran with the larger
	sample count; the cleaner needs diversity to generalise.
- **Artifacts misaligned** – Re-run dataset generation so noisy/clean pairs are
	re-created together. Each sample uses a shared filename to guarantee pairing.
