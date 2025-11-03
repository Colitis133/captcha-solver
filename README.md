# CAPTCHA Solver (ddddocr-focused)

This workspace ships the pieces you need to:

- generate synthetic CAPTCHA data,
- evaluate the stock **ddddocr** model with aggressive preprocessing, and
- fine-tune our Torch CRNN head (initialized to mimic ddddocr) while freezing
	the backbone so it learns your character distribution without catastrophic
	forgetting.

Legacy multi-model experiments have been removed; the repo is now focused on
one objective: make ddddocr read your CAPTCHA images reliably.

## 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
The OpenCV dependency in `ddd_advanced_preprocess_test.py` requires system X11
libs on Linux. Install them once if you have not already:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
```
## 2. Generate/Refresh the Synthetic Dataset

```bash
python generate_dataset.py --out data \
	--train-per-style 100 \
	--val-total 500
```
This produces `data/train/<style_name>/*.png` and
`data/val/<style_name>/*.png`. Filenames already embed the ground-truth code
(e.g. `A1WV__mixed_fonts_mayhem_0000.png`).

## 3. Build Annotation Manifests

ddddocr expects a simple `image_path<TAB>label` text file. Generate them from
the dataset structure:
```bash
python build_annotations.py --data-root data --out-dir annotations
```
You will get:

- `annotations/train.tsv`
- `annotations/val.tsv`

The preprocessing harness and training script both read these manifests by
default.

## 4. Baseline ddddocr Sweep (Optional)
Before fine-tuning, inspect how the stock model behaves with the included
preprocessing ablations:

```bash
python ddd_advanced_preprocess_test.py \
	--label-file annotations/train.tsv \
	--target-mode suffix \
	--suffix-length 4
```

Notes:
- `--filter mixed_fonts_mayhem` targets a specific style.
- `--target-mode suffix` scores only the trailing CAPTCHA code (`7ZFC` style)
	if you do not care about the prefix baked into the filename.

The script summarises per-pipeline hits, combined accuracy, and the closest
misses so you can see whether finetuning is still required.

## 5. Fine-tune the ddddocr Head
We expose a Torch implementation of the ddddocr recognition stack (`CRNN_AFFN`) and
train it with CTC. A few defaults changed to match the workflow:

- 12 training epochs with early stopping enabled (`patience=5`).
- ReduceLROnPlateau scheduler active by default.
- Backbone-freezing knobs to keep the CNN features stable while adapting the
	sequence head.

Recommended run (freeze CNN + AFFN + projection, train only BiLSTM + classifier):

```bash
python -m captcha_solver.train \
	--train-dir data/train \
	--val-dir data/val \
	--batch-size 32 \
	--lr 3e-4 \
	--freeze-backbone \
	--curriculum-epochs 4 \
	--patience 4
```

Key CLI options:
- `--freeze-backbone` freezes `stem`, `encoder`, `affn`, and projection layers.
- `--freeze` accepts an explicit subset of modules (`stem`, `encoder`, `affn`,
  `proj`, `rnn`) if you prefer fine-grained control. Combine with
  `--freeze-backbone` to lock everything except the classifier.
- `--early-stop/--no-early-stop` toggles validation-based early stopping.
- `--curriculum-epochs` keeps training on “easy” styles for the first *N*
	epochs (default 6). Adjust if your custom dataset is already balanced.

### Freezing Specific Model Parts

Use `--freeze` to lock individual modules while leaving the rest trainable. For
fine-grained control:

```bash
# Freeze only the convolutional stem and encoder
python -m captcha_solver.train --freeze stem encoder

# Freeze everything except the classifier
python -m captcha_solver.train --freeze-backbone --freeze rnn

# Train the full network (omit freeze flags entirely)
python -m captcha_solver.train
```

`--freeze-backbone` is shorthand for freezing `stem`, `encoder`, `affn`, and
`proj`. Pair it with `--freeze rnn` when you need to lock the entire network,
then rerun without the flags to fine-tune additional layers.

The script writes checkpoints to `checkpoints/best.pt` (Torch state dict) and
optionally exports a TorchScript model when `--export` is provided.

## 6. Evaluate Fine-tuned Weights
After training finishes (or early-stops), export the TorchScript module and
measure exact-match accuracy:

```bash
python -m captcha_solver.train --export captcha_model.pt --epochs 0 --no-scheduler

# Evaluate on the validation set
python eval_val.py --model captcha_model.pt --val-dir data/val --use-cuda
```

For suffix-only scoring, reuse `ddd_advanced_preprocess_test.py` with the new
manifests to see how many CAPTCHAs now decode the code correctly.

## 7. What to Tweak Next

- **Generate more synthetic data** once training converges (5k–50k samples helps).
- **Augment harder styles** inside `generate_synthetic.py` if certain styles
	still fail after finetuning.
- **Progressive unfreezing**: start with `--freeze-backbone`, then rerun without
	freezing to fine-tune the entire network for a few extra epochs at a lower LR
	(e.g. `1e-4`). Update the optimizer by restarting training or modify the
	script if you need on-the-fly unfreezing.

- **Inference integration**: export TorchScript and embed it in your service; a
	conversion to ONNX is straightforward using `torch.onnx.export` if you need
	to replace the stock `common_old.onnx`.

## FAQ

**Q: Why is there still a curriculum warm-up?**  
Because ddddocr’s base model already performs reasonably on easy fonts; the
warm-up keeps that behaviour while the BiLSTM learns your tail styles before we
mix in the gnarlier ones.

**Q: How many epochs do I need?**  
In practice, 8–12 epochs with early stopping (`patience` 4–5) is sufficient.
If the validation loss plateaus, the scheduler halves the LR automatically.

**Q: Do I need the prefixes in filenames?**  
No. The manifests only use the portion before the first underscore. If you only
care about the four-character CAPTCHA code, run evaluations in suffix mode or
strip prefixes during dataset generation.
