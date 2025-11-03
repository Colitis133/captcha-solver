# CAPTCHA Solver (TrOCR Fine-tuning)

This repository turns Microsoft’s **TrOCR** models into a plug-and-play
fine-tuning kit for CAPTCHA OCR. It ships with:

- a synthetic CAPTCHA generator (`captcha_solver.generate_synthetic`) able to
	produce 20+ distortion styles;
- utilities to materialise train/validation splits and manifest files;
- training and evaluation front-ends built on Hugging Face Transformers
	(`trocr_train.py` / `trocr_eval.py`).

Everything revolves around TSV manifests of the form
`relative/image/path.png<TAB>GROUNDTRUTH`. Once you generate the dataset and
manifests, you can fine-tune any TrOCR checkpoint with a single command.

## 1. Prerequisites

- Python 3.9 or newer with `pip` on your PATH
- CUDA-capable GPU recommended (CPU works but is slow)
- Adequate disk space (a default dataset run creates ~2 500 PNGs)

> 💡 Prefer isolated environments? You can still run `./setup_env.sh`, which
> creates a `.venv` automatically. The rest of this guide assumes you install
> packages directly into your active Python environment.

## 2. Install Dependencies

From the repository root, install everything required by the TrOCR pipeline:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file lists PyTorch, Transformers, Datasets, Accelerate,
Evaluate, Pillow, OpenCV, and a few quality-of-life utilities.

## 3. Generate Synthetic CAPTCHA Data

The helper script produces both train and validation splits. The defaults match
the project spec (2 000 training images evenly distributed across 20 styles plus
500 validation images using the “Mutant Hybrid” style):

```bash
python generate_dataset.py --out data \
	--train-per-style 100 \
	--val-total 500
```

Each PNG filename embeds the label, e.g. `A1WV__mixed_fonts_mayhem_0000.png`, so
you never need a separate transcription file when inspecting samples.

## 4. Build Training/Evaluation Manifests

Convert the image folders into TSV manifests once the dataset is ready:

```bash
python build_annotations.py --data-root data --out-dir annotations
```

This writes two files:

- `annotations/train.tsv`
- `annotations/val.tsv`

Both contain relative paths (from `data/`) and become the sole inputs to the
training/evaluation scripts.

## 5. Fine-tune TrOCR

`trocr_train.py` wraps Hugging Face’s `Seq2SeqTrainer` around
`VisionEncoderDecoderModel`. A typical run looks like:

```bash
python trocr_train.py \
	--train-manifest annotations/train.tsv \
	--val-manifest annotations/val.tsv \
	--pretrained microsoft/trocr-base-printed \
	--output-dir outputs/trocr \
	--epochs 8 \
	--batch-size 8 \
	--eval-strategy epoch \
	--save-strategy epoch
```

Useful knobs:

- `--lr`, `--weight-decay`, `--warmup-ratio` – schedule and regularisation
- `--gradient-accumulation` – mimic larger batches when GPU memory is tight
- `--max-target-length` – maximum decoded token length (label length + EOS)
- `--fp16` / `--bf16` – enable mixed precision when supported by your hardware
- `--resume-from` – resume from a previously saved Trainer checkpoint
- `--freeze-encoder` / `--freeze-decoder` – lock one side of the
	VisionEncoderDecoderModel if you only want to adapt the other

During training, the script measures character-error rate (CER) and exact match
on the validation manifest, keeps the best-performing checkpoint (lowest CER),
and finally saves both the model and processor into `--output-dir`.

## 6. Evaluate a Fine-tuned Checkpoint

Run beam-search decoding and compute CER/exact-match on any manifest:

```bash
python trocr_eval.py \
	--model-path outputs/trocr \
	--manifest annotations/val.tsv \
	--num-beams 5
```

Key options:

- `--limit N` – evaluate only the first *N* rows (quick smoke tests)
- `--report-mismatches K` – number of errors to print for manual inspection
- `--device` – override the device string (e.g. `cuda:1`, `cpu`)

## 7. Exporting & Using the Model Elsewhere

The saved directory in `outputs/trocr` (or your chosen path) is fully compatible
with Hugging Face Transformers. Example inference snippet:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("outputs/trocr")
model = VisionEncoderDecoderModel.from_pretrained("outputs/trocr").eval()

image = Image.open("sample.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
pred_ids = model.generate(**inputs, num_beams=5, max_length=processor.tokenizer.model_max_length)
print(processor.batch_decode(pred_ids, skip_special_tokens=True)[0])
```

## 8. Troubleshooting & Tips

- **Dataset scale** – increase `--train-per-style` / `--val-total` for harder
	captchas or to reduce overfitting.
- **Style curriculum** – regenerate manifests that emphasise difficult styles to
	focus fine-tuning on problematic distributions.
- **Beam search tuning** – play with `--num-beams`, `--max-target-length`, or
	add constraints (length penalty, forced tokens) by editing `trocr_eval.py`.
- **GPU memory** – combine `--gradient-accumulation`, reduced batch size, and
	mixed precision (`--fp16`/`--bf16`) to stay within your VRAM budget.

## 9. FAQ

**Do I need a GPU?**  
A single 16 GB GPU comfortably runs the base TrOCR model with batch size 8. CPU
training works but is orders of magnitude slower.

**Can I fine-tune a different TrOCR checkpoint?**  
Yes—swap `--pretrained` for any compatible identifier, e.g.
`microsoft/trocr-small-printed` or a custom path containing a previously saved
model.

**Why does `setup_env.sh` create a virtual environment if it’s optional?**  
The script is meant for reproducible automation (CI, fresh machines). Running
`pip install -r requirements.txt` directly is perfectly valid if you manage your
own environment.

**How do I decode captchas in production?**  
Load the saved directory with `VisionEncoderDecoderModel` and
`TrOCRProcessor`, feed batches through `model.generate`, and post-process the
strings as needed.
