# CAPTCHA Solver (TrOCR Fine-tuning)

This workspace contains utilities to generate synthetic CAPTCHA data and
fine-tune Microsoft’s **TrOCR** models so they adapt to your character
distribution. The workflow is built around Hugging Face Transformers and
expects two TSV manifests that map image paths to ground-truth text.

## 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
The TrOCR pipeline relies on PyTorch, Transformers, Datasets, Accelerate, and
Evaluate. `requirements.txt` pins the core libraries used in this repo.

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

TrOCR training uses simple `image_path<TAB>label` manifests. Generate them from
the dataset structure:
```bash
python build_annotations.py --data-root data --out-dir annotations
```
You will get:

- `annotations/train.tsv`
- `annotations/val.tsv`

The training and evaluation scripts consume the manifests directly.

## 4. Fine-tune TrOCR

`trocr_train.py` wraps Hugging Face’s `Seq2SeqTrainer` around
`VisionEncoderDecoderModel`. A default run (base model, beam search = 4) looks
like this:

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

Useful flags:
- `--lr`, `--weight-decay`, `--warmup-ratio`: learning schedule.
- `--gradient-accumulation`: increase effective batch size when GPU RAM is
  limited.
- `--max-target-length`: maximum token length for the CAPTCHA text.
- `--fp16/--bf16`: enable mixed precision when the hardware supports it.
- `--resume-from`: resume training from a saved Trainer checkpoint.

The script automatically saves the best performing checkpoint (by CER) along
with the processor in `outputs/trocr` (or your chosen `--output-dir`).

## 5. Evaluate a Fine-tuned Model

Run beam-search decoding against a manifest and report character error rate and
exact-match accuracy:

```bash
python trocr_eval.py \
	--model-path outputs/trocr \
	--manifest annotations/val.tsv \
	--num-beams 5
```

The evaluator prints sample mismatches and overall metrics. Use `--limit` to
spot-check a subset or `--report-mismatches` to increase the number of logged
errors.

## 6. Tips for Better Accuracy

- **Scale the dataset**: larger synthetic corpora generally yield better
  generalisation. Increase `--train-per-style` and regenerate manifests.
- **Curriculum / filtering**: create manifests that emphasise difficult styles
  or generate additional variants for misclassified examples.
- **Beam search tuning**: adjust `--num-beams`, `--max-target-length`, or apply
  constraints such as `--length-penalty` within `trocr_eval.py`.
- **Mixed precision**: on modern GPUs, `--fp16` or `--bf16` provides a sizeable
  throughput boost when fine-tuning.

## 7. FAQ

**Q: Do I need a GPU?**  
TrOCR fine-tuning is GPU-friendly (the base model fits on a single 16 GB card
with batch size 8). CPU-only runs are possible but much slower.

**Q: Can I change the pretrained checkpoint?**  
Yes. Swap `--pretrained` for any `VisionEncoderDecoderModel` compatible TrOCR
variant, e.g. `microsoft/trocr-small-printed` or `microsoft/trocr-base-handwritten`.

**Q: How do I run inference in production?**  
Load the saved model directory with `VisionEncoderDecoderModel` and
`TrOCRProcessor`. Use `model.generate(processor(images=..., return_tensors="pt"))` to decode.
