# PaddleOCR Finetuning Notes

This folder contains the pieces that complement the main PaddleOCR repository:

- `charset.txt` – character dictionary (digits + uppercase letters) referenced by the recognition head.
- `labels/` – generated manifest files (`train_label.txt`, `val_label.txt`) created by `prepare_paddle_labels.py`.

Usage overview:

1. Clone https://github.com/PaddlePaddle/PaddleOCR and install its requirements.
2. Run the local dataset generator and label preparation script:

   ```bash
   python generate_dataset.py --out data
   python prepare_paddle_labels.py --train-dir data/train --val-dir data/val --output-dir paddle_ocr/labels
   ```

3. Kick off training from the PaddleOCR repo, overriding paths to the generated
   label manifests and charset as described in the root `README.md`.
