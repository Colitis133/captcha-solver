"""Generate PaddleOCR label files from CAPTCHA style directories.

This script scans a directory tree such as `data/train/<style>/*.png` and
emits PaddleOCR-compatible label manifests where each line is formatted as:

    relative/path/to/image.png\tLABEL

By default, it produces `train_label.txt` and `val_label.txt` under
`paddle_ocr/labels/`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

LABEL_DELIMITER = "\t"


def _extract_label(filename: str) -> str:
    """Derive the ground-truth label from the captcha filename.

    Filenames are expected in the format `<label>__style_xxxx.png` as produced
    by `generate_dataset.py`. Non-alphanumeric characters in the label portion
    are stripped out as a safeguard.
    """

    stem = Path(filename).stem
    if "__" in stem:
        raw_label = stem.split("__", 1)[0]
    else:
        raw_label = stem
    return "".join(ch for ch in raw_label if ch.isalnum())


def _gather_samples(root: Path) -> Iterable[Tuple[Path, str]]:
    """Yield `(absolute_path, label)` pairs for every image under `root`."""

    if not root.exists():
        return []

    samples = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        label = _extract_label(path.name)
        if not label:
            continue
        samples.append((path.resolve(), label))
    return samples


def _write_manifest(samples: Iterable[Tuple[Path, str]], output_file: Path, base_dir: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        for abs_path, label in samples:
            rel_path = os.path.relpath(abs_path, base_dir)
            fh.write(f"{rel_path.replace(os.sep, '/')}" f"{LABEL_DELIMITER}{label}\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PaddleOCR label manifests from a CAPTCHA dataset")
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"), help="Directory containing training images")
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"), help="Directory containing validation images")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base path to make image paths relative to")
    parser.add_argument("--output-dir", type=Path, default=Path("paddle_ocr/labels"), help="Where to write the manifest files")
    parser.add_argument("--train-filename", default="train_label.txt", help="Filename for the training manifest")
    parser.add_argument("--val-filename", default="val_label.txt", help="Filename for the validation manifest")
    args = parser.parse_args()

    train_samples = _gather_samples(args.train_dir.resolve())
    val_samples = _gather_samples(args.val_dir.resolve())

    if not train_samples:
        print(f"Warning: no train samples found under {args.train_dir}")
    if not val_samples:
        print(f"Warning: no validation samples found under {args.val_dir}")

    output_dir = args.output_dir
    base_dir = args.base_dir.resolve()

    _write_manifest(train_samples, output_dir / args.train_filename, base_dir)
    _write_manifest(val_samples, output_dir / args.val_filename, base_dir)

    print("Wrote PaddleOCR manifests:")
    print(f"  Train: {output_dir / args.train_filename} ({len(train_samples)} samples)")
    print(f"  Val:   {output_dir / args.val_filename} ({len(val_samples)} samples)")


if __name__ == "__main__":
    main()
