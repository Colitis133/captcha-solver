#!/usr/bin/env python3
"""Build paired input/target splits from the real CAPTCHA dataset.

This script copies a subset of the Kaggle "Huge Captcha Dataset" images into a
paired structure and renders clean targets using the project font renderer. The
outputs are intended for training the cleaner U-Net and downstream OCR models.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from random import Random
from typing import Iterable, List, Sequence, Tuple

from PIL import Image

from captcha_solver.text_render import render_clean_label

SUPPORTED_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data"),
        help="Directory containing the original splits (train/, val/, test/).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/paired"),
        help="Output directory for the paired dataset structure.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("annotations"),
        help="Directory where TSV manifests will be written.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=50_000,
        help="Maximum number of training samples to keep.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=10_000,
        help="Maximum number of validation samples to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used when shuffling the source images.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Skip shuffling and keep the natural ordering of files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing output directories instead of failing.",
    )
    return parser.parse_args()


def collect_images(split_dir: Path) -> List[Path]:
    images: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(split_dir.glob(f"*{ext}"))
    images = [p for p in images if p.is_file()]
    if not images:
        raise FileNotFoundError(f"No images found in {split_dir}")
    return images


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def render_pair(
    source_path: Path,
    input_dest: Path,
    target_dest: Path,
    label: str,
) -> None:
    shutil.copy2(source_path, input_dest)
    with Image.open(source_path) as img:
        img = img.convert("RGB")
        width, height = img.size
    clean_image = render_clean_label(label, width, height)
    clean_image.save(target_dest, format="PNG")


def relative_manifest_entry(
    input_path: Path,
    target_path: Path,
    label: str,
    root: Path,
) -> Tuple[str, str, str]:
    input_rel = input_path.resolve().relative_to(root)
    target_rel = target_path.resolve().relative_to(root)
    return input_rel.as_posix(), target_rel.as_posix(), label


def write_manifest(entries: Iterable[Tuple[str, str, str]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for noisy_rel, clean_rel, label in entries:
            handle.write(f"{noisy_rel}\t{clean_rel}\t{label}\n")
            count += 1
    return count


def select_subset(images: Sequence[Path], limit: int, seed: int | None, shuffle: bool) -> List[Path]:
    chosen = list(images)
    if shuffle:
        rng = Random(seed)
        rng.shuffle(chosen)
    if limit > 0:
        chosen = chosen[:limit]
    return chosen


def process_split(
    split: str,
    limit: int,
    source_root: Path,
    out_root: Path,
    manifest_root: Path,
    overwrite: bool,
    seed: int,
    shuffle: bool,
) -> int:
    if limit <= 0:
        return 0

    source_dir = source_root / split
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {source_dir}")

    images = collect_images(source_dir)
    selected = select_subset(images, limit, seed, shuffle)
    if len(selected) < limit:
        print(f"Warning: requested {limit} {split} samples but only found {len(selected)}. Using all available.")

    split_out_root = out_root / split
    inputs_dir = split_out_root / "inputs"
    targets_dir = split_out_root / "targets"
    prepare_directory(split_out_root, overwrite)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Tuple[str, str, str]] = []
    out_root_resolved = out_root.resolve()

    for idx, source_path in enumerate(selected, start=1):
        label = source_path.stem
        filename = source_path.name
        input_dest = inputs_dir / filename
        target_dest = targets_dir / filename
        render_pair(source_path, input_dest, target_dest, label)
        manifest_entries.append(relative_manifest_entry(input_dest, target_dest, label, out_root_resolved))
        if idx % 1000 == 0:
            print(f"[{split}] processed {idx} / {len(selected)}")

    manifest_path = manifest_root / f"cleaner_{split}.tsv"
    count = write_manifest(manifest_entries, manifest_path)
    print(f"[{split}] wrote manifest with {count} entries -> {manifest_path}")
    return count


def main() -> None:
    args = parse_args()

    out_root = args.out_root.resolve()
    manifest_root = args.manifest_dir.resolve()

    if out_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output root {out_root} already exists. Pass --overwrite to rebuild the paired dataset."
        )

    out_root.mkdir(parents=True, exist_ok=True)

    total_train = process_split(
        "train",
        args.train_count,
        args.source_root,
        out_root,
        manifest_root,
        args.overwrite,
        args.seed,
        not args.no_shuffle,
    )
    total_val = process_split(
        "val",
        args.val_count,
        args.source_root,
        out_root,
        manifest_root,
        args.overwrite,
        args.seed + 1,
        not args.no_shuffle,
    )

    print(f"Done. Train samples: {total_train}, Val samples: {total_val}")


if __name__ == "__main__":
    main()
