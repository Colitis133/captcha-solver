#!/usr/bin/env python3
"""Generate TSV manifests for the cleaner and OCR stages."""

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple, List


FILENAME_LABEL_RE = re.compile(r"([A-Za-z0-9]+)_.*\.(?:png|jpg|jpeg)$")


def collect_images(split_dir: Path) -> Iterable[Tuple[str, str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    for path in sorted(split_dir.rglob("*.png")):
        match = FILENAME_LABEL_RE.search(path.name)
        if not match:
            continue
        label = match.group(1)
        yield str(path.as_posix()), label


def collect_pairs(split_dir: Path) -> List[Tuple[str, str, str]]:
    noisy_root = split_dir / "noisy"
    clean_root = split_dir / "clean"
    if not noisy_root.exists() or not clean_root.exists():
        raise FileNotFoundError(f"Expected noisy/clean subdirectories under {split_dir}")

    pairs: List[Tuple[str, str, str]] = []
    for clean_path in sorted(clean_root.rglob("*.png")):
        rel = clean_path.relative_to(clean_root)
        noisy_path = noisy_root / rel
        if not noisy_path.exists():
            raise FileNotFoundError(f"Missing noisy counterpart for {clean_path}")

        match = FILENAME_LABEL_RE.search(clean_path.name)
        if not match:
            continue
        label = match.group(1)
        pairs.append((str(noisy_path.as_posix()), str(clean_path.as_posix()), label))

    return pairs


def write_manifest(entries: Iterable[Tuple[str, str]], root: Path, outfile: Path) -> int:
    root = root.resolve()
    count = 0
    with outfile.open("w", encoding="utf-8") as f:
        for img_path, label in entries:
            rel = Path(img_path).resolve().relative_to(root)
            f.write(f"{rel.as_posix()}\t{label}\n")
            count += 1
    return count


def write_pair_manifest(entries: Iterable[Tuple[str, str, str]], root: Path, outfile: Path) -> int:
    root = root.resolve()
    count = 0
    with outfile.open("w", encoding="utf-8") as f:
        for noisy_path, clean_path, label in entries:
            noisy_rel = Path(noisy_path).resolve().relative_to(root)
            clean_rel = Path(clean_path).resolve().relative_to(root)
            f.write(f"{noisy_rel.as_posix()}\t{clean_rel.as_posix()}\t{label}\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root containing train/ and val/ folders.")
    parser.add_argument("--out-dir", type=Path, default=Path("annotations"), help="Output directory for TSV manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = collect_pairs(train_dir)
    val_pairs = collect_pairs(val_dir)

    train_clean_entries = [(clean_path, label) for _, clean_path, label in train_pairs]
    val_clean_entries = [(clean_path, label) for _, clean_path, label in val_pairs]

    train_count = write_manifest(train_clean_entries, args.data_root, args.out_dir / "train.tsv")
    val_count = write_manifest(val_clean_entries, args.data_root, args.out_dir / "val.tsv")

    pair_train_count = write_pair_manifest(train_pairs, args.data_root, args.out_dir / "cleaner_train.tsv")
    pair_val_count = write_pair_manifest(val_pairs, args.data_root, args.out_dir / "cleaner_val.tsv")

    print(f"Wrote {train_count} training entries -> {args.out_dir / 'train.tsv'}")
    print(f"Wrote {val_count} validation entries -> {args.out_dir / 'val.tsv'}")
    print(f"Wrote {pair_train_count} cleaner training pairs -> {args.out_dir / 'cleaner_train.tsv'}")
    print(f"Wrote {pair_val_count} cleaner validation pairs -> {args.out_dir / 'cleaner_val.tsv'}")


if __name__ == "__main__":
    main()
