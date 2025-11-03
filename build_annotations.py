#!/usr/bin/env python3
"""Generate tab-separated annotation files from the dataset folders."""

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple


FILENAME_LABEL_RE = re.compile(r"([A-Za-z0-9]+)_.*\.(?:png|jpg|jpeg)$")


def collect_images(split_dir: Path) -> Iterable[Tuple[str, str]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for path in sorted(split_dir.rglob("*.png")):
        match = FILENAME_LABEL_RE.search(path.name)
        if not match:
            continue
        label = match.group(1)
        yield str(path.as_posix()), label


def write_manifest(entries: Iterable[Tuple[str, str]], root: Path, outfile: Path) -> int:
    root = root.resolve()
    count = 0
    with outfile.open("w", encoding="utf-8") as f:
        for img_path, label in entries:
            rel = Path(img_path).resolve().relative_to(root)
            f.write(f"{rel.as_posix()}\t{label}\n")
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

    train_entries = list(collect_images(train_dir))
    val_entries = list(collect_images(val_dir))

    train_count = write_manifest(train_entries, args.data_root, args.out_dir / "train.tsv")
    val_count = write_manifest(val_entries, args.data_root, args.out_dir / "val.tsv")

    print(f"Wrote {train_count} training entries -> {args.out_dir / 'train.tsv'}")
    print(f"Wrote {val_count} validation entries -> {args.out_dir / 'val.tsv'}")


if __name__ == "__main__":
    main()
