"""Prepare paired noisy/clean CAPTCHA datasets for the cleaner + OCR pipeline.

The generator now writes **aligned pairs** for every sample:

* train/noisy/<style>/<label>__...png – heavy augmentations (20 styles).
* train/clean/<style>/<label>__...png – canonical single-font rendering.
* val/noisy and val/clean – Mutant Hybrid only for held-out validation.

This paired layout feeds the U-Net cleaner first; the cleaned images later drive
TrOCR fine-tuning. Pass `--dry-run` to preview counts without writing files.
"""

import argparse
import os
from typing import Dict

from captcha_solver.generate_synthetic import gen_captcha, CaptchaStyle, render_clean_captcha


def _ensure_split_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split in ("train", "val"):
        split_root = os.path.join(out_dir, split)
        for kind in ("noisy", "clean"):
            os.makedirs(os.path.join(split_root, kind), exist_ok=True)
        paths[split] = split_root
    return paths


def _style_subdirs(split_root: str, style: CaptchaStyle) -> Dict[str, str]:
    style_name = style.name.lower()
    noisy_dir = os.path.join(split_root, "noisy", style_name)
    clean_dir = os.path.join(split_root, "clean", style_name)
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    return {"noisy": noisy_dir, "clean": clean_dir}


def _sanitize_label(label: str) -> str:
    return "".join(c for c in label if c.isalnum())


def generate(out_dir: str = 'data', train_per_style: int = 100, val_total: int = 500, dry_run: bool = False) -> Dict[str, int]:
    """Generate the dataset split according to the project spec.

    Args:
        out_dir: Base directory where `train/` and `val/` will be created.
        train_per_style: Number of images per training style (styles 1-20).
        val_total: Total validation images, using style 21 exclusively.
        dry_run: When True, only prints the planned counts without writing files.

    Returns:
        Dict summarising how many images were scheduled or written for each split.
    """

    train_styles = [s for s in CaptchaStyle if s != CaptchaStyle.MUTANT_HYBRID]
    val_style = CaptchaStyle.MUTANT_HYBRID

    planned = {
        'train': train_per_style * len(train_styles),
        'val': val_total,
        'total': train_per_style * len(train_styles) + val_total,
    }

    print("Dataset plan:")
    print(f"  Training: {planned['train']} images ({train_per_style} each across {len(train_styles)} styles)")
    print(f"  Validation: {planned['val']} images (style {val_style.name})")

    if dry_run:
        print("Dry-run enabled, no files will be generated.")
        return planned

    dirs = _ensure_split_dirs(out_dir)

    print("\nGenerating training split (paired noisy ↔ clean images)...")
    for style in train_styles:
        style_dirs = _style_subdirs(dirs['train'], style)
        for idx in range(train_per_style):
            img, label = gen_captcha(style=style)
            safe_label = _sanitize_label(label)
            width, height = img.size
            clean_img = render_clean_captcha(label, width=width, height=height)
            fname = f"{safe_label}__{style.name.lower()}_{idx:04d}.png"
            img.save(os.path.join(style_dirs['noisy'], fname))
            clean_img.save(os.path.join(style_dirs['clean'], fname))

    print("Generating validation split (Mutant Hybrid)...")
    val_dirs = _style_subdirs(dirs['val'], val_style)
    for idx in range(val_total):
        img, label = gen_captcha(style=val_style)
        safe_label = _sanitize_label(label)
        width, height = img.size
        clean_img = render_clean_captcha(label, width=width, height=height)
        fname = f"{safe_label}__{val_style.name.lower()}_{idx:04d}.png"
        img.save(os.path.join(val_dirs['noisy'], fname))
        clean_img.save(os.path.join(val_dirs['clean'], fname))

    print("\nCompleted dataset generation.")
    print(f"  Training images: {planned['train']}")
    print(f"  Validation images: {planned['val']}")

    return planned


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the synthetic CAPTCHA dataset split.")
    parser.add_argument('--out', type=str, default='data', help='Output directory for the dataset root.')
    parser.add_argument('--train-per-style', type=int, default=100, help='Number of training images per style (styles 1-20).')
    parser.add_argument('--val-total', type=int, default=500, help='Total validation images (style 21).')
    parser.add_argument('--dry-run', action='store_true', help='Only print the plan without writing any files.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(out_dir=args.out, train_per_style=args.train_per_style, val_total=args.val_total, dry_run=args.dry_run)
