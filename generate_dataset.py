"""Prepare the synthetic CAPTCHA dataset split for training and validation.

The spec requires 2,500 total images:
* Training: 2,000 images, balanced across styles 1-20 (100 samples each).
* Validation: 500 images, all "Mutant Hybrid" (style 21) to measure generalisation.

This module only wires the logic. Run the script manually once satisfied with the
visual output of each style. Pass `--dry-run` to preview the plan without writing
any files.
"""

import argparse
import os
from typing import Dict

from captcha_solver.generate_synthetic import gen_captcha, CaptchaStyle


def _ensure_split_dirs(out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    return {'train': train_dir, 'val': val_dir}


def _style_subdir(root: str, style: CaptchaStyle) -> str:
    style_dir = os.path.join(root, style.name.lower())
    os.makedirs(style_dir, exist_ok=True)
    return style_dir


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

    print("\nGenerating training split...")
    for style in train_styles:
        style_dir = _style_subdir(dirs['train'], style)
        for idx in range(train_per_style):
            img, label = gen_captcha(style=style)
            safe_label = _sanitize_label(label)
            fname = f"{safe_label}__{style.name.lower()}_{idx:04d}.png"
            img.save(os.path.join(style_dir, fname))

    print("Generating validation split (Mutant Hybrid)...")
    val_dir = _style_subdir(dirs['val'], val_style)
    for idx in range(val_total):
        img, label = gen_captcha(style=val_style)
        safe_label = _sanitize_label(label)
        fname = f"{safe_label}__{val_style.name.lower()}_{idx:04d}.png"
        img.save(os.path.join(val_dir, fname))

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
