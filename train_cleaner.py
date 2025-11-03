#!/usr/bin/env python3
"""Train the U-Net captcha cleaner on paired noisy/clean images."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from captcha_solver.cleaner_dataset import CleanerDataset
from captcha_solver.cleaner_model import CleanerUNet


@dataclass
class Metrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_psnr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("annotations/cleaner_train.tsv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("annotations/cleaner_val.tsv"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cleaner"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10, help="Stop if val loss does not improve for this many epochs.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(mse: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CleanerDataset(args.train_manifest, args.image_root, augment=True)
    val_ds = CleanerDataset(args.val_manifest, args.image_root, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = CleanerUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val = float("inf")
    best_epoch = 0
    history: List[Metrics] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for noisy, clean, _ in progress:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(noisy)
                loss = criterion(pred, clean)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * noisy.size(0)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for noisy, clean, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                    mse = torch.mean((pred - clean) ** 2, dim=(1, 2, 3))
                val_loss += loss.item() * noisy.size(0)
                val_psnr += psnr(mse).sum().item()

        val_loss /= len(val_ds)
        val_psnr /= len(val_ds)
        scheduler.step(val_loss)

        history.append(Metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_psnr=val_psnr))

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            }, args.output_dir / "best_cleaner.pt")

        # Persist training history incrementally
        metrics_path = args.output_dir / "history.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in history], f, indent=2)

        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val PSNR {val_psnr:.2f} dB"
            + (" <-- best" if improved else "")
        )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping triggered (no improvement for {args.patience} epochs).")
            break

    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "image_root": str(args.image_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "seed": args.seed,
            "amp": amp_enabled,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }, f, indent=2)

    print(f"Training complete. Best epoch: {best_epoch}, best val loss: {best_val:.4f}")
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
