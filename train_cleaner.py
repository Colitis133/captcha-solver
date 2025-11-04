#!/usr/bin/env python3
"""Train the U-Net captcha cleaner on paired noisy/clean images."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torchvision import models

from captcha_solver.cleaner_dataset import CleanerDataset
from captcha_solver.cleaner_model import CleanerUNet


WINDOW_CACHE: Dict[tuple[int, float, int, str, torch.dtype], torch.Tensor] = {}


def _get_gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (window_size, sigma, channels, str(device), dtype)
    window = WINDOW_CACHE.get(key)
    if window is None:
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = torch.outer(gauss, gauss)
        kernel_2d = kernel_2d / kernel_2d.sum()
        window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
        WINDOW_CACHE[key] = window
    return window


def structural_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("SSIM expects NCHW tensors")
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape for SSIM computation")

    _, channels, height, width = x.shape

    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window -= 1
    effective_window = max(effective_window, 1)

    padding = effective_window // 2
    window = _get_gaussian_window(
        effective_window,
        sigma,
        channels,
        x.device,
        x.dtype,
    )

    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + eps)

    return torch.clamp(ssim_map.mean(dim=(1, 2, 3)), min=-1.0, max=1.0)


@dataclass
class Metrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_psnr: float
    val_ssim: float | None = None


class VGGPerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using ImageNet feature activations."""

    def __init__(self, layers: Sequence[int] | None = None, resize_to: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_FEATURES
        vgg = models.vgg16(weights=weights).features

        if layers is None:
            # Default to relu1_2, relu2_2, relu3_3 blocks.
            layers = (4, 9, 16)

        slices: List[nn.Sequential] = []
        start = 0
        for idx in layers:
            block = nn.Sequential(*[vgg[i] for i in range(start, idx)])
            slices.append(block)
            start = idx

        self.blocks = nn.ModuleList(slices)
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.resize_to = resize_to

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = (x.float() - self.mean) / self.std
        y = (y.float() - self.mean) / self.std

        if self.resize_to is not None:
            x = F.interpolate(x, size=self.resize_to, mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=self.resize_to, mode="bilinear", align_corners=False)

        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("annotations/cleaner_train.tsv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("annotations/cleaner_val.tsv"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cleaner"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (requires workers > 0).")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pin_memory even when CUDA is available.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20, help="Stop if val loss does not improve for this many epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l1-weight", type=float, default=0.8, help="Weight for L1 loss component.")
    parser.add_argument("--ssim-weight", type=float, default=0.2, help="Weight for SSIM structural component.")
    parser.add_argument("--perceptual-weight", type=float, default=0.1, help="Weight for perceptual VGG16 loss component.")
    parser.add_argument("--encoder-pretrained", action="store_true", help="Use ImageNet-pretrained weights for the ResNet34 encoder.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights during training.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint (.pt) to resume from (e.g. outputs/cleaner/best_cleaner.pt).",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Restore optimizer/scheduler/scaler states when resuming (defaults to model only).",
    )
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

    if args.l1_weight < 0 or args.ssim_weight < 0 or args.perceptual_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if args.l1_weight + args.ssim_weight + args.perceptual_weight <= 0:
        raise ValueError("At least one loss component must carry positive weight.")
    if args.prefetch_factor < 0:
        raise ValueError("Prefetch factor must be non-negative.")
    if args.num_workers > 0 and args.prefetch_factor != 0 and args.prefetch_factor < 2:
        raise ValueError("Prefetch factor must be >= 2 when using worker processes.")

    try:
        from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler  # type: ignore

        autocast_ctx = amp_autocast
        grad_scaler_cls = AmpGradScaler
        autocast_kwargs = {"device_type": device.type}
    except (ImportError, AttributeError):
        autocast_ctx = torch.cuda.amp.autocast  # type: ignore[attr-defined]
        grad_scaler_cls = torch.cuda.amp.GradScaler  # type: ignore[attr-defined]
        autocast_kwargs = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CleanerDataset(args.train_manifest, args.image_root, augment=True)
    val_ds = CleanerDataset(args.val_manifest, args.image_root, augment=False)

    pin_memory = False if args.no_pin_memory else device.type == "cuda"
    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if args.num_workers > 0 and args.prefetch_factor > 0:
        common_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = CleanerUNet(
        pretrained_encoder=args.encoder_pretrained,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    criterion = nn.L1Loss()
    perceptual_loss_fn = VGGPerceptualLoss().to(device) if args.perceptual_weight > 0 else None
    if perceptual_loss_fn is not None:
        perceptual_loss_fn.eval()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = grad_scaler_cls(enabled=amp_enabled)

    best_val = float("inf")
    best_epoch = 0
    history: List[Metrics] = []
    start_epoch = 0
    metrics_path = args.output_dir / "history.json"

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if args.resume_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            if "scaler_state" in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state"])
                except Exception:
                    pass
        best_val = checkpoint.get("val_loss", best_val)
        best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", 0))
        start_epoch = checkpoint.get("epoch", 0)
        print(
            f"Loaded checkpoint {args.resume_checkpoint} (epoch {start_epoch}, best epoch {best_epoch}, best val {best_val:.4f})"
        )
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                history = [Metrics(**item) for item in existing]
            except Exception:
                history = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_perc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for noisy, clean, _ in progress:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(enabled=amp_enabled, **autocast_kwargs):
                pred = model(noisy)
                l1_term = criterion(pred, clean)
                ssim_scores = structural_similarity(pred, clean)
                ssim_term = 1.0 - ssim_scores.mean()
                loss = args.l1_weight * l1_term + args.ssim_weight * ssim_term

            perceptual_term = torch.tensor(0.0, device=device)
            if perceptual_loss_fn is not None:
                perceptual_term = perceptual_loss_fn(pred, clean)
                loss = loss + args.perceptual_weight * perceptual_term

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * noisy.size(0)
            if perceptual_loss_fn is not None:
                train_perc += perceptual_term.detach().item() * noisy.size(0)

            postfix = {
                "loss": f"{loss.item():.4f}",
                "ssim": f"{ssim_scores.mean().detach().item():.3f}",
            }
            if perceptual_loss_fn is not None:
                postfix["perc"] = f"{perceptual_term.detach().item():.4f}"
            progress.set_postfix(postfix)

        train_loss /= len(train_ds)
        if perceptual_loss_fn is not None:
            train_perc /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim_total = 0.0
        val_perc_total = 0.0
        with torch.no_grad():
            for noisy, clean, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with autocast_ctx(enabled=amp_enabled, **autocast_kwargs):
                    pred = model(noisy)
                    l1_term = criterion(pred, clean)
                    ssim_scores = structural_similarity(pred, clean)
                    ssim_term = 1.0 - ssim_scores.mean()
                    loss = args.l1_weight * l1_term + args.ssim_weight * ssim_term
                    mse = torch.mean((pred - clean) ** 2, dim=(1, 2, 3))
                if perceptual_loss_fn is not None:
                    perceptual_term = perceptual_loss_fn(pred, clean)
                    loss = loss + args.perceptual_weight * perceptual_term
                    val_perc_total += perceptual_term.detach().item() * noisy.size(0)
                val_loss += loss.item() * noisy.size(0)
                val_psnr += psnr(mse).sum().item()
                val_ssim_total += ssim_scores.sum().item()

        val_loss /= len(val_ds)
        val_psnr /= len(val_ds)
        val_ssim = val_ssim_total / len(val_ds)
        val_perc = val_perc_total / len(val_ds) if perceptual_loss_fn is not None else 0.0
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < old_lr:
            print(f"Validation plateau detected, reducing LR from {old_lr:.6f} to {new_lr:.6f}")

        history.append(Metrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_psnr=val_psnr, val_ssim=val_ssim))

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_epoch = epoch

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_epoch": best_epoch,
        }

        if improved:
            torch.save(checkpoint_payload, args.output_dir / "best_cleaner.pt")

        torch.save(checkpoint_payload, args.output_dir / "last_cleaner.pt")

        # Persist training history incrementally
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in history], f, indent=2)

        message = (
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | val PSNR {val_psnr:.2f} dB | val SSIM {val_ssim:.3f}"
        )
        if perceptual_loss_fn is not None:
            message += f" | train Perc {train_perc:.4f} | val Perc {val_perc:.4f}"
        if improved:
            message += " <-- best"
        print(message)

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
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": not args.no_pin_memory and device.type == "cuda",
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "seed": args.seed,
            "amp": amp_enabled,
            "l1_weight": args.l1_weight,
            "ssim_weight": args.ssim_weight,
            "perceptual_weight": args.perceptual_weight,
            "encoder_pretrained": args.encoder_pretrained,
            "freeze_encoder": args.freeze_encoder,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }, f, indent=2)

    print(f"Training complete. Best epoch: {best_epoch}, best val loss: {best_val:.4f}")
    print(f"Artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
