"""U-Net style architecture for CAPTCHA cleaning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2 with optional dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with max-pool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x)


class CleanerUNet(nn.Module):
    """Moderate-depth U-Net tuned for 160×60 captchas."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down4 = Down(base_channels * 8, base_channels * 8, dropout=dropout)
        self.up1 = Up(base_channels * 16, base_channels * 4, dropout=dropout)
        self.up2 = Up(base_channels * 8, base_channels * 2, dropout=dropout)
        self.up3 = Up(base_channels * 4, base_channels, dropout=dropout)
        self.up4 = Up(base_channels * 2, base_channels, dropout=dropout)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


__all__ = [
    "CleanerUNet",
]
