"""ResNet-based U-Net (ResUNet) for CAPTCHA cleaning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _DoubleConv(nn.Module):
    """Two stacked Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class _UpBlock(nn.Module):
    """Upsample, fuse skip connection, and refine with convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = _DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class _ResNet34Encoder(nn.Module):
    """ResNet-34 feature extractor used as the encoder trunk."""

    def __init__(self, in_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = models.resnet34(weights=weights)

        if in_channels != 3:
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained and backbone.conv1.weight.shape[1] == 3:
                if in_channels == 1:
                    with torch.no_grad():
                        conv1.weight.copy_(backbone.conv1.weight.sum(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = conv1

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.layer0(x)  # 1/2 resolution
        x1 = self.layer1(self.maxpool(x0))  # 1/4 resolution
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        return x0, x1, x2, x3, x4


class CleanerUNet(nn.Module):
    """ResUNet with a ResNet-34 encoder and lightweight decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        *,
        pretrained_encoder: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 64)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x0, x1, x2, x3, x4 = self.encoder(x)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        x = self.head(x)
        return torch.sigmoid(x)


__all__ = ["CleanerUNet"]
