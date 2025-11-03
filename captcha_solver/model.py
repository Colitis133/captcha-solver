"""CRNN + AFFN model implementation with residual CNN encoder."""

from typing import Tuple

import torch
import torch.nn as nn


class AFFN(nn.Module):
    """Adaptive Fusion Filtering Network."""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 8, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualConvBlock(nn.Module):
    """Two-layer residual conv block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class CRNN_AFFN(nn.Module):
    """Residual CNN + BiLSTM sequence model for CAPTCHA decoding."""

    def __init__(
        self,
        n_classes: int,
        img_h: int = 60,
        img_w: int = 160,
        dropout: float = 0.25,
        cnn_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
    ):
        super().__init__()
        if len(cnn_channels) != 4:
            raise ValueError("cnn_channels must contain exactly four entries")
        c1, c2, c3, c4 = cnn_channels

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.Sequential(
            ResidualConvBlock(c1, c2, stride=2),
            nn.Dropout(dropout),
            ResidualConvBlock(c2, c3, stride=2),
            nn.Dropout(dropout),
            ResidualConvBlock(c3, c3),
            nn.Dropout(dropout),
            ResidualConvBlock(c3, c4),
        )

        self.affn = AFFN(c4)
        self.conv_proj = nn.Conv2d(c4, c4, 3, padding=1, bias=False)
        self.conv_proj_bn = nn.BatchNorm2d(c4)
        self.conv_proj_relu = nn.ReLU(inplace=True)
        self.conv_proj_dropout = nn.Dropout(dropout)

        feat_h = max(1, img_h // 4)
        rnn_input_size = c4 * feat_h
        self.rnn = nn.LSTM(
            rnn_input_size,
            lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.rnn_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.encoder(feat)
        feat = self.affn(feat)
        feat = self.conv_proj(feat)
        feat = self.conv_proj_bn(feat)
        feat = self.conv_proj_relu(feat)
        feat = self.conv_proj_dropout(feat)

        b, c, h, w = feat.size()
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = feat.view(b, w, c * h)

        rnn_out, _ = self.rnn(feat)
        rnn_out = self.rnn_dropout(rnn_out)
        logits = self.classifier(rnn_out)
        return logits.permute(1, 0, 2)


def make_model(n_classes: int, **kwargs) -> CRNN_AFFN:
    return CRNN_AFFN(n_classes, **kwargs)


if __name__ == '__main__':
    m = make_model(36 + 1)
    x = torch.randn(2, 1, 60, 160)
    y = m(x)
    print('out shape', y.shape)
