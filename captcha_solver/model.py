"""CRNN + AFFN model implementation (compact).

This is a small, deployable model intended for CPU inference and quick training.
"""
import torch
import torch.nn as nn


class AFFN(nn.Module):
    """Adaptive Fusion Filtering Network - small gating/fusion module."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8 + 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8 + 1, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.gate(x)
        return x * g


class CRNN_AFFN(nn.Module):
    """Small CRNN with AFFN filtering.

    Input: (B, 1, H, W)
    Output: (T, B, n_classes)
    """

    def __init__(self, n_classes, img_h=60, img_w=160, dropout=0.25):
        super().__init__()
        self.n_classes = n_classes

        # convolutional feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.affn = AFFN(256)

        # projection to sequence
        self.conv_proj = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_proj_dropout = nn.Dropout(0.25)

        # recurrent layers
        # The RNN input size depends on the convolutional feature height (collapsed)
        # After two MaxPool2d(2,2) operations the feature height is img_h // 4
        feat_h = max(1, img_h // 4)
        rnn_input_size = 256 * feat_h
        # use explicit modules because LSTM returns (output, (h,c)) not a Tensor
        self.rnn_layer = nn.LSTM(rnn_input_size, 128, bidirectional=True, batch_first=True)
        self.rnn_dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(128 * 2, n_classes)

    def forward(self, x):
        # x: B,1,H,W
        feat = self.cnn(x)
        feat = self.affn(feat)
        feat = self.conv_proj(feat)
        feat = self.conv_proj_dropout(feat)  # B, C, H', W'

        # collapse height -> features, treat width dimension as time
        b, c, h, w = feat.size()
        feat = feat.permute(0, 3, 1, 2).contiguous()  # B, W, C, H
        feat = feat.view(b, w, c * h)  # B, T, D

        # RNN
        rnn_out, _ = self.rnn_layer(feat)  # B, T, 2*H
        rnn_out = self.rnn_dropout(rnn_out)

        logits = self.classifier(rnn_out)  # B, T, n_classes
        # For CTC we need T, B, C
        logits = logits.permute(1, 0, 2)
        return logits


def make_model(n_classes, **kwargs):
    return CRNN_AFFN(n_classes, **kwargs)


if __name__ == '__main__':
    m = make_model(36 + 1)  # e.g., 36 chars + blank
    x = torch.randn(2, 1, 60, 160)
    y = m(x)
    print('out shape', y.shape)
