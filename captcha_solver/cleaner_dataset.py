"""Dataset utilities for paired noisy/clean CAPTCHA images."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class CleanerSample:
    noisy_rel: Path
    clean_rel: Path
    label: str


class CleanerDataset(Dataset):
    """Loads noisy→clean image pairs for training the captcha cleaner."""

    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        augment: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.augment = augment
        self.samples = self._read_manifest(self.manifest_path)
        if not self.samples:
            raise ValueError(f"No samples found in manifest: {manifest_path}")

        self.to_tensor = transforms.ToTensor()

    def _read_manifest(self, manifest_path: Path) -> List[CleanerSample]:
        entries: List[CleanerSample] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                noisy_rel, clean_rel, label = row[:3]
                entries.append(
                    CleanerSample(
                        noisy_rel=Path(noisy_rel),
                        clean_rel=Path(clean_rel),
                        label=label.strip(),
                    )
                )
        return entries

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        noisy = Image.open(self.image_root / sample.noisy_rel).convert("RGB")
        clean = Image.open(self.image_root / sample.clean_rel).convert("RGB")

        if self.augment:
            noisy = self._augment_noisy(noisy)

        noisy_tensor = self.to_tensor(noisy)
        clean_tensor = self.to_tensor(clean)
        return noisy_tensor, clean_tensor, sample.label

    def _augment_noisy(self, img: Image.Image) -> Image.Image:
        augmented = img

        if random.random() < 0.6:
            factor = random.uniform(0.85, 1.15)
            augmented = ImageEnhance.Brightness(augmented).enhance(factor)

        if random.random() < 0.6:
            factor = random.uniform(0.85, 1.2)
            augmented = ImageEnhance.Contrast(augmented).enhance(factor)

        if random.random() < 0.4:
            radius = random.uniform(0.2, 1.1)
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=radius))

        if random.random() < 0.35:
            arr = np.array(augmented).astype(np.float32)
            noise = np.random.normal(0, 12.0, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(arr)

        if random.random() < 0.3:
            quality = random.randint(35, 70)
            augmented = self._jpeg_artifacts(augmented, quality)

        return augmented

    @staticmethod
    def _jpeg_artifacts(img: Image.Image, quality: int) -> Image.Image:
        from io import BytesIO

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


__all__ = ["CleanerDataset", "CleanerSample"]
