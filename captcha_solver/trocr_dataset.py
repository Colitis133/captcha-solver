
"""Datasets for TrOCR fine-tuning and evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TrOCRManifestDataset(Dataset):
    """Dataset that reads image paths and labels from a TSV manifest."""

    def __init__(
        self,
        manifest_path: Path | str,
        processor,
        *,
        image_root: Path | str | None = None,
        max_target_length: int = 32,
        return_metadata: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        if image_root is None:
            candidate = self.manifest_path.parent.parent / "data"
            if candidate.exists():
                image_root = candidate
            else:
                image_root = self.manifest_path.parent.parent
        self.image_root = Path(image_root)

        self.processor = processor
        self.max_target_length = max_target_length
        self.return_metadata = return_metadata

        self.samples: List[Tuple[Path, str]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "\t" not in line:
                    continue
                rel_path, label = line.split("\t", 1)
                img_path = (self.image_root / rel_path).resolve()
                if not img_path.exists():
                    raise FileNotFoundError(f"Image path not found: {img_path}")
                self.samples.append((img_path, label.strip()))

        if not self.samples:
            raise ValueError(f"No samples found in manifest: {self.manifest_path}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        encoding = self.processor(
            images=image,
            text=label,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        )
        sample = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.return_metadata:
            sample["path"] = str(img_path)
            sample["label_text"] = label
        return sample


# Backwards compatibility alias for older imports.
CaptchaManifestDataset = TrOCRManifestDataset

__all__ = ["TrOCRManifestDataset", "CaptchaManifestDataset"]
