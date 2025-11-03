"""Dataset and transforms for CAPTCHA images."""
from collections import defaultdict, deque
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageStat
import os
import re
import random
import torchvision.transforms as T

from .generate_synthetic import gen_captcha


FILENAME_LABEL_RE = re.compile(r"([A-Za-z0-9]+)_.*\.(?:png|jpg|jpeg)$")


def _is_useful_image(path: str, min_stddev: float = 4.0) -> bool:
    """Return False for unreadable files or nearly blank images."""
    try:
        with Image.open(path) as img:
            gray = img.convert('L')
            stat = ImageStat.Stat(gray)
            stddev = stat.stddev[0] if stat.stddev else 0.0
            return stddev >= min_stddev
    except (OSError, ValueError):
        return False


class CaptchaDataset(Dataset):
    """Loads images from a folder and optionally mixes synthetic samples.

    Filename labels must be embedded, e.g., `A7K9P_42.png`.
    """

    def __init__(
        self,
        root_dir,
        img_size=(160, 60),
        use_synthetic=False,
        synth_ratio=0.5,
        transform=None,
        allowed_styles=None,
    ):
        self.root_dir = root_dir
        self.files = []
        self.styles = []
        allowed = {s.lower() for s in allowed_styles} if allowed_styles else None
        for dirpath, _, filenames in os.walk(root_dir):
            style = os.path.basename(dirpath) if dirpath != root_dir else ""  # root-level style marker
            for fname in filenames:
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                full_path = os.path.join(dirpath, fname)
                label = self._label_from_filename(full_path)
                if not label:
                    continue
                style_key = style.lower()
                if allowed is not None and style_key not in allowed:
                    continue
                if not _is_useful_image(full_path):
                    continue
                self.files.append(full_path)
                self.styles.append(style)
        self.img_size = img_size
        self.use_synthetic = use_synthetic
        self.synth_ratio = synth_ratio
        self.transform = transform or T.Compose([
            T.Resize(img_size[::-1]),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return max(len(self.files), 1) if not self.use_synthetic else 1000000

    def _label_from_filename(self, path):
        m = FILENAME_LABEL_RE.search(os.path.basename(path))
        return m.group(1) if m else ""

    def __getitem__(self, idx):
        # decide synthetic or real
        style_tag = "synthetic"
        if self.use_synthetic and random.random() < self.synth_ratio:
            img, label = gen_captcha()
        else:
            if not self.files:
                raise IndexError("No image files found in dataset root")
            path = self.files[idx % len(self.files)]
            style = self.styles[idx % len(self.styles)] if self.styles else ""
            label = self._label_from_filename(path)
            img = Image.open(path).convert('RGB')
            style_tag = style or "unknown"

        img = img.resize(self.img_size, resample=Image.BILINEAR)
        img = self.transform(img)

        # label as string returned; decoding to integer ids handled in collate
        return img, label, style_tag


def collate_fn(batch, charmap=None):
    """Collate to tensors and returns targets for CTC training.

    charmap: dict char->int mapping (optional)
    Returns: images tensor, targets (concatenated ints), target_lengths, labels(list)
    """
    imgs, labels, styles = zip(*batch)
    import torch
    imgs = torch.stack(imgs, 0)

    if charmap is None:
        # build charmap from batch
        chars = sorted({c for s in labels for c in s})
        charmap = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for blank

    targets = []
    lengths = []
    for s in labels:
        ids = [charmap[c] for c in s]
        targets.extend(ids)
        lengths.append(len(ids))

    targets = torch.tensor(targets, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return imgs, targets, lengths, list(labels), list(styles), charmap


class StyleBalancedBatchSampler(Sampler):
    """Yield batches that cycle through available styles to avoid collapse."""

    def __init__(self, dataset: CaptchaDataset, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.style_to_indices = defaultdict(list)
        for idx, style in enumerate(dataset.styles):
            key = style or "unknown"
            self.style_to_indices[key].append(idx)
        self.num_samples = sum(len(v) for v in self.style_to_indices.values())

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.num_samples == 0:
            return iter([])

        local_buckets = {style: deque(indices) for style, indices in self.style_to_indices.items() if indices}
        for dq in local_buckets.values():
            indices_list = list(dq)
            random.shuffle(indices_list)
            dq.clear()
            dq.extend(indices_list)

        active_styles = list(local_buckets.keys())
        random.shuffle(active_styles)

        batch = []
        while active_styles:
            for style in list(active_styles):
                bucket = local_buckets[style]
                if not bucket:
                    active_styles.remove(style)
                    continue
                batch.append(bucket.popleft())
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            # If some styles emptied out during the loop, we continue with the rest

        if batch:
            yield batch


if __name__ == '__main__':
    print('Dataset module')
