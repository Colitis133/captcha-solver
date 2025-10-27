"""Dataset and transforms for CAPTCHA images."""
from torch.utils.data import Dataset
from PIL import Image
import os
import re
import random
import torchvision.transforms as T

from .generate_synthetic import gen_captcha


FILENAME_LABEL_RE = re.compile(r"([A-Za-z0-9]+)_.*\.(?:png|jpg|jpeg)$")


class CaptchaDataset(Dataset):
    """Loads images from a folder and optionally mixes synthetic samples.

    Filename labels must be embedded, e.g., `A7K9P_42.png`.
    """

    def __init__(self, root_dir, img_size=(160, 60), use_synthetic=False, synth_ratio=0.5, transform=None):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
        if self.use_synthetic and random.random() < self.synth_ratio:
            img, label = gen_captcha()
        else:
            path = self.files[idx % len(self.files)]
            label = self._label_from_filename(path)
            img = Image.open(path).convert('RGB')

        img = img.resize(self.img_size, resample=Image.BILINEAR)
        img = self.transform(img)

        # label as string returned; decoding to integer ids handled in collate
        return img, label


def collate_fn(batch, charmap=None):
    """Collate to tensors and returns targets for CTC training.

    charmap: dict char->int mapping (optional)
    Returns: images tensor, targets (concatenated ints), target_lengths, labels(list)
    """
    imgs, labels = zip(*batch)
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

    return imgs, targets, lengths, list(labels), charmap


if __name__ == '__main__':
    print('Dataset module')
