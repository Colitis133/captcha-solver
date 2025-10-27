"""Generate a synthetic dataset on-disk with labeled filenames.

Writes images into OUT_DIR/train and OUT_DIR/val. Filenames include the label, e.g. A7K9P_0001.png
"""
import argparse
import os
from captcha_solver.generate_synthetic import gen_captcha


def generate(total=2000, out_dir='data', train_ratio=0.8):
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    n_train = int(total * train_ratio)
    n_val = total - n_train

    print(f'Generating {n_train} train and {n_val} val images into {out_dir}')

    i = 0
    for dest, n in ((train_dir, n_train), (val_dir, n_val)):
        for k in range(n):
            img, label = gen_captcha()
            fname = f"{label}_{i:06d}.png"
            path = os.path.join(dest, fname)
            img.save(path)
            i += 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--total', type=int, default=2000)
    p.add_argument('--out', type=str, default='data')
    p.add_argument('--train-ratio', type=float, default=0.8)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(total=args.total, out_dir=args.out, train_ratio=args.train_ratio)
