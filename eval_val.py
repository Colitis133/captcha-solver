#!/usr/bin/env python3
"""Evaluate exported TorchScript model on validation set (exact-match accuracy).

Usage:
  source .venv/bin/activate
  python eval_val.py --model captcha_model.pt --val-dir data/val --batch-size 32
"""
import argparse
import torch
from torch.utils.data import DataLoader

from captcha_solver.dataset import CaptchaDataset, collate_fn
from captcha_solver.utils import ctc_greedy_decode


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='captcha_model.pt')
    p.add_argument('--val-dir', type=str, default='data/val')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--use-cuda', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # charmap consistent with training
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i + 1: c for i, c in enumerate(chars)}

    print('Loading model', args.model)
    model = torch.jit.load(args.model, map_location=device)
    model.to(device)
    model.eval()

    ds = CaptchaDataset(args.val_dir, img_size=(160, 60), use_synthetic=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, char2idx))

    total = 0
    correct = 0
    mismatches = []
    with torch.no_grad():
        for imgs, targets, lengths, labels, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)  # T,B,C
            preds = ctc_greedy_decode(logits, idx2char)
            for p, t in zip(preds, labels):
                total += 1
                if p == t:
                    correct += 1
                else:
                    if len(mismatches) < 20:
                        mismatches.append((p, t))

    acc = 100.0 * correct / total if total else 0.0
    print(f'Validation exact-match: {correct}/{total} = {acc:.2f}%')
    if mismatches:
        print('\nSample mismatches (predicted, true):')
        for p, t in mismatches:
            print(p, '  ', t)


if __name__ == '__main__':
    main()
