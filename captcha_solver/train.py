"""Training script for CAPTCHA solver (small, runnable example).

This script provides a convenient entrypoint for training on a synthetic dataset
split into train/ and val/ directories.
"""
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CaptchaDataset, collate_fn
from .model import make_model
from .utils import save_checkpoint, ctc_greedy_decode


def train(args):
    # basic charmap (digits + uppercase)
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    n_classes = len(chars) + 1  # +1 for CTC blank (0)

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    model = make_model(n_classes, img_h=args.img_h, img_w=args.img_w, dropout=args.dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    train_dataset = CaptchaDataset(args.train_dir, img_size=(args.img_w, args.img_h))
    val_dataset = CaptchaDataset(args.val_dir, img_size=(args.img_w, args.img_h)) if args.val_dir else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, char2idx))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, char2idx)) if val_dataset else None

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        for imgs, targets, lengths, labels, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)  # T,B,C
            input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
            loss = criterion(logits.log_softmax(2), targets, input_lengths, lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss)

        avg = epoch_loss / (len(train_loader) if len(train_loader) else 1)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(args.checkpoint, model, optimizer=optimizer, epoch=epoch)

        # validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets, lengths, labels, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                    loss = criterion(logits.log_softmax(2), targets, input_lengths, lengths)
                    val_loss += loss.item()
            avg_val = val_loss / (len(val_loader) if len(val_loader) else 1)
            print(f"Validation avg loss: {avg_val:.4f}")

            # show a sample prediction
            with torch.no_grad():
                for imgs, targets, lengths, labels, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    pred = ctc_greedy_decode(logits, idx2char)
                    print('example pred/true:', pred[0], labels[0])
                    break

    # export
    if args.export:
        model.eval()
        scripted = torch.jit.script(model.cpu())
        scripted.save(args.export)
        print('Saved TorchScript to', args.export)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', type=str, default='data/train', help='directory with labeled train images')
    p.add_argument('--val-dir', type=str, default='data/val', help='directory with labeled val images')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--img-w', type=int, default=160)
    p.add_argument('--img-h', type=int, default=60)
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    p.add_argument('--export', type=str, default='captcha_model.pt')
    p.add_argument('--use-cuda', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
