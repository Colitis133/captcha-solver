"""Training script for CAPTCHA solver."""

import argparse
import torch
from torch import nn, optim, amp
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CaptchaDataset, collate_fn, StyleBalancedBatchSampler
from .model import make_model
from .utils import save_checkpoint, ctc_greedy_decode


def train(args):
    # basic charmap (digits + uppercase)
    chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    n_classes = len(chars) + 1  # +1 for CTC blank (0)

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    cnn_channels = tuple(args.cnn_channels) if args.cnn_channels else (64, 128, 256, 512)
    if len(cnn_channels) != 4:
        raise ValueError('--cnn-channels expects exactly four integers, e.g. 64 128 256 512')

    model = make_model(
        n_classes,
        img_h=args.img_h,
        img_w=args.img_w,
        dropout=args.dropout,
        cnn_channels=cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    amp_enabled = args.amp and device.type == 'cuda'
    if torch.cuda.is_available():
        scaler = amp.GradScaler('cuda', enabled=amp_enabled)
    else:
        scaler = amp.GradScaler('cpu', enabled=False)

    train_dataset = CaptchaDataset(args.train_dir, img_size=(args.img_w, args.img_h))
    val_dataset = CaptchaDataset(args.val_dir, img_size=(args.img_w, args.img_h)) if args.val_dir else None

    if args.balance_styles and len(train_dataset.files) > 0:
        batch_sampler = StyleBalancedBatchSampler(train_dataset, batch_size=args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                                  collate_fn=lambda b: collate_fn(b, char2idx))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_fn(b, char2idx))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, char2idx)) if val_dataset else None

    best_loss = float('inf')
    # early stopping state
    best_val_loss = float('inf')
    epochs_no_improve = 0

    scheduler = None
    if val_loader and args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_gamma,
                                      patience=args.lr_patience, min_lr=args.min_lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0.0
        for imgs, targets, lengths, labels, styles, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=amp_enabled):
                logits = model(imgs)
                input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                loss = criterion(logits.log_softmax(2), targets, input_lengths, lengths)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            lr = optimizer.param_groups[0]['lr']
            unique_styles = len(set(styles))
            pbar.set_postfix(loss=loss.item(), lr=lr, styles=unique_styles)

        avg = epoch_loss / (len(train_loader) if len(train_loader) else 1)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")
        # Save checkpoint on training loss improvement if no validation is available
        if not val_loader:
            if avg + 1e-12 < best_loss:
                best_loss = avg
                save_checkpoint(args.checkpoint, model, optimizer=optimizer, epoch=epoch)

        # validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets, lengths, labels, styles, _ in val_loader:
                    imgs = imgs.to(device)
                    with amp.autocast('cuda', enabled=amp_enabled):
                        logits = model(imgs)
                        input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                        loss = criterion(logits.log_softmax(2), targets, input_lengths, lengths)
                    val_loss += loss.item()
            avg_val = val_loss / (len(val_loader) if len(val_loader) else 1)
            print(f"Validation avg loss: {avg_val:.4f}")

            # show a sample prediction
            with torch.no_grad():
                for imgs, targets, lengths, labels, styles, _ in val_loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    pred = ctc_greedy_decode(logits, idx2char)
                    print('example pred/true:', pred[0], labels[0])
                    break

            if scheduler:
                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(avg_val)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr:
                    print(f"Scheduler reduced lr: {prev_lr:.6f} -> {new_lr:.6f}")

            improved = (avg_val + args.min_delta) < best_val_loss
            if improved:
                print(f"Validation loss improved ({best_val_loss:.4f} -> {avg_val:.4f}), saving checkpoint.")
                best_val_loss = avg_val
                epochs_no_improve = 0
                save_checkpoint(args.checkpoint, model, optimizer=optimizer, epoch=epoch)
            elif args.early_stop:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{args.patience} epochs.")
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered (patience={args.patience}).")
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
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--dropout', type=float, default=0.25)
    p.add_argument('--img-w', type=int, default=160)
    p.add_argument('--img-h', type=int, default=60)
    p.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    p.add_argument('--export', type=str, default='captcha_model.pt')
    # Use CUDA by default when available. Pass --no-cuda to force CPU.
    p.add_argument('--use-cuda', dest='use_cuda', action='store_true', help='use CUDA if available (deprecated, use --no-cuda to disable)')
    p.add_argument('--no-cuda', dest='use_cuda', action='store_false', help='disable CUDA and use CPU')
    p.set_defaults(use_cuda=True)
    p.add_argument('--early-stop', dest='early_stop', action='store_true', help='enable early stopping based on validation loss')
    p.add_argument('--no-early-stop', dest='early_stop', action='store_false', help='disable early stopping')
    p.set_defaults(early_stop=False)
    p.add_argument('--patience', type=int, default=5, help='epochs with no improvement before stopping')
    p.add_argument('--min-delta', type=float, default=1e-3, help='minimum change in monitored quantity to qualify as improvement')
    p.add_argument('--balance-styles', action='store_true', help='cycle batches across styles for the training split')
    p.add_argument('--no-balance-styles', dest='balance_styles', action='store_false')
    p.set_defaults(balance_styles=True)
    p.add_argument('--scheduler', action='store_true', help='enable ReduceLROnPlateau scheduler on validation loss')
    p.add_argument('--no-scheduler', dest='scheduler', action='store_false')
    p.set_defaults(scheduler=True)
    p.add_argument('--lr-patience', type=int, default=2, help='epochs with no val improvement before LR decay')
    p.add_argument('--lr-gamma', type=float, default=0.5, help='multiplicative LR decay factor when scheduler steps')
    p.add_argument('--min-lr', type=float, default=1e-6, help='floor for the learning rate when scheduler is active')
    p.add_argument('--cnn-channels', type=int, nargs='+', default=None, help='Override CNN channel widths (4 ints)')
    p.add_argument('--lstm-hidden', type=int, default=256, help='Hidden size of BiLSTM layers')
    p.add_argument('--lstm-layers', type=int, default=2, help='Number of stacked BiLSTM layers')
    p.add_argument('--max-grad-norm', type=float, default=5.0, help='Gradient clipping norm (<=0 disables)')
    p.add_argument('--amp', dest='amp', action='store_true', help='Enable mixed precision when CUDA is available')
    p.add_argument('--no-amp', dest='amp', action='store_false', help='Disable mixed precision')
    p.set_defaults(amp=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
