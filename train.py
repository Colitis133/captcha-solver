#This program trains a CRNN model for captcha recognition using PyTorch.
#It includes data loading, model training, validation, and checkpointing functionalities.
#The code is structured to allow for easy configuration through a JSON file and supports mixed precision training.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import math

from model import CRNN
from data import CaptchaDataset, collate_fn
from utils import CTCDecoder, Metrics, setup_logger, load_config
from backup_manager import BackupManager

# Try to import TPU libraries
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    XLA_AVAILABLE = True
except ImportError:
    xmp = None
    XLA_AVAILABLE = False

#Define a custom learning rate scheduler with warmup.
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, main_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.main_scheduler = main_scheduler
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.main_scheduler.step()

#Train one epoch of the model.
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, config, scheduler, decoder, charset):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    progress_bar = tqdm(dataloader, desc="Training")
    
    for i, (images, labels_padded, label_lengths) in enumerate(progress_bar):
        if not TPU_AVAILABLE:
            images = images.to(device, non_blocking=True)
            labels_padded = labels_padded.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='xla' if TPU_AVAILABLE else 'cuda', enabled=config['training']['mixed_precision']):
            preds = model(images)
            log_probs = F.log_softmax(preds, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(1), dtype=torch.long, device=device)
            
            loss = criterion(log_probs.permute(1, 0, 2), labels_padded, input_lengths, label_lengths)
            
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Skipping batch {i} due to NaN/Inf loss.")
            continue
            
        if TPU_AVAILABLE:
            loss.backward()
            xm.optimizer_step(optimizer)
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        if i % 100 == 0 and i > 0:
            decoded_preds = decoder(preds)
            batch_targets = []
            for j in range(len(labels_padded)):
                label_indices = labels_padded[j][:label_lengths[j]]
                target_str = ''.join(charset[idx] for idx in label_indices)
                batch_targets.append(target_str)
            all_preds.extend(decoded_preds[:3])
            all_targets.extend(batch_targets[:3])
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
        
    return total_loss / len(dataloader), all_preds, all_targets

#Validate the model on the validation dataset.
def validate(model, dataloader, criterion, decoder, device, config, charset):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, labels_padded, label_lengths in tqdm(dataloader, desc="Validating"):
            if not TPU_AVAILABLE:
                images = images.to(device, non_blocking=True)
                labels_padded = labels_padded.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)
            
            with autocast(device_type='xla' if TPU_AVAILABLE else 'cuda', enabled=config['training']['mixed_precision']):
                preds = model(images)
                log_probs = F.log_softmax(preds, dim=2)
                input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(1), dtype=torch.long, device=device)
                
                loss = criterion(log_probs.permute(1, 0, 2), labels_padded, input_lengths, label_lengths)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
            
            decoded_preds = decoder(preds)
            all_preds.extend(decoded_preds)
            
            batch_targets = []
            for i in range(len(labels_padded)):
                label_indices = labels_padded[i][:label_lengths[i]]
                target_str = ''.join(charset[idx] for idx in label_indices)
                batch_targets.append(target_str)
            all_targets.extend(batch_targets)

    val_cer = Metrics.character_error_rate(all_preds, all_targets)
    val_seq_acc = Metrics.sequence_accuracy(all_preds, all_targets)
    val_char_acc = Metrics.character_accuracy(all_preds, all_targets)
    
    return total_loss / len(dataloader), val_cer, val_seq_acc, val_char_acc, all_preds, all_targets

#Save the model checkpoint.
def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, config, is_best=False, previous_checkpoint=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.main_scheduler.state_dict() if hasattr(scheduler, 'main_scheduler') else scheduler.state_dict(),
        'val_acc': val_acc,
    }
    
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    if previous_checkpoint:
        prev_path = os.fspath(previous_checkpoint)
        if os.path.exists(prev_path) and os.path.abspath(prev_path) != os.path.abspath(checkpoint_path):
            try:
                os.remove(prev_path)
            except OSError:
                pass
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        torch.save(model.state_dict(), config['paths']['model_save'])

    return os.path.abspath(checkpoint_path)


# Multiprocessing entrypoint for TPUs must be a top-level function so it can be
# pickled/imported by worker processes. Import torch_xla modules here to avoid
# initializing XLA before spawn.
def _mp_fn(index, args):
    global device, TPU_AVAILABLE, pl, xm
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    device = xm.xla_device()
    TPU_AVAILABLE = True
    # Now run the normal main routine in the child process
    main(args)

#Main training loop.
def main(args):
    config = load_config(args.config)

    if args.azure_connection_string:
        backup_section = config.setdefault('backup', {})
        azure_section = backup_section.setdefault('azure', {})
        azure_section['connection_string'] = args.azure_connection_string
        env_name = azure_section.get('connection_string_env')
        if env_name:
            os.environ[env_name] = args.azure_connection_string

    if args.backup_interval is not None:
        backup_section = config.setdefault('backup', {})
        backup_section['incremental_interval'] = args.backup_interval

    base_dir = Path(__file__).resolve().parent
    for key in ['model_save', 'checkpoint_dir', 'log_dir']:
        if key in config['paths']:
            config['paths'][key] = str((base_dir / config['paths'][key]).resolve())
    best_model_path = Path(config['paths']['model_save'])
    os.makedirs(best_model_path.parent, exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)

    logger = setup_logger(config['paths']['log_dir'])

    backup_cfg = config.get('backup', {}) or {}
    backup_manager = None
    if backup_cfg.get('enabled'):
        azure_cfg = backup_cfg.get('azure') or {}
        if not azure_cfg:
            logger.warning("Backup enabled but no Azure configuration provided; disabling backups.")
        else:
            backup_manager = BackupManager(
                base_dir=base_dir,
                logger=logger,
                enabled=True,
                run_folder_prefix=backup_cfg.get('run_folder_prefix', 'run'),
                baseline_epoch=backup_cfg.get('baseline_epoch', 3),
                incremental_interval=backup_cfg.get('incremental_interval', 10),
                incremental_directories=backup_cfg.get('incremental_directories'),
                incremental_files=backup_cfg.get('incremental_files'),
                azure_config=azure_cfg,
            )
    
    logger.info(f"Using device: {device}")

    charset = config['data']['charset']
    
    train_dataset = CaptchaDataset(config['data']['train_path'], charset, config['data']['image_height'], config['data']['image_width'], is_train=True)
    val_dataset = CaptchaDataset(config['data']['val_path'], charset, config['data']['image_height'], config['data']['image_width'], is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=not TPU_AVAILABLE)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=not TPU_AVAILABLE)

    if TPU_AVAILABLE:
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(val_loader, device)

    model = CRNN(
        vocab_size=len(charset), 
        hidden_size=config['model']['hidden_size'], 
        attention_heads=config['model']['attention_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    logger.info(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    #Initialize the loss function.
    criterion = nn.CTCLoss(
        blank=len(charset),
        reduction='mean',
        zero_infinity=True
    )
    
    #Initialize the optimizer and learning rate scheduler.
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay'], 
        eps=config['optimizer']['eps'],
        betas=config['optimizer']['betas']
    )
    
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = int(total_steps * 0.15)
    
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler']['T_0'] * len(train_loader),
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['scheduler']['eta_min']
    )
    
    scheduler = WarmupScheduler(optimizer, warmup_steps, main_scheduler)
    
    scaler = torch.amp.GradScaler(enabled=config['training']['mixed_precision'] and not TPU_AVAILABLE)
    decoder = CTCDecoder(charset)

    #Initialize variables for training.
    best_val_acc = 0
    patience_counter = 0
    start_epoch = 0
    last_checkpoint_path = None
    
    #Load checkpoint if resuming training.
    if args.resume:
        if os.path.isfile(args.resume):
            resume_path = Path(args.resume).resolve()
            logger.info(f"Loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'main_scheduler'):
                 scheduler.main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Resuming from epoch {start_epoch}")
            last_checkpoint_path = resume_path
        else:
            logger.info(f"No checkpoint found at '{args.resume}', starting from scratch.")

    #Start training loop.
    for epoch in range(start_epoch, config['training']['epochs']):
        start_time = time.time()
        
        avg_train_loss, train_preds, train_targets = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config, scheduler, decoder, charset)
        avg_val_loss, val_cer, val_seq_acc, val_char_acc, val_preds, val_targets = validate(model, val_loader, criterion, decoder, device, config, charset)
        
        epoch_duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} | Time: {epoch_duration:.1f}s | LR: {current_lr:.2e}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val CER: {val_cer:.4f} | Seq Acc: {val_seq_acc*100:.1f}% | Char Acc: {val_char_acc*100:.1f}%")
        
        if len(train_preds) > 0:
            logger.info(f"Train Samples - Pred: {train_preds[:3]} | Target: {train_targets[:3]}")
        if len(val_preds) > 0:
            logger.info(f"Val Samples - Pred: {val_preds[:3]} | Target: {val_targets[:3]}")

        is_best = val_char_acc > best_val_acc
        if is_best:
            best_val_acc = val_char_acc
            patience_counter = 0
            logger.info(f"* New best model saved with char accuracy: {val_char_acc*100:.2f}%")
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or is_best:
            last_checkpoint_path = Path(
                save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_char_acc,
                config,
                is_best,
                    previous_checkpoint=last_checkpoint_path,
                )
            )
        if backup_manager:
            checkpoint_path_obj = last_checkpoint_path if last_checkpoint_path else None
            backup_manager.maybe_backup(epoch + 1, checkpoint_path_obj, best_model_path)
            
        if patience_counter >= config['validation']['early_stopping_patience']:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Captcha Recognition Model")
    #Add command line arguments for configuration and checkpointing.
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file.")
    #Add argument for resuming training from a checkpoint.
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument(
        "--azure-connection-string",
        type=str,
        default=None,
        help="Azure storage connection string to override config/env settings for backups.",
    )
    parser.add_argument(
        "--backup-interval",
        type=int,
        default=None,
        help="Epoch interval for incremental backups (overrides config).",
    )
    args = parser.parse_args()

    if XLA_AVAILABLE:
        # Use the module-level `_mp_fn` so it can be imported by worker processes.
        # Use nprocs=None to let PJRT choose the correct number of replicas (Kaggle TPU/PJRT friendly).
        xmp.spawn(_mp_fn, args=(args,), nprocs=None)
    else:
        global device, TPU_AVAILABLE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPU_AVAILABLE = False
        main(args)