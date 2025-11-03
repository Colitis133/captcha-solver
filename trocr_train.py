#!/usr/bin/env python3
"""Fine-tune TrOCR on the CAPTCHA dataset manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import evaluate
import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from captcha_solver.trocr_dataset import TrOCRManifestDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=Path("annotations/train.tsv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("annotations/val.tsv"))
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Base directory for relative image paths")
    parser.add_argument("--pretrained", type=str, default="microsoft/trocr-base-printed", help="Pretrained TrOCR checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trocr"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-target-length", type=int, default=12, help="Maximum decoded token length (including EOS)")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam width for generation during evaluation")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training if CUDA is available")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training if supported")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a Trainer checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the vision encoder (useful when only decoder fine-tuning is desired).",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze the text decoder (useful for encoder-only adaptation).",
    )
    return parser.parse_args()


def setup_regeneration_settings(model: VisionEncoderDecoderModel, processor: TrOCRProcessor, args: argparse.Namespace) -> None:
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = args.max_target_length
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 1.0
    model.config.num_beams = args.num_beams

    if processor.image_processor is not None:
        size = processor.image_processor.size
        if isinstance(size, dict):
            side = size.get("height", size.get("width", 384))
        else:
            side = 384
        processor.image_processor.size = {"height": side, "width": side}
        processor.image_processor.crop_size = {"height": side, "width": side}


def build_datasets(processor: TrOCRProcessor, args: argparse.Namespace):
    train_ds = TrOCRManifestDataset(
        args.train_manifest,
        processor,
        image_root=args.image_root,
        max_target_length=args.max_target_length,
    )
    eval_ds = TrOCRManifestDataset(
        args.val_manifest,
        processor,
        image_root=args.image_root,
        max_target_length=args.max_target_length,
    )
    return train_ds, eval_ds


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    processor = TrOCRProcessor.from_pretrained(args.pretrained)
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained)
    setup_regeneration_settings(model, processor, args)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen (requires_grad=False).")
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("Decoder parameters frozen (requires_grad=False).")

    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("All model parameters are frozen. Disable at least one freeze flag to train.")

    train_dataset, eval_dataset = build_datasets(processor, args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    metric = evaluate.load("cer")

    def compute_metrics(pred) -> Dict[str, float]:
        label_ids = pred.label_ids
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = metric.compute(predictions=pred_str, references=label_str)
        exact = np.mean([p.strip() == l.strip() for p, l in zip(pred_str, label_str)])
        return {"cer": cer, "exact_match": exact}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
        save_total_limit=2,
        seed=args.seed,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True if args.eval_strategy != "no" else False,
        metric_for_best_model="cer" if args.eval_strategy != "no" else None,
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.eval_strategy != "no" else None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
