#!/usr/bin/env python3
"""Evaluate a fine-tuned TrOCR model on a manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from captcha_solver.trocr_dataset import TrOCRManifestDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Directory of the fine-tuned model")
    parser.add_argument("--manifest", type=Path, default=Path("annotations/val.tsv"))
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-target-length", type=int, default=12)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, cuda, cuda:0)")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N samples")
    parser.add_argument("--report-mismatches", type=int, default=20)
    return parser.parse_args()


def collate_fn(batch: List[Dict]):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["label_text"] for item in batch]
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "paths": paths}


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    dataset = TrOCRManifestDataset(
        args.manifest,
        processor,
        image_root=args.image_root,
        max_target_length=args.max_target_length,
        return_metadata=True,
    )

    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    cer_metric = evaluate.load("cer")
    preds: List[str] = []
    refs: List[str] = []
    mismatches: List[Dict[str, str]] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                num_beams=args.num_beams,
                max_length=args.max_target_length,
                early_stopping=True,
            )
        pred_text = processor.batch_decode(generated, skip_special_tokens=True)
        preds.extend(pred_text)
        refs.extend(batch["labels"])

        for path, pred, ref in zip(batch["paths"], pred_text, batch["labels"]):
            if pred.strip() != ref.strip() and len(mismatches) < args.report_mismatches:
                mismatches.append({"path": path, "pred": pred, "label": ref})

    cer = cer_metric.compute(predictions=preds, references=refs)
    exact = sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs)

    print("Samples evaluated:", len(refs))
    print(f"CER: {cer:.4f}")
    print(f"Exact match: {exact * 100:.2f}%")

    if mismatches:
        print("\nSample mismatches:")
        for item in mismatches:
            print(f"{item['path']}: pred={item['pred']} | label={item['label']}")


if __name__ == "__main__":
    main()
