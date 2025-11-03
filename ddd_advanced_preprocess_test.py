#!/usr/bin/env python3
"""Sweep ddddocr across multiple preprocessing pipelines."""

import argparse
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import ddddocr

ROOT = Path.cwd()
DEFAULT_LABEL_FILE = ROOT / "annotations" / "train.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=DEFAULT_LABEL_FILE,
        help="Path to label file (image\ttext).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Comma-separated substrings to filter image paths (case-insensitive).",
    )
    parser.add_argument(
        "--target-mode",
        choices=["full", "suffix"],
        default="full",
        help="full: compare whole label; suffix: compare only trailing characters.",
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=4,
        help="When --target-mode suffix, compare only the last N characters (default 4).",
    )
    return parser.parse_args()


def read_entries(label_file: Path, filters: Iterable[str]) -> List[Tuple[Path, str]]:
    if not label_file.exists():
        raise FileNotFoundError(label_file)
    lines = [
        line.strip()
        for line in label_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: List[Tuple[Path, str]] = []
    for line in lines:
        if "\t" not in line:
            continue
        rel_path, gt = line.split("\t", 1)
        img_path = (ROOT / rel_path).resolve()
        if filters and not any(f in img_path.as_posix().lower() for f in filters):
            continue
        results.append((img_path, gt.strip()))
    return results


def to_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def unsharp_mask(img: np.ndarray, ksize=(5, 5), sigma=1.0, amount=1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, ksize, sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def color_filter_hsv(img: np.ndarray, lower, upper) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def top_hat(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def morph_background_subtract(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel) + 1.0
    norm = np.clip((gray / bg) * 255.0, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def contrast_stretch(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(gamma, 1e-3)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(img, table)


def deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 255))
    if coords.size == 0:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def make_pipelines(img: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out["orig"] = bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    median = cv2.medianBlur(gray, 3)
    _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["median_otsu"] = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_mean = cv2.adaptiveThreshold(
        gaussian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10
    )
    out["gaussian_adapt"] = cv2.cvtColor(adaptive_mean, cv2.COLOR_GRAY2BGR)

    out["clahe"] = contrast_stretch(bgr)
    out["unsharp"] = unsharp_mask(bgr)
    out["morph_bg_sub"] = morph_background_subtract(bgr)
    out["tophat"] = top_hat(bgr)
    out["color_filter_dark"] = color_filter_hsv(bgr, (0, 0, 0), (180, 255, 80))

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    adaptive_gauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    open_close = cv2.morphologyEx(adaptive_gauss, cv2.MORPH_OPEN, kernel3)
    open_close = cv2.morphologyEx(open_close, cv2.MORPH_CLOSE, kernel3)
    out["adaptive_open_close"] = cv2.cvtColor(open_close, cv2.COLOR_GRAY2BGR)

    deskewed = deskew(gray)
    out["deskew"] = cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    if max(img.shape[:2]) < 400:
        big = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        out["big_unsharp"] = unsharp_mask(big)

    bilat_gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, th_bilat = cv2.threshold(
        bilat_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dilated = cv2.dilate(th_bilat, kernel3, iterations=1)
    out["bilat_otsu_dilate"] = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

    nlmeans = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    nlmeans_gray = cv2.cvtColor(nlmeans, cv2.COLOR_BGR2GRAY)
    _, nlmeans_otsu = cv2.threshold(
        nlmeans_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    out["nlmeans_otsu"] = cv2.cvtColor(nlmeans_otsu, cv2.COLOR_GRAY2BGR)

    gamma_img = gamma_correction(bgr, 0.7)
    gamma_gray = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
    gamma_adapt = cv2.adaptiveThreshold(
        gamma_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 12
    )
    gamma_closed = cv2.morphologyEx(gamma_adapt, cv2.MORPH_CLOSE, kernel3, iterations=1)
    out["gamma_adapt_close"] = cv2.cvtColor(gamma_closed, cv2.COLOR_GRAY2BGR)

    laplace = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    laplace_abs = cv2.convertScaleAbs(laplace)
    _, laplace_bin = cv2.threshold(
        laplace_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    out["laplace_binary"] = cv2.cvtColor(laplace_bin, cv2.COLOR_GRAY2BGR)

    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.addWeighted(
        cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0
    )
    _, sobel_bin = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sobel_dil = cv2.morphologyEx(sobel_bin, cv2.MORPH_DILATE, kernel3, iterations=1)
    out["sobel_dilate"] = cv2.cvtColor(sobel_dil, cv2.COLOR_GRAY2BGR)

    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel3)
    _, gradient_bin = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    out["morph_gradient"] = cv2.cvtColor(gradient_bin, cv2.COLOR_GRAY2BGR)

    clahe_bgr = cv2.cvtColor(cv2.cvtColor(contrast_stretch(bgr), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out["clahe_unsharp"] = unsharp_mask(clahe_bgr)

    gamma_dark = gamma_correction(bgr, 1.4)
    out["gamma_dark_unsharp"] = unsharp_mask(gamma_dark)

    return out


def normalize_text(value: str) -> str:
    return "".join(ch for ch in value.upper().strip() if ch.isalnum())


def determine_target(gt: str, mode: str, suffix_length: int) -> Tuple[str, str]:
    normalized = normalize_text(gt)
    if mode == "suffix":
        if suffix_length <= 0:
            raise ValueError("suffix_length must be > 0 when using suffix mode")
        if len(normalized) <= suffix_length:
            target_norm = normalized
        else:
            target_norm = normalized[-suffix_length:]
        target_display = gt[-suffix_length:] if len(gt) >= suffix_length else gt
        return target_norm, target_display
    return normalized, gt


def match_score(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 1.0
    return SequenceMatcher(None, gt, pred).ratio()


def run():
    args = parse_args()
    filters = [f.strip().lower() for f in args.filter.split(",") if f.strip()]
    entries = read_entries(args.label_file, filters)
    if not entries:
        raise SystemExit("No samples matched the provided criteria")

    samples = []
    for path, gt in entries:
        target_norm, target_display = determine_target(
            gt, args.target_mode, args.suffix_length
        )
        samples.append((path, gt, target_norm, target_display))

    if args.target_mode == "suffix":
        print(
            f"Matching on the last up to {args.suffix_length} normalized characters for each sample."
        )

    ocr = ddddocr.DdddOcr(ocr=True, det=False, use_gpu=False, show_ad=False)
    pipelines: Dict[str, int] = {}
    combined_hits = 0
    mismatches: List[Tuple[str, str, str, str, float]] = []

    for img_path, gt, target_norm, target_display in samples:
        if not img_path.exists():
            print("MISSING", img_path)
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print("LOAD FAIL", img_path)
            continue

        variants = make_pipelines(img)
        if not pipelines:
            pipelines.update({name: 0 for name in variants.keys()})

        variant_records = []
        for name, variant in variants.items():
            variant_bgr = (
                cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
                if variant.ndim == 2
                else variant
            )
            try:
                pred_raw = ocr.classification(to_bytes(variant_bgr), png_fix=True)
                if isinstance(pred_raw, dict):
                    pred_text = pred_raw.get("result") or pred_raw.get("text") or ""
                else:
                    pred_text = str(pred_raw)
            except Exception as exc:  # pragma: no cover - defensive
                pred_text = f"<ERROR:{exc}>"

            pred_norm = normalize_text(pred_text)
            is_match = pred_norm == target_norm
            score = match_score(target_norm, pred_norm)
            pipelines[name] += int(is_match)
            variant_records.append((name, pred_text, is_match, score))

        if any(record[2] for record in variant_records):
            combined_hits += 1
        else:
            best = max(variant_records, key=lambda item: item[3], default=None)
            if best:
                name, pred_text, _, score = best
                mismatches.append(
                    (
                        str(img_path),
                        gt,
                        target_display,
                        pred_text,
                        name,
                        score,
                    )
                )

    total = len(samples)
    print("\nAdvanced preprocessing results:")
    print("method\tcorrect\taccuracy")
    for name, correct in pipelines.items():
        accuracy = (correct / total * 100.0) if total else 0.0
        print(f"{name}\t{correct}\t{accuracy:.2f}%")
    combined_acc = (combined_hits / total * 100.0) if total else 0.0
    print(f"\nAny variant match: {combined_hits}/{total} ({combined_acc:.2f}%)")

    if mismatches:
        print("\nClosest predictions (no matches):")
        for path, gt, target, pred, variant, score in mismatches:
            print(
                f"{path}\tgt={gt}\ttarget={target}\tpred={pred}\tvariant={variant}\tscore={score:.2f}"
            )


if __name__ == "__main__":
    run()
