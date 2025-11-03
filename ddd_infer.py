#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    import ddddocr
except Exception as e:
    print('Failed to import ddddocr:', e)
    raise

ROOT = Path.cwd()
LABEL_FILE = ROOT / 'annotations' / 'train.tsv'

if not LABEL_FILE.exists():
    print('Label file not found:', LABEL_FILE)
    sys.exit(1)

ocr = ddddocr.DdddOcr(ocr=True, det=False, use_gpu=False)

lines = [l.strip() for l in LABEL_FILE.read_text(encoding='utf-8').splitlines() if l.strip()]

total = 0
correct = 0
mismatches = []

for line in lines:
    if '\t' not in line:
        continue
    img_path, gt = line.split('\t', 1)
    img_path = (ROOT / img_path).resolve()
    gt = gt.strip()
    total += 1
    if not img_path.exists():
        mismatches.append((str(img_path), gt, '<MISSING>'))
        print(f'MISSING: {img_path}')
        continue
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        res = ocr.classification(img_bytes, png_fix=True)
        if isinstance(res, dict):
            pred = res.get('result') or res.get('text') or str(res)
        else:
            pred = str(res)
    except Exception as e:
        pred = f'<ERROR: {e}>'
    if pred.strip() == gt:
        correct += 1
    else:
        mismatches.append((str(img_path), gt, pred))

# Print summary
print('\nInference summary:')
print('Total samples:', total)
print('Correct:', correct)
print('Accuracy: {:.2f}%'.format(100.0 * correct / total if total > 0 else 0.0))

if mismatches:
    print('\nMismatches (path, ground_truth, prediction):')
    for p, g, pr in mismatches:
        print(p, '\t', g, '\t', pr)

# exit code nonzero if any errors/missing
if any('MISSING' in m[2] or m[2].startswith('<ERROR') for m in mismatches):
    sys.exit(2)
else:
    sys.exit(0)
