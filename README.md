# CRNN + AFFN CAPTCHA Solver

Quick runnable project that trains a small CRNN+AFFN model on synthetic CAPTCHA data.

See `generate_dataset.py` to create a synthetic dataset (default 2000 images, 80/20 split).

Usage (Linux):

```bash
cd ~/captcha_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate data
python generate_dataset.py --total 2000 --out data

# quick smoke test
python -m captcha_solver.smoke_test

# short training run
python -m captcha_solver.train --train-dir data/train --val-dir data/val --epochs 2 --batch-size 8
```
