# CRNN + AFFN CAPTCHA Solver

Quick runnable project that trains a small CRNN+AFFN model on synthetic CAPTCHA data.

See `generate_dataset.py` to create the synthetic dataset (defaults to 2,000 train images
balanced across the 20 training styles, plus 500 validation images from the Mutant Hybrid style).

Usage (Linux):

```bash
cd ~/captcha_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate data (2,000 train / 500 val by default)
python generate_dataset.py --out data

# quick smoke test
python -m captcha_solver.smoke_test

# short training run (residual CNN + BiLSTM, AMP/grad clipping on by default when CUDA is available)
python -m captcha_solver.train --train-dir data/train --val-dir data/val --epochs 2 --batch-size 8
```

Or use the bundled setup helper which creates a venv, installs deps, generates data,
runs the smoke test and a short training run:

```bash
./setup_env.sh         # full flow (create .venv, install deps, make data, smoke test, short train)
./setup_env.sh --no-data    # skip dataset generation
./setup_env.sh --no-smoke   # skip smoke test
./setup_env.sh --no-train   # skip the short training run
```

Fonts
-----

The repository vendors a small set of open-licensed fonts under `fonts/` so every generator
style works out of the box and offline. See `fonts/README.md` for the list of families and
their licenses. Feel free to add more `.ttf` files to that directory if you want additional
visual variety—styles will automatically pick them up.

GPU notes
---------

Training will use a GPU by default when one is available. The training CLI accepts an opt-out flag:

```bash
# GPU enabled by default when available
python -m captcha_solver.train --train-dir data/train --val-dir data/val --epochs 10 --batch-size 32

# force CPU
python -m captcha_solver.train --no-cuda --train-dir data/train --val-dir data/val --epochs 10 --batch-size 32
```

If you need a specific CUDA-enabled PyTorch wheel, install that inside the venv before installing the rest of `requirements.txt`.
