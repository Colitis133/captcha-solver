GPU Finetuning Quickstart

This repository is prepared so you can pull it to a GPU host and run a small finetune job with PaddleOCR.

Preconditions on the GPU host
- A CUDA-capable GPU with compatible drivers.
- git, python3, and python venv support.

Quick checklist (one-time on GPU host)

1) Clone the repository and enter it:

```bash
git clone https://github.com/Colitis133/captcha-solver.git
cd captcha-solver
```

2) Create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3) Install PaddlePaddle GPU wheel matching your CUDA version.
   - Check CUDA driver/GPU info:

```bash
nvidia-smi
```

   - Visit https://www.paddlepaddle.org.cn/whl/stable.html and pick the wheel for your CUDA.
     Example (CUDA 11.8) — run on the GPU host, replacing with the recommended command from the official site:

```bash
python -m pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/stable.html
```

4) Install PaddleOCR python dependencies:

```bash
pip install -r paddle_ocr_repo/requirements.txt
```

5) (Optional) Install small repo deps:

```bash
pip install -r requirements.txt
```

6) Verify imports (quick smoke):

```bash
python - <<'PY'
import paddle
print('paddle', paddle.__version__)
import sys
sys.path.insert(0, 'paddle_ocr_repo')
from ppocr.utils.utility import _initialize
print('ppocr import OK')
PY
```

7) Prepare manifests (already present here but safe to re-run):

```bash
python generate_dataset.py --out data
python prepare_paddle_labels.py --train-dir data/train --val-dir data/val --output-dir paddle_ocr/labels
```

8) Set PaddleOCR root and launch finetune (this repo contains a small wrapper):

```bash
export PADDLE_OCR_ROOT=$(pwd)/paddle_ocr_repo
./paddle_ocr/run_finetune.sh
```

Notes
- The wrapper uses `configs/rec/rec_mv3_none_bilstm_ctc.yml`. Adjust overrides inside `paddle_ocr/run_finetune.sh` if you want different LR, max_text_length, or pretrained checkpoint.
- Validation manifest currently contains a small sample for quick smoke runs; for real finetuning, regenerate a larger validation split.
- If you want me to tune the exact command for a specific CUDA/Paddle version, provide `nvidia-smi` output and Python version on the GPU host and I’ll return an exact `pip install` line.
