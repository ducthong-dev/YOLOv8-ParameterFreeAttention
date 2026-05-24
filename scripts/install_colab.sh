#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade --quiet pip setuptools wheel
python -m pip install --quiet -r requirements.txt

python - <<'PY'
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f_HPC_Lite, SSC

root = Path.cwd()
print(f"Python: {sys.version.split()[0]}")
print(f"Repo: {root}")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"YOLO class: {YOLO.__module__}.YOLO")
print(f"Custom modules: {SSC.__name__}, {C2f_HPC_Lite.__name__}")
PY
