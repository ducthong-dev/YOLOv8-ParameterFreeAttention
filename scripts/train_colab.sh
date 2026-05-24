#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/content/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/output}"
MODEL_CONFIG="${MODEL_CONFIG:-ultralytics/cfg/models/v8/yolov8n_sma_cls.yaml}"
RUN_NAME="${RUN_NAME:-yolov8n_sma_cls}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-224}"
CACHE="${CACHE:-False}"
WORKERS="${WORKERS:-2}"
DEVICE="${DEVICE:-auto}"
LR0="${LR0:-0.001}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
PROJECT="${PROJECT:-${OUTPUT_ROOT%/}/runs/classify}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python - <<'PY'
import torch
print("0" if torch.cuda.is_available() else "cpu")
PY
)"
fi

if [[ ! -d "$DATA_ROOT/train" ]]; then
  echo "ERROR: DATA_ROOT must contain train/ and val/ or test/ folders. Got: $DATA_ROOT" >&2
  exit 2
fi

mkdir -p "$PROJECT"

python yolo_local.py classify train \
  model="$MODEL_CONFIG" \
  data="$DATA_ROOT" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  optimizer="$OPTIMIZER" \
  lr0="$LR0" \
  project="$PROJECT" \
  name="$RUN_NAME" \
  device="$DEVICE" \
  workers="$WORKERS" \
  cache="$CACHE" \
  cos_lr=True \
  dropout=0.1 \
  label_smoothing=0.05 \
  plots=True \
  $EXTRA_ARGS
