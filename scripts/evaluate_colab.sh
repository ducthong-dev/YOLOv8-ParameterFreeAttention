#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-/content/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/output}"
RUN_NAME="${RUN_NAME:-yolov8n_sma_cls}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-${OUTPUT_ROOT%/}/runs/classify/${RUN_NAME}/weights/best.pt}"
SPLIT="${SPLIT:-test}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-224}"
WORKERS="${WORKERS:-2}"
DEVICE="${DEVICE:-auto}"
PROJECT="${PROJECT:-${OUTPUT_ROOT%/}/runs/classify}"
VAL_NAME="${VAL_NAME:-${RUN_NAME}_${SPLIT}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export WANDB_DISABLED="${WANDB_DISABLED:-true}"

if [[ "$DEVICE" == "auto" ]]; then
  DEVICE="$(python - <<'PY'
import torch
print("0" if torch.cuda.is_available() else "cpu")
PY
)"
fi

if [[ ! -f "$MODEL_WEIGHTS" ]]; then
  echo "ERROR: MODEL_WEIGHTS not found: $MODEL_WEIGHTS" >&2
  exit 2
fi

python yolo_local.py classify val \
  model="$MODEL_WEIGHTS" \
  data="$DATA_ROOT" \
  split="$SPLIT" \
  imgsz="$IMGSZ" \
  batch="$BATCH" \
  device="$DEVICE" \
  workers="$WORKERS" \
  project="$PROJECT" \
  name="$VAL_NAME" \
  $EXTRA_ARGS
