# Google Colab Workflow

This repository contains a local Ultralytics YOLOv8 fork with custom classification modules (`SSC` and `C2f_HPC_Lite`). In Colab, run commands through `python yolo_local.py ...` or the scripts in `scripts/` so Python imports this repository instead of the pip `ultralytics` package.

## Setup

```bash
git clone -b feature/google-colab https://github.com/ducthong-dev/YOLOv8-ParameterFreeAttention.git
cd YOLOv8-ParameterFreeAttention
pip install -r requirements.txt
```

Recommended Colab install:

```bash
bash scripts/install_colab.sh
```

Colab already includes a CUDA-enabled PyTorch build. Do not install `torch` or `torchvision` from this repository's `requirements.txt`.

## Dataset

Use the Ultralytics classification folder format:

```text
/content/data/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

You can also keep the dataset in Google Drive:

```bash
export DATA_ROOT=/content/drive/MyDrive/dataset
```

Default environment variables:

```bash
export DATA_ROOT=/content/data
export OUTPUT_ROOT=/content/output
export WANDB_DISABLED=true
```

## Train

```bash
DATA_ROOT=/content/data OUTPUT_ROOT=/content/output \
bash scripts/train_colab.sh
```

Equivalent direct command:

```bash
python yolo_local.py classify train \
  model=ultralytics/cfg/models/v8/yolov8n_sma_cls.yaml \
  data=/content/data \
  imgsz=224 \
  epochs=100 \
  batch=64 \
  optimizer=AdamW \
  lr0=0.001 \
  project=/content/output/runs/classify \
  name=yolov8n_sma_cls \
  device=0
```

Useful overrides:

```bash
EPOCHS=50 BATCH=32 CACHE=ram bash scripts/train_colab.sh
```

## Resume

```bash
python yolo_local.py classify train \
  model=/content/output/runs/classify/yolov8n_sma_cls/weights/last.pt \
  data=/content/data \
  resume=True \
  project=/content/output/runs/classify \
  name=yolov8n_sma_cls \
  device=0
```

## Evaluate

```bash
DATA_ROOT=/content/data OUTPUT_ROOT=/content/output SPLIT=test \
bash scripts/evaluate_colab.sh
```

## Export

```bash
python yolo_local.py classify export \
  model=/content/output/runs/classify/yolov8n_sma_cls/weights/best.pt \
  format=onnx \
  imgsz=224 \
  device=0
```

## TensorBoard

```bash
tensorboard --logdir /content/output/runs/classify
```

In a notebook:

```python
%load_ext tensorboard
%tensorboard --logdir /content/output/runs/classify
```

## Weights & Biases

W&B is disabled by default in the Colab scripts because Colab often has `wandb` preinstalled and this older Ultralytics fork uses the filesystem output path as the default W&B project name. TensorBoard remains enabled.

To opt in to W&B, enable it explicitly. The callback sanitizes the Ultralytics output path before passing it to W&B:

```bash
export WANDB_DISABLED=false
```

## Notes

- The normal `yolo` shell command may import the installed pip package and fail to find custom modules. Prefer `python yolo_local.py`.
- Training outputs are written to `OUTPUT_ROOT`, which should be `/content/output` or a Google Drive path if you want persistence across Colab sessions.
- If Colab GPU memory is limited, reduce `BATCH` to `32`, `16`, or `8`.
