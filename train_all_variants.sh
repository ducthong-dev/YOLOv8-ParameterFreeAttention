#!/bin/bash
# Quick training script for YOLOv8-SimAM experiments on macOS
# Usage: bash train_all_variants.sh

echo "======================================================================"
echo "YOLOv8-SimAM Plant Leaf Disease Classification - macOS Training"
echo "======================================================================"
echo ""

# Configuration
DATA_YAML="dataset/Plant_leaf_diseases_dataset/data.yaml"
SCALE="n"  # Start with nano for quick experiments
EPOCHS=100
BATCH=32   # Safe for macOS
DEVICE="mps"  # Apple Silicon GPU

echo "Configuration:"
echo "  Dataset: $DATA_YAML"
echo "  Scale: $SCALE"
echo "  Epochs: $EPOCHS"
echo "  Batch: $BATCH"
echo "  Device: $DEVICE"
echo ""
echo "======================================================================"

# Train SimAM Basic
echo ""
echo "[1/4] Training YOLOv8-SimAM Basic..."
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale $SCALE \
  --data $DATA_YAML \
  --epochs $EPOCHS \
  --batch $BATCH \
  --device $DEVICE \
  --name yolov8_SimAM_basic_${SCALE}

# Train SimAM Backbone
echo ""
echo "[2/4] Training YOLOv8-SimAM Backbone..."
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_backbone_cls.yaml \
  --scale $SCALE \
  --data $DATA_YAML \
  --epochs $EPOCHS \
  --batch $BATCH \
  --device $DEVICE \
  --name yolov8_SimAM_backbone_${SCALE}

# Train Hybrid
echo ""
echo "[3/4] Training YOLOv8-SimAM-ECA Hybrid..."
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_ECA_hybrid_cls.yaml \
  --scale $SCALE \
  --data $DATA_YAML \
  --epochs $EPOCHS \
  --batch $BATCH \
  --device $DEVICE \
  --name yolov8_SimAM_hybrid_${SCALE}

# Train Custom e_lambda
echo ""
echo "[4/4] Training YOLOv8-SimAM Custom..."
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_custom_cls.yaml \
  --scale $SCALE \
  --data $DATA_YAML \
  --epochs $EPOCHS \
  --batch $BATCH \
  --device $DEVICE \
  --name yolov8_SimAM_custom_${SCALE}

echo ""
echo "======================================================================"
echo "All training complete!"
echo "Results saved in: runs/classify/"
echo "======================================================================"
