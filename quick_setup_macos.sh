#!/bin/bash
# Quick setup for macOS - Install only essential packages

echo "======================================================================"
echo "Installing Essential Packages for YOLOv8-SimAM on macOS"
echo "======================================================================"
echo ""

echo "✓ Using Python: $(which python)"
echo "✓ Using pip: $(python -m pip --version)"
echo ""

echo "Installing packages..."
python -m pip install --upgrade pip setuptools wheel --quiet

# Install in smaller batches to avoid interruptions
echo "1. Installing NumPy & SciPy..."
python -m pip install numpy scipy --quiet

echo "2. Installing image processing..."
python -m pip install opencv-python Pillow --quiet

echo "3. Installing PyTorch (this may take a while)..."
python -m pip install torch torchvision torchaudio --quiet

echo "4. Installing data & plotting..."
python -m pip install pandas matplotlib seaborn --quiet

echo "5. Installing YOLO utilities..."
python -m pip install pyyaml tqdm requests psutil py-cpuinfo --quiet

echo "6. Installing specialized packages..."
python -m pip install thop timm --quiet

echo ""
echo "======================================================================"
echo "✓ Installation Complete!"
echo "======================================================================"
echo ""

# Verify
echo "Verifying installation..."
python -c "
import cv2
import torch
import numpy as np
import pandas as pd
print('✓ All critical packages installed successfully!')
print('')
print('Versions:')
print(f'  cv2: {cv2.__version__}')
print(f'  torch: {torch.__version__}')
print(f'  numpy: {np.__version__}')
print(f'  pandas: {pd.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
"

echo ""
echo "Ready to train! Run:"
echo "  python start_train_simam_mac.py --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml --scale n --data dataset/Plant_leaf_diseases_dataset/data.yaml --epochs 10 --batch 32 --device mps --name quick_test"
