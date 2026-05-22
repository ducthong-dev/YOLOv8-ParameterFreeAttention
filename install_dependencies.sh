#!/bin/bash
# Install all required dependencies for YOLOv8-SimAM training on macOS

echo "======================================================================"
echo "Installing YOLOv8-SimAM Dependencies"
echo "======================================================================"
echo ""

# Check if in conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Not in a conda environment!"
    echo "Please activate your conda environment first:"
    echo "  conda activate yolov8"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

echo "Installing core dependencies..."
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing PyTorch and torchvision..."
# For macOS with Apple Silicon
pip install torch torchvision torchaudio

echo ""
echo "Installing OpenCV and image processing..."
pip install opencv-python pillow imageio

echo ""
echo "Installing data and utility packages..."
pip install numpy scipy pandas matplotlib seaborn scikit-learn

echo ""
echo "Installing YOLO dependencies..."
pip install pyyaml tqdm requests psutil py-cpuinfo

echo ""
echo "Installing specialized packages..."
pip install thop timm streamlit

echo ""
echo "======================================================================"
echo "✓ Installation Complete!"
echo "======================================================================"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; import cv2; import numpy; print(\"All OK\")'  "
echo ""
