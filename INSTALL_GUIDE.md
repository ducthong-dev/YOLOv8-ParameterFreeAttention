# 🔧 Installation Guide - YOLOv8-SimAM Dependencies

## ⚠️ Issue: Missing `cv2` (OpenCV)

Lỗi `ModuleNotFoundError: No module named 'cv2'` có nghĩa là opencv-python chưa được cài trong conda environment `yolov8`.

## ✅ Solution: Install Missing Dependencies

### Option 1: Quick Install (Recommended)

Chạy script install tự động:

```bash
cd "/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8"

bash install_dependencies.sh
```

**Time**: ~5-10 phút tùy theo connection speed

### Option 2: Manual Install (Specific packages)

Nếu script không hoạt động, cài individual packages:

```bash
# Activate conda environment
conda activate yolov8

# Install OpenCV (critical)
pip install opencv-python

# Install other image processing libraries
pip install pillow imageio

# Install PyTorch (nếu chưa có)
pip install torch torchvision torchaudio

# Install data & utils
pip install numpy scipy pandas matplotlib seaborn

# Install YOLO dependencies
pip install pyyaml tqdm requests psutil

# Install specialized packages
pip install thop timm
```

### Option 3: Install from requirements.txt

```bash
conda activate yolov8
pip install -r requirements.txt
```

## 📋 Essential Packages Checklist

Verify bằng cách chạy:

```bash
python -c "
import cv2
import torch
import numpy as np
import pandas as pd
print('✓ cv2 (OpenCV) OK')
print('✓ torch OK')
print('✓ numpy OK')
print('✓ pandas OK')
print('All dependencies installed correctly!')
"
```

Nếu không error, bạn có thể train!

## 🚀 Sau khi cài dependencies

Chạy quick test:

```bash
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale n \
  --data dataset/Plant_leaf_diseases_dataset/data.yaml \
  --epochs 10 \
  --batch 32 \
  --device mps \
  --name quick_test
```

## ❓ Troubleshooting

### cv2 still not found after pip install

```bash
# Try with opencv-python-headless instead
pip uninstall opencv-python
pip install opencv-python-headless
```

### PyTorch issues on macOS

```bash
# For Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Permission denied errors

```bash
# Install with user flag
pip install --user opencv-python
```

## ✅ Expected Output After Installation

```
Collecting opencv-python
  Using cached opencv_python-4.8.1.78-cp39-cp39-macosx_11_6_arm64.whl
Installing collected packages: opencv-python
Successfully installed opencv-python-4.8.1.78
```

Then you should see no errors when running `python start_train_simam_mac.py ...`

---

**Next**: Run the installation script above, then try training again!
