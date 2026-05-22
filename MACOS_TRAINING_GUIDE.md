# 🚀 Quick Start Guide - macOS Training

## ✅ Những gì đã được chuẩn bị

### 1. Data.yaml đã được fixed ✓
- Thêm `nc: 39` field
- Path đúng: `/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/dataset/Plant_leaf_diseases_dataset`
- 39 classes (0-38) từ Apple diseases đến Tomato diseases

### 2. Training scripts đã được optimize cho macOS ✓
- `start_train_simam_mac.py` - Optimized cho Apple Silicon
- `train_all_variants.sh` - Batch training script
- Batch size = 32 (safe cho macOS MPS)
- Cache = False (save memory)
- Workers = 4 (optimal cho macOS)

---

## 🎯 Cách Train

### Option 1: Train Single Model (Recommended để test trước)

```bash
cd "/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8"

# Train YOLOv8-SimAM nano (basic variant)
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale n \
  --data dataset/Plant_leaf_diseases_dataset/data.yaml \
  --epochs 100 \
  --batch 32 \
  --device mps
```

**Expected time**: ~2-4 hours cho 100 epochs trên MacBook Pro M1/M2

### Option 2: Train All Variants (Batch Training)

```bash
cd "/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8"

# Train all 4 SimAM variants
bash train_all_variants.sh
```

**Expected time**: ~8-16 hours total cho 4 models

### Option 3: Quick Test (10 epochs để verify)

```bash
# Test với 10 epochs để verify everything works
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale n \
  --data dataset/Plant_leaf_diseases_dataset/data.yaml \
  --epochs 10 \
  --batch 32 \
  --device mps \
  --name quick_test
```

**Expected time**: ~10-20 phút

---

## 🔧 Customization Options

### Adjust Batch Size (nếu memory issues)
```bash
# Nếu bị out of memory, giảm batch size
python start_train_simam_mac.py ... --batch 16

# Nếu có nhiều RAM, tăng lên
python start_train_simam_mac.py ... --batch 64
```

### Change Model Scale
```bash
# Nano (fastest, 3.2M params)
--scale n

# Small (better accuracy, 11.2M params)
--scale s

# Medium (best accuracy, 25.9M params) - chỉ nên dùng nếu có >=16GB RAM
--scale m
```

### Use CPU instead of MPS
```bash
# Nếu gặp vấn đề với MPS
python start_train_simam_mac.py ... --device cpu
```

---

## 📊 What to Expect

### Training Progress
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/100      1.2G      0.586      0.812      0.734        128        224
...
50/100     1.2G      0.234      0.156      0.189        128        224
...
100/100    1.2G      0.198      0.123      0.145        128        224

Speed: 0.5ms preprocess, 8.2ms inference, 0.3ms loss, 1.1ms postprocess per image
Results saved to runs/classify/yolov8_SimAM_basic_n
```

### Expected Performance (39 classes)
| Model | Top-1 Acc | Top-5 Acc | Speed (MPS) | Memory |
|-------|-----------|-----------|-------------|--------|
| SimAM-n | 92-95% | 98-99% | ~8ms | ~1.5GB |
| SimAM-Backbone-n | 93-96% | 98-99% | ~10ms | ~1.8GB |
| Hybrid-n | 94-97% | 99%+ | ~12ms | ~2.0GB |

---

## 📁 Output Structure

After training, results are in:
```
runs/classify/
├── yolov8_SimAM_basic_n/
│   ├── weights/
│   │   ├── best.pt          # Best model checkpoint
│   │   └── last.pt          # Last epoch checkpoint
│   ├── confusion_matrix.png
│   ├── results.csv
│   └── train_batch*.jpg
├── yolov8_SimAM_backbone_n/
├── yolov8_SimAM_hybrid_n/
└── yolov8_SimAM_custom_n/
```

---

## 🔍 Monitor Training

### Real-time monitoring
Training logs show:
- Loss values (should decrease)
- GPU memory usage
- Speed metrics
- Accuracy improvements

### Check results during training
```bash
# View results CSV
cat runs/classify/yolov8_SimAM_basic_n/results.csv

# Check latest checkpoint
ls -lh runs/classify/yolov8_SimAM_basic_n/weights/
```

---

## ⚠️ Troubleshooting

### Issue 1: MPS Error
```
RuntimeError: MPS backend out of memory
```
**Solution**: Reduce batch size
```bash
python start_train_simam_mac.py ... --batch 16
```

### Issue 2: Slow Training
```
Speed: 0.5ms preprocess, 45.2ms inference...
```
**Solution**: 
1. Check if MPS is actually being used (should see ~8-12ms inference)
2. Close other applications
3. Verify GPU activity in Activity Monitor

### Issue 3: NaN Loss
```
Epoch loss: nan
```
**Solution**: 
1. Check data quality
2. Reduce learning rate (modify script: `lr0=0.0005`)
3. Verify data.yaml paths are correct

---

## 🧪 Verify Everything Works

### Step 1: Test data loading
```python
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml')

# Test on a small batch
results = model.train(
    data='dataset/Plant_leaf_diseases_dataset/data.yaml',
    epochs=1,
    batch=4,
    imgsz=224,
    device='mps'
)
```

### Step 2: Check MPS
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

Should output:
```
MPS available: True
MPS built: True
```

---

## 📈 Next Steps After Training

### 1. Evaluate Best Model
```python
from ultralytics import YOLO

model = YOLO('runs/classify/yolov8_SimAM_basic_n/weights/best.pt')
metrics = model.val(data='dataset/Plant_leaf_diseases_dataset/data.yaml')

print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")
```

### 2. Test Inference
```python
# Predict on new images
results = model.predict('path/to/test/image.jpg', save=True)
```

### 3. Compare Models
```bash
python compare_models.py \
  --data dataset/Plant_leaf_diseases_dataset/data.yaml \
  --output comparison_results
```

---

## ✅ Pre-Training Checklist

- [x] data.yaml has `nc: 39` field
- [x] data.yaml paths are correct
- [x] Training script optimized for macOS
- [x] Batch size appropriate (32)
- [x] MPS device configured
- [ ] **You**: Run quick test (10 epochs)
- [ ] **You**: Verify training works
- [ ] **You**: Start full training (100 epochs)

---

## 🎯 Recommended Workflow

### Day 1: Quick Test & Verification
```bash
# 1. Quick test (10 epochs, ~15 minutes)
python start_train_simam_mac.py --epochs 10 --name quick_test

# 2. If successful, start first real training
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale n \
  --epochs 100 \
  --name simam_basic_run1
```

### Day 2-3: Train All Variants
```bash
# Let it run overnight
bash train_all_variants.sh
```

### Day 4: Analysis
```bash
# Compare results
python compare_models.py --data dataset/Plant_leaf_diseases_dataset/data.yaml
```

---

## 🚀 Ready to Start!

**Recommended First Command**:
```bash
cd "/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8"

# Quick 10-epoch test
python start_train_simam_mac.py \
  --model ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml \
  --scale n \
  --data dataset/Plant_leaf_diseases_dataset/data.yaml \
  --epochs 10 \
  --batch 32 \
  --device mps \
  --name quick_test

# If successful, you'll see output like:
# ✓ Using Apple Silicon GPU (MPS)
# Epoch 1/10: ...
# ...
# Training Complete!
```

**Good luck with your training!** 🌱🔬🚀
