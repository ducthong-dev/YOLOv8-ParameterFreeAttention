# YOLOv8-SimAM: Plant Leaf Disease Classification

Implementation of SimAM (Simple, Parameter-Free Attention Module) for YOLOv8 classification applied to plant leaf disease detection.

## 📋 Overview

This project enhances YOLOv8 classification with SimAM attention mechanism for improved plant leaf disease recognition. SimAM is a parameter-free attention module that computes 3D attention weights based on spatial energy functions, offering computational efficiency without sacrificing performance.

### Key Features

- ✅ **Parameter-Free**: SimAM adds zero learnable parameters
- ✅ **Lightweight**: Minimal computational overhead (~0.0 GFLOPs)
- ✅ **Plug-and-Play**: Easy integration into existing YOLOv8 architectures
- ✅ **Multiple Variants**: 4 different architectures for comprehensive comparison
- ✅ **Comprehensive Evaluation**: Automated training and comparison tools

## 🏗️ Architecture Variants

### 1. YOLOv8-SimAM (Basic)
**Config**: `yolov8_SimAM_cls.yaml`

SimAM applied after SPPF layer in the head:
```yaml
backbone:
  - ... (standard YOLOv8 backbone)
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, SimAM, []]          # Parameter-free attention
  - [-1, 1, Classify, [nc]]
```

**Use Case**: Balanced performance with minimal overhead

### 2. YOLOv8-SimAM-Backbone
**Config**: `yolov8_SimAM_backbone_cls.yaml`

Multi-scale SimAM after each C2f block:
```yaml
backbone:
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, SimAM, []]          # P2 attention
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, SimAM, []]          # P3 attention
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, SimAM, []]          # P4 attention
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SimAM, []]          # P5 attention
```

**Use Case**: Rich multi-scale feature attention

### 3. YOLOv8-Hybrid (SimAM + ECA)
**Config**: `yolov8_SimAM_ECA_hybrid_cls.yaml`

Combines SimAM (spatial) and ECA (channel) attention:
```yaml
backbone:
  - ... (with SimAM at mid-level)
  
head:
  - [-1, 1, SimAM, []]          # Spatial attention
  - [-1, 1, ECAAttention, [1024]]  # Channel attention
  - [-1, 1, Classify, [nc]]
```

**Use Case**: Complementary spatial-channel attention fusion

### 4. YOLOv8-SimAM-Custom
**Config**: `yolov8_SimAM_custom_cls.yaml`

Custom e_lambda hyperparameter tuning:
```yaml
head:
  - [-1, 1, SimAM, [1e-3]]      # e_lambda = 1e-3 (stronger regularization)
```

**Use Case**: Domain-specific optimization

## 🔧 Implementation Details

### SimAM Module

```python
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Spatial energy function
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activation(y)
```

### Energy Function

SimAM computes attention weights using:

$$E = \frac{1}{M-1} \sum_{i=1}^{M} (t_i - \hat{t})^2$$

where:
- $M = H \times W$ (spatial dimensions)
- $t_i$ is the feature at position $i$
- $\hat{t}$ is the mean feature value
- $\lambda$ is the regularization parameter

## 📊 Experimental Setup

### Dataset
- **Task**: Plant Leaf Disease Classification
- **Classes**: 39 disease categories
- **Image Size**: 224×224
- **Augmentation**: HSV, flip, rotation, translation

### Training Configuration

```python
epochs = 100
batch_size = 32
optimizer = AdamW
lr0 = 0.001
weight_decay = 0.0005
warmup_epochs = 3
```

### Model Scales

All variants support YOLOv8 scales:
- **n**: nano (3.2M params)
- **s**: small (11.2M params)
- **m**: medium (25.9M params)
- **l**: large (43.7M params)
- **x**: extra-large (68.2M params)

## 🚀 Usage

### Installation

```bash
cd /Users/ducthong/Desktop/Research\ 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/

# Install dependencies (if needed)
pip install ultralytics torch torchvision thop pandas seaborn matplotlib
```

### Training

#### Train Single Model

```bash
# YOLOv8-SimAM (nano)
python train_simam.py \
  --model yolov8_SimAM_cls.yaml \
  --scale n \
  --data /path/to/your/data.yaml \
  --epochs 100 \
  --batch 32 \
  --device 0

# YOLOv8-SimAM-Backbone (small)
python train_simam.py \
  --model yolov8_SimAM_backbone_cls.yaml \
  --scale s \
  --data /path/to/your/data.yaml \
  --epochs 100 \
  --batch 32 \
  --device 0

# YOLOv8-Hybrid (medium)
python train_simam.py \
  --model yolov8_SimAM_ECA_hybrid_cls.yaml \
  --scale m \
  --data /path/to/your/data.yaml \
  --epochs 100 \
  --batch 32 \
  --device 0
```

#### Train All Variants (Batch Script)

```bash
#!/bin/bash

DATA_YAML="/path/to/your/data.yaml"
EPOCHS=100
BATCH=32
DEVICE=0
SCALE="n"  # or s, m, l, x

# Baseline
python train_simam.py --model yolov8-cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE

# ECA
python train_simam.py --model yolov8_ECA_cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE

# SimAM variants
python train_simam.py --model yolov8_SimAM_cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE
python train_simam.py --model yolov8_SimAM_backbone_cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE
python train_simam.py --model yolov8_SimAM_ECA_hybrid_cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE
python train_simam.py --model yolov8_SimAM_custom_cls.yaml --scale $SCALE --data $DATA_YAML --epochs $EPOCHS --batch $BATCH --device $DEVICE
```

### Model Comparison

```bash
python compare_models.py \
  --data /path/to/your/data.yaml \
  --imgsz 224 \
  --device 0 \
  --output comparison_results
```

This generates:
- `comparison.csv`: Detailed metrics table
- `comparison.json`: Results in JSON format
- `comparison_plots.png`: Visualization charts

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/classify/yolov8_SimAM_cls_n/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg')

# Get predictions
probs = results[0].probs
top_class = probs.top1
confidence = probs.top1conf
```

## 📈 Evaluation Metrics

The comparison script evaluates:

### Performance Metrics
- **Top-1 Accuracy**: Single best prediction accuracy
- **Top-5 Accuracy**: Top 5 predictions accuracy
- **Precision/Recall/F1**: Per-class and overall

### Computational Metrics
- **Parameters**: Total model parameters
- **GFLOPs**: Computational complexity
- **Model Size (MB)**: Disk size
- **Inference Speed (ms)**: Mean inference time
- **FPS**: Frames per second
- **Efficiency**: Accuracy per GFLOPs ratio

## 🔬 Experimental Protocol

### Phase 1: Baseline Comparison (Week 1)
1. Train YOLOv8-Baseline (all scales: n, s, m)
2. Train YOLOv8-ECA (all scales)
3. Establish baseline metrics

### Phase 2: SimAM Variants (Week 2-3)
1. Train YOLOv8-SimAM basic (all scales)
2. Train YOLOv8-SimAM-Backbone (select scales)
3. Train YOLOv8-Hybrid (select scales)
4. Compare performance vs baselines

### Phase 3: Hyperparameter Tuning (Week 4)
1. Test e_lambda values: [1e-5, 1e-4, 1e-3, 1e-2]
2. Test placement strategies
3. Optimize best-performing variant

### Phase 4: Analysis & Publication (Week 5)
1. Statistical significance testing
2. Confusion matrix analysis
3. Attention visualization
4. Write research paper

## 📊 Expected Results

### Hypothesis
SimAM should provide:
- ✅ Similar or better accuracy than ECA
- ✅ Zero additional parameters (vs ECA's learnable weights)
- ✅ Minimal computational overhead
- ✅ Better feature discrimination for similar disease classes

### Comparison Table Template

| Model | Top-1 Acc | Top-5 Acc | Params (M) | GFLOPs | Speed (ms) | FPS |
|-------|-----------|-----------|------------|--------|------------|-----|
| YOLOv8n-Baseline | ? | ? | 3.2 | 8.9 | ? | ? |
| YOLOv8n-ECA | ? | ? | 3.2+ | 9.0+ | ? | ? |
| YOLOv8n-SimAM | ? | ? | 3.2 | 8.9 | ? | ? |
| YOLOv8n-SimAM-Backbone | ? | ? | 3.2 | 9.1+ | ? | ? |
| YOLOv8n-Hybrid | ? | ? | 3.2+ | 9.2+ | ? | ? |

## 🎯 Research Questions

1. **Q1**: Does SimAM outperform ECA despite being parameter-free?
2. **Q2**: Which placement strategy (head vs backbone vs hybrid) works best?
3. **Q3**: How does e_lambda affect performance on plant disease dataset?
4. **Q4**: What is the trade-off between accuracy and efficiency?
5. **Q5**: Which disease classes benefit most from attention mechanisms?

## 📁 Project Structure

```
YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/
├── ultralytics/
│   ├── cfg/models/v8/
│   │   ├── yolov8-cls.yaml                    # Baseline
│   │   ├── yolov8_ECA_cls.yaml                # ECA variant
│   │   ├── yolov8_SimAM_cls.yaml              # SimAM basic ✨
│   │   ├── yolov8_SimAM_backbone_cls.yaml     # SimAM backbone ✨
│   │   ├── yolov8_SimAM_ECA_hybrid_cls.yaml   # Hybrid ✨
│   │   └── yolov8_SimAM_custom_cls.yaml       # Custom e_lambda ✨
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── conv.py                        # SimAM class added ✨
│   │   │   └── __init__.py                    # SimAM exported ✨
│   │   └── tasks.py                           # SimAM parsing ✨
├── train_simam.py                             # Training script ✨
├── compare_models.py                          # Comparison tool ✨
└── README_SimAM.md                            # This file ✨
```

## 🧪 Validation Tests

### Test Model Loading

```python
from ultralytics import YOLO

# Test each config
configs = [
    'ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml',
    'ultralytics/cfg/models/v8/yolov8_SimAM_backbone_cls.yaml',
    'ultralytics/cfg/models/v8/yolov8_SimAM_ECA_hybrid_cls.yaml',
    'ultralytics/cfg/models/v8/yolov8_SimAM_custom_cls.yaml',
]

for config in configs:
    print(f"Testing: {config}")
    model = YOLO(config)
    print(f"✓ Loaded successfully")
    print(f"  Layers: {len(list(model.model.modules()))}")
    print()
```

### Test Forward Pass

```python
import torch
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml')
dummy_input = torch.randn(1, 3, 224, 224)

output = model.model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: torch.Size([1, 39])")  # 39 classes
```

## 📚 References

1. **SimAM**: Yang, L., Zhang, R. Y., Li, L., & Xie, X. (2021). SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks. ICML 2021.

2. **ECA-Net**: Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. CVPR 2020.

3. **YOLOv8**: Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/

## 🤝 Contributing

For research collaboration or questions:
- Open an issue on GitHub
- Contact: [Your contact information]

## 📄 License

This project inherits the AGPL-3.0 license from Ultralytics YOLOv8.

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@misc{yolov8simam2026,
  title={YOLOv8-SimAM: Parameter-Free Attention for Plant Leaf Disease Classification},
  author={[Your Name]},
  year={2026},
  note={Research implementation based on Ultralytics YOLOv8 and SimAM}
}
```

---

**Status**: ✅ Implementation Complete - Ready for Training & Experimentation

**Next Steps**:
1. Prepare dataset and data.yaml configuration
2. Run baseline experiments
3. Train SimAM variants
4. Compare results and publish findings

Good luck with your research! 🌱🔬
