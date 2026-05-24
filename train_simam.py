"""
Training script for YOLOv8-SimAM classification models
Plant Leaf Disease Classification with SimAM Attention Module
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import os


def auto_device():
    return '0' if torch.cuda.is_available() else 'cpu'


def train_model(model_yaml, data_yaml, model_name, epochs=100, imgsz=224, batch=32, device='0',
                output_root='runs/classify'):
    """
    Train YOLOv8 classification model with specified configuration
    
    Args:
        model_yaml: Path to model configuration YAML
        data_yaml: Path to data configuration YAML
        model_name: Name for saving model checkpoints
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: GPU device ID
    """
    
    # Initialize model
    model = YOLO(model_yaml)
    
    # Print model summary
    print(f"\n{'='*60}")
    print(f"Training Model: {model_name}")
    print(f"Configuration: {model_yaml}")
    print(f"{'='*60}\n")
    
    # Training arguments
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=output_root,
        name=model_name,
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
        seed=42,
        deterministic=True,
    )
    
    # Validation
    print(f"\n{'='*60}")
    print(f"Validating Model: {model_name}")
    print(f"{'='*60}\n")
    
    metrics = model.val()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Completed: {model_name}")
    print(f"Best Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    print(f"{'='*60}\n")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-SimAM Classification Models')
    parser.add_argument('--model', type=str, default='yolov8_SimAM_cls.yaml', 
                        help='Model config: yolov8_SimAM_cls.yaml, yolov8_SimAM_backbone_cls.yaml, yolov8_SimAM_ECA_hybrid_cls.yaml, yolov8_SimAM_custom_cls.yaml')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model scale')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to classification dataset root')
    parser.add_argument('--output-root', type=str, default=os.getenv('OUTPUT_ROOT', 'runs/classify'),
                        help='Output root for classification runs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default=os.getenv('DEVICE', auto_device()),
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Construct paths
    model_path = f"ultralytics/cfg/models/v8/{args.model}"
    model_name = f"{Path(args.model).stem}_{args.scale}"
    
    print(f"\n{'='*60}")
    print(f"YOLOv8-SimAM Plant Leaf Disease Classification")
    print(f"{'='*60}")
    print(f"Model Config: {model_path}")
    print(f"Model Scale: {args.scale}")
    print(f"Data Config: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Output Root: {args.output_root}")
    print(f"{'='*60}\n")
    
    # Train model
    results, metrics = train_model(
        model_yaml=model_path,
        data_yaml=args.data,
        model_name=model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_root=args.output_root
    )


if __name__ == '__main__':
    main()
