"""
Training script for YOLOv8-SimAM on macOS with MPS (Apple Silicon GPU)
Optimized for MacBook with Apple Silicon
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-SimAM on macOS")
    parser.add_argument("--model", type=str, 
                       default="ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml",
                       help="Model config path")
    parser.add_argument("--data", type=str, 
                       default="dataset/Plant_leaf_diseases_dataset/data.yaml",
                       help="Data YAML path")
    parser.add_argument("--scale", type=str, default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="Model scale")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32,
                       help="Batch size (32 recommended for macOS)")
    parser.add_argument("--imgsz", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps (Apple GPU) | cpu")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of workers for data loading")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Classification datasets expect a directory with train/val/test subfolders.
    data_path = Path(args.data)
    if data_path.is_file():
        args.data = str(data_path.parent)
    
    # Check MPS availability
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            print("⚠️  MPS not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print("✓ Using Apple Silicon GPU (MPS)")
    
    # Generate experiment name if not provided
    if args.name is None:
        model_base = args.model.split('/')[-1].replace('.yaml', '')
        args.name = f"{model_base}_{args.scale}"
    
    print("\n" + "="*80)
    print(f"Training YOLOv8-SimAM on macOS")
    print("="*80)
    print(f"Model:      {args.model}")
    print(f"Scale:      {args.scale}")
    print(f"Data:       {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Name:       {args.name}")
    print("="*80 + "\n")
    
    # Load model
    model = YOLO(args.model)
    
    # Train with optimized settings for macOS
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/classify',
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=10,
        cache=False,  # Don't cache on macOS to save memory
        workers=args.workers,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        # Training settings
        verbose=True,
        seed=42,
        deterministic=True,
    )
    
    # Validation
    print("\n" + "="*80)
    print("Validating Best Model")
    print("="*80)
    metrics = model.val()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best Model: runs/classify/{args.name}/weights/best.pt")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
