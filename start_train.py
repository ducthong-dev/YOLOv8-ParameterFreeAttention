import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def auto_device():
    return "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parse = argparse.ArgumentParser(description="Train a YOLO classification model")
    parse.add_argument("--model", type=str, default="ultralytics/cfg/models/v8/yolov8n_sma_cls.yaml", help="model YAML or weights")
    parse.add_argument("--data_dir", type=str, default=os.getenv("DATA_ROOT", "dataset"), help="classification dataset root")
    parse.add_argument("--output_root", type=str, default=os.getenv("OUTPUT_ROOT", "runs"), help="output root directory")
    parse.add_argument("--name", type=str, default="yolov8n_sma_cls", help="run name")
    parse.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parse.add_argument("--imgsz", type=int, default=224, help="image size")
    parse.add_argument("--batch", type=int, default=64, help="batch size")
    parse.add_argument("--device", type=str, default=os.getenv("DEVICE", auto_device()), help="device: cpu | 0 | 0,1,2,3")
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    model = YOLO(args.model)
    project = Path(args.output_root) / "classify"
    model.train(data=args.data_dir, device=args.device, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=str(project), name=args.name, optimizer="AdamW", lr0=0.001)


if __name__ == "__main__":
    main()
