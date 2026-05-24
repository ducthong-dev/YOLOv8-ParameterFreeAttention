import argparse
import os

import torch
from ultralytics import YOLO


def auto_device():
    return "0" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Classification Model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_ROOT", "dataset"), help="classification dataset root")
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", auto_device()), help="device: cpu | 0 | 0,1,2,3")
    parser.add_argument("--imgsz", type=int, default=224, help="image size")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="dataset split")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.val(data=args.data_dir, device=args.device, imgsz=args.imgsz, batch=args.batch, split=args.split)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
