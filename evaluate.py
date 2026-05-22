import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 Classification Model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data.yaml")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = YOLO(args.model)
    results = model.val(data=args.data_dir, device="mps", imgsz=224, batch=32)
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
