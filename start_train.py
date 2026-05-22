import argparse
from ultralytics import YOLO


def parse_args():
    parse = argparse.ArgumentParser(description="Data Postprocess")
    parse.add_argument("--model", type=str, default=None, help="load the model")
    parse.add_argument("--data_dir", type=str, default=None, help="the dir to data")
    parse.add_argument("--device", type=str, default="mps", help="device: cpu | mps | 0 | 0,1,2,3")
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data_dir, device=args.device, epochs=50, imgsz=224, batch=128)


if __name__ == "__main__":
    main()
