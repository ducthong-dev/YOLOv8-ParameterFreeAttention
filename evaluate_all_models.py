import argparse
import csv
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def auto_device():
    return "0" if torch.cuda.is_available() else "cpu"


def find_model_files(models_dir):
    """Recursively find all .pt files in the given directory."""
    return [str(p) for p in Path(models_dir).rglob("*.pt")]


def evaluate_model(model_path, data_root, device="0", imgsz=224, batch=64):
    try:
        model = YOLO(model_path)
        metrics = model.val(data=data_root, device=device, imgsz=imgsz, batch=batch, split="test")
        return {
            "model": model_path,
            "top1": getattr(metrics, "top1", None),
            "top5": getattr(metrics, "top5", None),
        }
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return {"model": model_path, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate all YOLOv8 classification weights in a directory.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=str(Path(os.getenv("OUTPUT_ROOT", "runs")) / "classify"),
        help="Directory containing .pt model files",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.getenv("DATA_ROOT", "dataset"),
        help="Classification dataset root with train/val/test folders",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(Path(os.getenv("OUTPUT_ROOT", "runs")) / "evaluation_report.csv"),
        help="Output CSV report file",
    )
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", auto_device()), help="device: cpu | 0 | 0,1,2,3")
    parser.add_argument("--imgsz", type=int, default=224, help="image size")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    args = parser.parse_args()

    pt_files = find_model_files(args.models_dir)
    print(f"Found {len(pt_files)} model(s) in {args.models_dir}")

    report_rows = []
    for pt_file in pt_files:
        print(f"Evaluating {pt_file} ...")
        report_rows.append(evaluate_model(pt_file, args.data_root, args.device, args.imgsz, args.batch))

    print("\n===== Evaluation Summary =====")
    for row in report_rows:
        print(f"\nModel: {row['model']}")
        for k, v in row.items():
            if k != "model":
                print(f"  {k}: {v}")
    print("\n===== End of Summary =====")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "top1", "top5", "error"]
    with report_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
