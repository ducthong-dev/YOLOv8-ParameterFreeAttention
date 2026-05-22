import argparse
import os
import csv
from ultralytics import YOLO


def find_model_files(models_dir):
    """Recursively find all .pt files in the given directory."""
    pt_files = []
    for root, _, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(os.path.join(root, file))
    return pt_files


def evaluate_model(model_path, data_yaml, device="mps", imgsz=224, batch=128):
    try:
        model = YOLO(model_path)
        metrics = model.val(
            data=data_yaml, device=device, imgsz=imgsz, batch=batch, split="test"
        )
        row = {"model": model_path}
        # Directly access top1 and top5
        top1 = getattr(metrics, "top1", None)
        top5 = getattr(metrics, "top5", None)
        row["top1"] = top1
        row["top5"] = top5
        # Optionally, add more metrics if needed
        return row
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return {"model": model_path, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all YOLOv8 models in a directory."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="Fracture_Detection_Improved_YOLOv8/models/drive-download-20250529T055242Z-1-001",
        help="Directory containing .pt model files",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/dataset/Plant_leaf_diseases_dataset/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="/Users/ducthong/Desktop/Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/report/hardest_with_albumentation_evaluation_report.csv",
        help="Output CSV report file",
    )
    args = parser.parse_args()

    pt_files = find_model_files(args.models_dir)
    print(f"Found {len(pt_files)} model(s) in {args.models_dir}")

    report_rows = []
    for pt_file in pt_files:
        print(f"Evaluating {pt_file} ...")
        result = evaluate_model(pt_file, args.data_yaml)
        report_rows.append(result)

    print("\n===== Evaluation Summary =====")
    for row in report_rows:
        print(f"\nModel: {row['model']}")
        for k, v in row.items():
            if k != "model":
                print(f"  {k}: {v}")
    print("\n===== End of Summary =====")

    # Save to CSV
    fieldnames = ["model", "top1", "top5", "error"]
    with open(args.report, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
