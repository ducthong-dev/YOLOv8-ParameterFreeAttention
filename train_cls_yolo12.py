#!/usr/bin/env python3
"""
YOLOv12n Classification Training Script
--------------------------------------
Script tương tự `start_train.py` nhưng dành cho bài toán Classification với YOLOv12n.

Yêu cầu:
- ultralytics >= version hỗ trợ YOLOv12 (nếu chưa có hãy nâng cấp: `pip install -U ultralytics`)
- Trọng số pretrained: `yolo12n-cls.pt` (Ultralytics sẽ tự tải nếu có trên hub). Nếu chưa có (chưa release), hãy dùng tạm `yolov8n-cls.pt` với tham số `--model yolov8n-cls.pt`.
- Cấu trúc dữ liệu classification tiêu chuẩn:
    dataset_root/
        train/
            class_a/ img1.jpg ...
            class_b/ ...
        val/   (hoặc valid/)
            class_a/ ...
            class_b/ ...
  Hoặc cung cấp 1 file YAML chỉ rõ đường dẫn `train:` và `val:`.

Ví dụ chạy:
  python train_cls_yolo12.py \
      --data dataset/Plant_leaf_diseases_dataset/cls_yaml_or_root \
      --model yolo12n-cls.pt \
      --epochs 80 --batch 128 --imgsz 224 --device mps

Sau khi train xong sẽ tự động đánh giá (val) và tùy chọn export ONNX.
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import os
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv12n Classification Training")
    p.add_argument(
        "--model",
        type=str,
        default="yolo12n-cls.pt",
        help="Đường dẫn weight .pt hoặc kiến trúc .yaml (vd: yolo12n-cls.yaml)",
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset root folder (with train/val) hoặc đường dẫn YAML",
    )
    p.add_argument("--epochs", type=int, default=50, help="Số epoch train")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--imgsz", type=int, default=224, help="Kích thước ảnh (square)")
    p.add_argument(
        "--device", type=str, default="mps", help="Thiết bị: mps | cpu | 0 | 0,1 ..."
    )
    p.add_argument("--workers", type=int, default=8, help="Số worker load data")
    p.add_argument(
        "--project", type=str, default="runs/classify", help="Thư mục project output"
    )
    p.add_argument("--name", type=str, default="yolo12n_cls", help="Tên run")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    p.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction")
    p.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum / Adam beta1"
    )
    p.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    p.add_argument("--warmup_epochs", type=float, default=3.0, help="Số epoch warmup")
    p.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout cho head classify"
    )
    p.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing")
    p.add_argument(
        "--augment", action="store_true", help="Bật augment mặc định của Ultralytics"
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume training nếu có last.pt"
    )
    p.add_argument("--export", action="store_true", help="Export ONNX sau khi train")
    p.add_argument("--half", action="store_true", help="Dùng FP16 nếu hỗ trợ")
    p.add_argument(
        "--strict-nc",
        action="store_true",
        help="Không tự động chỉnh sửa số lớp (nc) nếu mismatch",
    )
    p.add_argument(
        "--auto_drive",
        action="store_true",
        help="Nếu chạy Colab: tự động lưu kết quả vào Google Drive (/content/drive/MyDrive)",
    )
    p.add_argument(
        "--drive_subdir",
        type=str,
        default="ultralytics_runs/classify",
        help="Thư mục con trong MyDrive để lưu (khi --auto_drive)",
    )
    return p.parse_args()


def check_dataset_path(path_str: str):
    p = Path(path_str)
    if not p.exists():
        sys.exit(f"[ERROR] Dataset path '{p}' không tồn tại.")
    # Nếu là folder thì kiểm tra train/val
    if p.is_dir():
        has_train = (p / "train").exists()
        has_val = (p / "val").exists() or (p / "valid").exists()
        if not (has_train and has_val):
            print(
                "[WARN] Thư mục không thấy train/ và val/ (hoặc valid/). Nếu bạn dùng YAML thì bỏ qua cảnh báo này."
            )
    return p


def try_load_model(model_path: str) -> YOLO:
    """Load model from weight (.pt) or architecture (.yaml)."""
    try:
        return YOLO(model_path)
    except Exception as e:
        if "yolo12" in model_path.lower():
            print(
                f"[WARN] Không load được '{model_path}'. Thử fallback 'yolov8n-cls.pt'. Lỗi gốc: {e}"
            )
            try:
                return YOLO("yolov8n-cls.pt")
            except Exception as ee:
                sys.exit(f"[ERROR] Fallback cũng thất bại: {ee}")
        sys.exit(f"[ERROR] Không load được model: {e}")


def _list_subdirs(path: Path) -> List[str]:
    return sorted([d.name for d in path.iterdir() if d.is_dir()])


def infer_classes_from_folder(root: Path) -> List[str]:
    train_dir = root / "train"
    if not train_dir.exists():
        return []
    return _list_subdirs(train_dir)


def read_dataset_yaml(path: Path):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_dataset_class_info(data_path: Path):
    """Return (class_names list, num_classes) if detectable else ([], None)."""
    if data_path.is_file() and data_path.suffix in {".yaml", ".yml"}:
        cfg = read_dataset_yaml(data_path)
        names = []
        if "names" in cfg:
            if isinstance(cfg["names"], dict):
                names = [
                    v
                    for k, v in sorted(
                        cfg["names"].items(),
                        key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0],
                    )
                ]
            elif isinstance(cfg["names"], list):
                names = cfg["names"]
        # If names empty but train path exists, try to derive
        if not names and "train" in cfg:
            train_path = Path(cfg["train"])
            if train_path.exists():
                names = _list_subdirs(train_path)
        return names, len(names) if names else ([], None)[1]
    # Folder style
    names = infer_classes_from_folder(data_path)
    return names, len(names) if names else ([], None)[1]


def ensure_model_nc(model: YOLO, class_names: List[str], strict: bool = False):
    if not class_names:
        print("[INFO] Không suy ra được class names để kiểm tra nc.")
        return
    num_classes = len(class_names)
    # Access classify head
    try:
        head = model.model[-1]
        current_nc = getattr(head, "nc", None)
        if current_nc != num_classes:
            msg = f"[INFO] Mismatch nc (model={current_nc}) vs dataset={num_classes}."
            if strict:
                print(msg + " Giữ nguyên do --strict-nc.")
                return
            print(msg + " Tiến hành cập nhật head.")
            setattr(head, "nc", num_classes)
            # Replace linear classifier if exists
            if hasattr(head, "linear"):
                import torch.nn as nn

                in_f = head.linear.in_features
                head.linear = nn.Linear(in_f, num_classes, bias=True)
            model.names = {i: n for i, n in enumerate(class_names)}
    except Exception as e:
        print(f"[WARN] Không thể cập nhật nc tự động: {e}")


def main():
    args = parse_args()
    data_path = check_dataset_path(args.data)

    # Redirect project folder to Google Drive if requested and environment looks like Colab
    if args.auto_drive:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            drive_project = drive_root / args.drive_subdir
            try:
                drive_project.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] Sử dụng Google Drive project path: {drive_project}")
                args.project = str(drive_project)
            except Exception as e:
                print(f"[WARN] Không tạo được thư mục trên Drive: {e}. Dùng project local: {args.project}")
        else:
            print(
                "[WARN] Không thấy /content/drive/MyDrive. Bạn cần chạy: from google.colab import drive; drive.mount('/content/drive')"
            )

    print("=========== YOLOv12n Classification Training ===========")
    print(f"Model weights : {args.model}")
    print(f"Data          : {data_path}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch         : {args.batch}")
    print(f"Image Size    : {args.imgsz}")
    print(f"Device        : {args.device}")
    print(f"Project/Name  : {args.project}/{args.name}")

    model = try_load_model(args.model)

    # Auto adjust nc if architecture YAML or user wants alignment
    class_names, _ = get_dataset_class_info(data_path)
    if args.model.endswith((".yaml", ".yml")):
        ensure_model_nc(model, class_names, strict=args.strict_nc)

    # Một số tham số train dành cho classification (Ultralytics sẽ bỏ qua nếu không hợp lệ theo version)
    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        augment=args.augment,
        resume=args.resume,
        half=args.half,
    )

    print("\n[INFO] Train finished. Running validation...")
    metrics = model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)
    print("[INFO] Validation metrics:")
    for k, v in metrics.items():
        try:
            print(f"  {k}: {float(v):.4f}")
        except Exception:
            pass

    if args.export:
        print("[INFO] Exporting ONNX model...")
        onnx_path = model.export(format="onnx")
        print(f"[INFO] Exported ONNX: {onnx_path}")

    print("\nHoàn tất training classification cho YOLOv12n (hoặc fallback).")


if __name__ == "__main__":
    main()
