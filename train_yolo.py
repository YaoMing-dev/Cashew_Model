"""
train_yolo.py
-------------
Train YOLOv8 để detect hạt điều (1 class: "hat").

Sau khi train xong, model được export sang TorchScript (.pt) để
dùng trong inference pipeline với độ trễ thấp.

Chạy:
    python train_yolo.py
    python train_yolo.py --model yolov8n.pt  # nhanh hơn, nhỏ hơn
    python train_yolo.py --epochs 50 --batch 8
"""

import argparse
import shutil
from pathlib import Path

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "data.yaml"  # Use Roboflow data.yaml
OUTPUT_MODEL = BASE_DIR / "yolo_cashew.pt"
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 - Cashew Nut Detection")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model: yolov8n.pt (nhanh) hoặc yolov8s.pt (cân bằng)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0",
                        help="GPU device ID hoặc 'cpu'")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def train(args):
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_YAML}\n"
            "Hãy chạy: python prepare_yolo_dataset.py"
        )

    print("=" * 60)
    print("  Train YOLOv8 - Cashew Nut Detection")
    print("=" * 60)
    print(f"  Model base : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Device     : {args.device}")
    print(f"  Data yaml  : {DATA_YAML}")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=True,           # FP16: giảm memory, tăng speed
        workers=args.workers,
        project=str(BASE_DIR / "runs" / "yolo"),
        name="cashew_detect",
        exist_ok=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        # Tối ưu convergence
        patience=20,         # early stopping nếu không cải thiện sau 20 epochs
        save_period=10,      # lưu checkpoint mỗi 10 epochs
        plots=True,
    )

    # Validate
    print("\n[Validate] Đánh giá model trên val set...")
    metrics = model.val()
    print(f"  mAP50   : {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # Export sang TorchScript .pt
    best_weights = BASE_DIR / "runs" / "yolo" / "cashew_detect" / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback: tìm trong runs mới nhất
        runs_dir = BASE_DIR / "runs" / "yolo"
        candidates = sorted(runs_dir.glob("cashew_detect*/weights/best.pt"))
        if candidates:
            best_weights = candidates[-1]

    print(f"\n[Export] TorchScript từ: {best_weights}")
    export_model = YOLO(str(best_weights))
    export_path = export_model.export(
        format="torchscript",
        imgsz=args.imgsz,
        half=True,           # FP16 TorchScript
        optimize=False,      # optimize=True chỉ dành cho mobile
    )

    # Copy sang output path
    shutil.copy2(export_path, OUTPUT_MODEL)
    print(f"\n[OK] Model đã lưu: {OUTPUT_MODEL}")

    print("\n" + "=" * 60)
    print(f"  mAP50   : {metrics.box.map50:.4f}  (target: >0.85)")
    print(f"  Output  : {OUTPUT_MODEL}")
    print(f"  Tiếp theo: python prepare_resnet_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    train(args)
