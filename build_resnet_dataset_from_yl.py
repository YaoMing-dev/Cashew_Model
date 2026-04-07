"""
build_resnet_dataset_from_yl.py
--------------------------------
Dùng YOLO detect + crop từng hạt trong các folder _yl (ảnh thật băng chuyền).
Output: resnet_dataset_v2/train/<class>/ và resnet_dataset_v2/val/<class>/
Val split: 15%
"""
import random
import shutil
from pathlib import Path
from PIL import Image
import torch

BASE = Path(__file__).parent
YOLO_MODEL = BASE / "runs" / "yolo" / "cashew_detect" / "weights" / "best.pt"
OUT_DIR    = BASE / "resnet_dataset_v2"

CLASS_MAP = {
    "loai1":      BASE / "loai1_yl",
    "loai2":      BASE / "loai2_yl",
    "loai3":      BASE / "loai3_yl",
    "tb":         BASE / "tb_yl",
    "lbw":        BASE / "lbw_yl",
    "bad_output": BASE / "badoutput_yl",
}

VAL_RATIO  = 0.15
MIN_CROP   = 48      # minimum crop size (px) — skip partial/edge nuts
YOLO_CONF  = 0.4     # detection confidence threshold

def crop_nuts(yolo, img_path: Path, cls_name: str, out_train: Path, out_val: Path,
              val_ratio: float, counters: dict):
    pil = Image.open(img_path).convert("RGB")
    results = yolo(str(img_path), verbose=False, conf=YOLO_CONF)
    boxes = results[0].boxes

    w, h = pil.size
    crops_saved = 0
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        # small padding
        x1c = max(0, x1 - 4)
        y1c = max(0, y1 - 4)
        x2c = min(w, x2 + 4)
        y2c = min(h, y2 + 4)
        cw, ch = x2c - x1c, y2c - y1c
        if cw < MIN_CROP or ch < MIN_CROP:
            continue  # skip tiny edge crops

        crop = pil.crop((x1c, y1c, x2c, y2c))
        is_val = random.random() < val_ratio
        dest_dir = out_val if is_val else out_train
        dest_dir.mkdir(parents=True, exist_ok=True)

        idx = counters[cls_name]
        counters[cls_name] += 1
        crop.save(dest_dir / f"{cls_name}_{idx:04d}.jpg", quality=92)
        crops_saved += 1

    return crops_saved

def main():
    random.seed(42)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_MODEL))

    counters = {cls: 0 for cls in CLASS_MAP}
    summary  = {}

    for cls_name, folder in CLASS_MAP.items():
        imgs = list(folder.glob("*.png")) + list(folder.glob("*.jpg"))
        if not imgs:
            print(f"  WARNING: no images in {folder}"); continue

        train_dir = OUT_DIR / "train" / cls_name
        val_dir   = OUT_DIR / "val"   / cls_name

        total_crops = 0
        for img_path in imgs:
            n = crop_nuts(yolo, img_path, cls_name, train_dir, val_dir,
                          VAL_RATIO, counters)
            total_crops += n

        train_n = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        val_n   = len(list(val_dir.glob("*.jpg")))   if val_dir.exists()   else 0
        summary[cls_name] = (train_n, val_n)
        print(f"  {cls_name:<14}: {total_crops:4d} crops  (train={train_n}, val={val_n})")

    print("\nDataset saved to:", OUT_DIR)
    print(f"{'Class':<14} {'Train':>7} {'Val':>7}")
    print("-" * 32)
    for cls, (tr, vl) in summary.items():
        print(f"{cls:<14} {tr:>7} {vl:>7}")

if __name__ == "__main__":
    main()
