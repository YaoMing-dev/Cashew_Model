"""
test_models.py
--------------
Test pipeline: YOLO detect cashew nuts -> crop bbox -> ResNet classify each nut.

Usage:
    python test_models.py
    python test_models.py --images 5   # test 5 random images
"""

import argparse
import random
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

BASE = Path(__file__).parent

YOLO_MODEL   = BASE / "runs" / "yolo" / "cashew_detect" / "weights" / "best.pt"
RESNET_MODEL = BASE / "resnet50_cashew.pt"
VAL_DIR      = BASE / "valid" / "images"

# ImageFolder sorts alphabetically -> this is the actual class order
CLASS_NAMES = ["bad_output", "lbw", "loai1", "loai2", "loai3", "tb"]

RESNET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_models():
    from ultralytics import YOLO
    print(f"  Loading YOLO  : {YOLO_MODEL.name}")
    yolo = YOLO(str(YOLO_MODEL))
    print(f"  Loading ResNet: {RESNET_MODEL.name}")
    resnet = torch.jit.load(str(RESNET_MODEL), map_location="cpu")
    resnet.eval()
    return yolo, resnet


def classify_crop(resnet, pil_img: Image.Image, box_xyxy) -> tuple[str, float]:
    """Crop bbox from image and classify with ResNet."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    # small padding
    pad = 4
    w, h = pil_img.size
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = pil_img.crop((x1, y1, x2, y2))
    tensor = RESNET_TF(crop).unsqueeze(0)
    with torch.inference_mode():
        logits = resnet(tensor)
    idx = logits.argmax(1).item()
    conf = torch.softmax(logits, dim=1)[0, idx].item()
    return CLASS_NAMES[idx], conf


def test_pipeline(n_images: int = 5):
    if not VAL_DIR.exists():
        print(f"ERROR: val dir not found: {VAL_DIR}")
        return

    yolo, resnet = load_models()

    all_imgs = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))
    if not all_imgs:
        print("No images found in val/images/"); return

    random.seed(42)
    samples = random.sample(all_imgs, min(n_images, len(all_imgs)))

    total_det = 0
    class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}

    print("\n" + "="*70)
    print("  YOLO detect -> ResNet classify  (cashew grading pipeline)")
    print("="*70)

    for img_path in samples:
        pil = Image.open(img_path).convert("RGB")
        results = yolo(str(img_path), verbose=False)
        boxes = results[0].boxes

        print(f"\n  Image : {img_path.name}")
        print(f"  Size  : {pil.size[0]}x{pil.size[1]}  |  Detected: {len(boxes)} cashews")

        if len(boxes) == 0:
            print("  (no detections)")
            continue

        print(f"  {'#':<4} {'YOLO conf':<12} {'Class':<14} {'ResNet conf'}")
        print(f"  {'-'*46}")

        for i, box in enumerate(boxes):
            yolo_conf = box.conf.item()
            xyxy = box.xyxy[0].tolist()
            cls_name, resnet_conf = classify_crop(resnet, pil, xyxy)
            class_counts[cls_name] += 1
            total_det += 1
            print(f"  {i+1:<4} {yolo_conf:.3f}       {cls_name:<14} {resnet_conf:.3f}")

    print("\n" + "="*70)
    print(f"  Total images tested : {len(samples)}")
    print(f"  Total detections    : {total_det}")
    if total_det:
        print(f"  Avg detections/img  : {total_det/len(samples):.1f}")
        print(f"\n  Classification summary:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                pct = cnt / total_det * 100
                print(f"    {cls:<14}: {cnt:4d}  ({pct:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=5, help="Number of images to test")
    args = parser.parse_args()
    test_pipeline(args.images)
