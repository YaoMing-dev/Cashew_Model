"""
Generate demo images: YOLO bbox + ResNet label, saved to assets/
"""
import random
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent
YOLO_MODEL   = BASE / "runs" / "yolo" / "cashew_detect" / "weights" / "best.pt"
RESNET_MODEL = BASE / "resnet50_cashew.pt"
VAL_DIR      = BASE / "valid" / "images"
OUT_DIR      = BASE / "assets"
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["bad_output", "lbw", "loai1", "loai2", "loai3", "tb"]
CLASS_COLORS = {
    "tb":         (52,  199, 89),
    "loai1":      (0,   122, 255),
    "loai2":      (90,  200, 250),
    "loai3":      (255, 214, 10),
    "lbw":        (175, 82,  222),
    "bad_output": (255, 59,  48),
}

RESNET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify(resnet, pil, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    pad = 4
    w, h = pil.size
    crop = pil.crop((max(0,x1-pad), max(0,y1-pad), min(w,x2+pad), min(h,y2+pad)))
    t = RESNET_TF(crop).unsqueeze(0)
    with torch.inference_mode():
        logits = resnet(t)
    idx = logits.argmax(1).item()
    conf = torch.softmax(logits, dim=1)[0, idx].item()
    return CLASS_NAMES[idx], conf

def draw_results(pil, boxes, resnet):
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_sm = font

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_name, cls_conf = classify(resnet, pil, xyxy)
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))

        # bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # label background + text
        label = f"{cls_name} {cls_conf:.0%}"
        bbox_text = draw.textbbox((x1, y1 - 26), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 26), label, fill="white", font=font)

    return pil

def main():
    from ultralytics import YOLO
    yolo   = YOLO(str(YOLO_MODEL))
    resnet = torch.jit.load(str(RESNET_MODEL), map_location="cpu")
    resnet.eval()

    all_imgs = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png"))
    random.seed(7)
    samples = random.sample(all_imgs, min(4, len(all_imgs)))

    saved = []
    for i, img_path in enumerate(samples, 1):
        pil = Image.open(img_path).convert("RGB")
        results = yolo(str(img_path), verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            continue
        annotated = draw_results(pil.copy(), boxes, resnet)
        # resize for web (max width 1280)
        w, h = annotated.size
        if w > 1280:
            annotated = annotated.resize((1280, int(h * 1280 / w)), Image.LANCZOS)
        out_path = OUT_DIR / f"demo_{i}.jpg"
        annotated.save(out_path, quality=88)
        saved.append(out_path)
        print(f"  Saved: {out_path.name}  ({len(boxes)} detections)")

    print(f"\nDone. {len(saved)} images saved to assets/")

if __name__ == "__main__":
    main()
