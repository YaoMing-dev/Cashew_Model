# Cashew Nut Grading — YOLOv8 + ResNet-50

Automated cashew grading system using a two-stage pipeline:

1. **YOLOv8s** — detects and localizes each individual cashew nut in frame
2. **ResNet-50** — classifies the grade of each detected nut

---

## Classes

| Class | Visual characteristic |
|-------|----------------------|
| `loai1` | Pure white, clean surface |
| `loai2` | Yellowish tint |
| `loai3` | Shriveled / undersized |
| `lbw` | Dark spots / scorching on surface |
| `tb` | Skin membrane residue on surface |
| `bad_output` | Broken / physically damaged |

---

## Pipeline

```
[Input frame]
      |
      v
  YOLOv8s detect
  (bounding box per nut)
      |
      v
  Crop each bbox
      |
      v
  ResNet-50 classify
  (grade per nut)
      |
      v
[Output: position + grade for every nut in frame]
```

---

## Why YOLOv8 + ResNet-50

### Common misconceptions

**YOLO is not a classifier.**
YOLO is optimized for localization — it finds *where* objects are. Asking it to also distinguish fine-grained quality grades (color, surface texture, shape deformation) is outside its design intent. Its classification head operates on coarse feature maps not suited for subtle inter-class differences like `loai1` vs `loai2`.

**SAM2 is not a classifier either.**
SAM2 (Segment Anything Model) excels at segmentation — producing precise masks. But segmentation and grading are different problems. SAM2 has no concept of cashew quality; it would need an additional classification head and substantial fine-tuning, making it overengineered for this task.

### Why this combination works

| Requirement | Solution |
|-------------|----------|
| Real-time throughput on embedded hardware | YOLOv8s (small, fast) |
| Fine-grained grade discrimination | ResNet-50 (trained on crops) |
| Limited labeled data | Transfer learning from ImageNet |
| Simple deployment | Two `.pt` files, no server required |

ResNet-50 with transfer learning reaches high accuracy on ~400–2000 samples per class — practical for a factory floor where labeling budget is limited. The full pipeline runs comfortably in real-time on a mid-range GPU.

---

## Dataset and Training Notes

### What the previous approach got wrong

Earlier attempts collected data with **mixed grades in the same frame** and **overlapping nuts** (nuts stacked on top of each other). This creates two hard problems:

1. **Mixed frames**: if a frame contains multiple grades, there is no clean label for the whole image — the model receives contradictory signals during training.
2. **Overlapping / occluded nuts**: neither the camera nor the model can see the full surface of a nut that is partially hidden. Any label assigned to an occluded nut is inherently noisy.

The correct data collection protocol is to stage **one grade per frame** with **nuts separated** (not touching). Train on clean, unambiguous examples first. Harder edge cases (partial occlusion, borderline grades) can be introduced later — but there is a hard ceiling set by physics: if the camera cannot see the relevant surface features, no model can recover that information.

> Clean data first. Harder cases later. Acknowledge the camera's physical limits.

### ResNet training strategy

Two-phase transfer learning:
- **Phase 1** (10 epochs, backbone frozen): only the classification head trains — fast convergence from ImageNet features
- **Phase 2** (30 epochs, full fine-tune): entire network adapts to cashew domain at a lower learning rate

Dataset for ResNet is built by running YOLO on labeled single-grade frames, cropping each detected nut, and using the frame label as the crop label. This keeps training and inference in the same visual domain (same camera, same conveyor, same lighting).

---

## Model Files

| File | Description | Format |
|------|-------------|--------|
| `runs/yolo/cashew_detect/weights/best.pt` | YOLOv8s trained weights | Ultralytics |
| `yolo_cashew.pt` | YOLOv8s TorchScript export | TorchScript |
| `resnet50_cashew.pt` | ResNet-50 for inference | TorchScript |
| `resnet50_cashew_weights.pt` | ResNet-50 state dict (resume training) | PyTorch |

---

## Training

### YOLOv8

Dataset: 391 train / 98 val images (1 class: `cashew`)

```bash
python train_yolo.py
python train_yolo.py --model yolov8s.pt --epochs 100 --batch 16 --device 0
```

### ResNet-50

Dataset: ~9000 crops extracted from single-grade production frames

```bash
python train_resnet.py
python train_resnet.py --epochs_phase1 10 --epochs_phase2 30
```

---

## Test Pipeline

```bash
python test_models.py             # test 5 images
python test_models.py --images 10
```

Requires `valid/images/` directory.

---

## Requirements

```bash
pip install -r requirements.txt
```

```
torch
torchvision
ultralytics
Pillow
tqdm
```
