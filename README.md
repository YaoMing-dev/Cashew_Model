# Cashew Nut Grading — YOLOv8 + ResNet-50

Automated cashew grading system using a two-stage pipeline:

1. **YOLOv8s** — detects and localizes each individual cashew nut in frame
2. **ResNet-50** — classifies the grade of each detected nut

---

## Pipeline Results

Each frame is classified consistently — one grade per image, confidence ~100%.

| loai3 (shriveled) | lbw (surface spots) |
|---|---|
| ![loai3](assets/demo_1.jpg) | ![lbw](assets/demo_2.jpg) |

| loai2 (yellowish) | loai1 (white) |
|---|---|
| ![loai2](assets/demo_3.jpg) | ![loai1](assets/demo_4.jpg) |

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

### Architecture matters

**YOLO** is built around a detection architecture (CSPDarknet backbone + FPN neck + detection head). Its entire design is optimized for answering *"where is the object?"* — predicting bounding boxes and a coarse object category. The feature representations it produces are spatially rich but semantically shallow for fine-grained discrimination. Asking YOLO to distinguish `loai1` (white) from `loai2` (yellow) or `lbw` (surface spots) is asking a detection architecture to do classification work — the two tasks require fundamentally different inductive biases.

**SAM2** is a segmentation architecture. It produces pixel-level masks, which is a different problem entirely. SAM2 carries no concept of cashew quality grades; retrofitting it as a classifier would require adding a classification head and retraining on domain data, at which point you are rebuilding ResNet inside a much heavier model. Overengineered and slower.

**ResNet-50** is a pure classification architecture. Its deep residual blocks learn discriminative feature hierarchies specifically designed to answer *"what is this?"* — exactly the question needed here. Color gradients (loai1 vs loai2), surface texture (lbw spots, tb skin residue), shape deformation (loai3 shriveling) — these are precisely the patterns residual networks are built to capture.

### Why this combination works

The split is clean: YOLO handles space, ResNet handles semantics.

| Stage | Model | Role | Architecture |
|-------|-------|------|-------------|
| Detection | YOLOv8s | Find each nut, output bbox | CSPDarknet + FPN |
| Classification | ResNet-50 | Grade each crop | Deep residual network |

Running both sequentially on a mid-range GPU, the pipeline sustains **50–60 FPS** in real-time — well within factory line throughput requirements. YOLOv8s is lightweight enough that the classification step adds minimal latency on top.

Transfer learning from ImageNet means ResNet-50 reaches strong accuracy with as few as 400–2000 labeled samples per class — realistic for a factory labeling budget.

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
