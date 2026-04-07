"""
train_resnet.py
---------------
Train ResNet-50 để phân loại hạt điều (6 classes).

Chiến lược Transfer Learning:
  - Phase 1 (epoch 1-10):  Freeze backbone, chỉ train fc layer (lr=1e-3)
  - Phase 2 (epoch 11-30): Unfreeze toàn bộ, fine-tune (lr=1e-4)

Output:
  - resnet50_cashew.pt  (TorchScript format, dùng cho inference)
  - resnet50_cashew_weights.pt  (state_dict, dùng để resume training)

Chạy:
    python train_resnet.py
    python train_resnet.py --epochs_phase1 15 --epochs_phase2 40
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ── Cấu hình ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "resnet_dataset_v2"
OUTPUT_SCRIPT = BASE_DIR / "resnet50_cashew.pt"        # TorchScript
OUTPUT_WEIGHTS = BASE_DIR / "resnet50_cashew_weights.pt"  # state_dict

NUM_CLASSES = 6
CLASS_NAMES = ["tb", "loai1", "loai2", "loai3", "lbw", "bad_output"]
INPUT_SIZE = 224
BATCH_SIZE = 32

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 - Cashew Classification")
    parser.add_argument("--epochs_phase1", type=int, default=10,
                        help="Epochs freeze backbone (chỉ train fc)")
    parser.add_argument("--epochs_phase2", type=int, default=30,
                        help="Epochs fine-tune toàn bộ mạng")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = dùng main process (khuyến nghị trên Windows)")
    parser.add_argument("--device", default=None,
                        help="'cuda' hoặc 'cpu' (mặc định: tự động)")
    return parser.parse_args()


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """Khởi tạo ResNet-50 với pretrained ImageNet weights."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Thay fc layer cuối
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(device)


def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    """Tính class weights để xử lý imbalanced dataset."""
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    weights = [total / (len(class_counts) * c) if c > 0 else 1.0 for c in class_counts]
    print("  Class weights:")
    for name, count, w in zip(dataset.classes, class_counts, weights):
        print(f"    {name:15s}: {count:4d} ảnh, weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def freeze_backbone(model: nn.Module):
    """Freeze tất cả layers trừ fc."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze toàn bộ model."""
    for param in model.parameters():
        param.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_train: bool,
    scaler=None,
) -> tuple[float, float]:
    """Chạy 1 epoch, trả về (loss, accuracy)."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train and scaler is not None:
                with torch.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_phase(
    phase: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epochs: int,
    device: torch.device,
) -> float:
    """Train một phase, trả về best val accuracy."""
    best_acc = 0.0
    best_state = None

    print(f"\n{'─'*60}")
    print(f"  Phase {phase}: {epochs} epochs")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, scaler=scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            is_train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        marker = " ← best" if val_acc > best_acc else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Khôi phục best weights
    if best_state:
        model.load_state_dict(best_state)
    return best_acc


def export_torchscript(model: nn.Module, device: torch.device, output_path: Path):
    """Export model sang TorchScript .pt."""
    model.eval()
    # Tạo dummy input để trace
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    with torch.inference_mode():
        scripted = torch.jit.trace(model, dummy)
    scripted.save(str(output_path))
    print(f"  TorchScript saved: {output_path}")


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATASET_DIR}\n"
            "Hãy chạy: python prepare_resnet_dataset.py"
        )

    # Transforms & Datasets
    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(str(DATASET_DIR / "train"), transform=train_tf)
    val_dataset   = datasets.ImageFolder(str(DATASET_DIR / "val"),   transform=val_tf)

    print(f"\n  Train: {len(train_dataset)} ảnh | Val: {len(val_dataset)} ảnh")
    print(f"  Classes: {train_dataset.classes}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    model = build_model(NUM_CLASSES, device)

    # Class weights
    print("\n  Tính class weights...")
    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Mixed precision scaler (chỉ trên CUDA)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Phase 1: Freeze backbone ──────────────────────────────────────────────
    freeze_backbone(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Phase 1 - Trainable params: {trainable_params:,} (fc only)")

    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.epochs_phase1
    )

    best_acc_p1 = train_phase(
        1, model, train_loader, val_loader, criterion,
        optimizer1, scheduler1, scaler, args.epochs_phase1, device
    )

    # ── Phase 2: Unfreeze all ─────────────────────────────────────────────────
    unfreeze_all(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Phase 2 - Trainable params: {trainable_params:,} (full network)")

    optimizer2 = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.epochs_phase2
    )

    best_acc_p2 = train_phase(
        2, model, train_loader, val_loader, criterion,
        optimizer2, scheduler2, scaler, args.epochs_phase2, device
    )

    best_acc = max(best_acc_p1, best_acc_p2)

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n[Export]")

    # 1. TorchScript (dùng cho pipeline)
    export_torchscript(model, device, OUTPUT_SCRIPT)

    # 2. State dict (dùng để resume training)
    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"  State dict saved: {OUTPUT_WEIGHTS}")

    print("\n" + "=" * 60)
    print(f"  Best Val Accuracy: {best_acc:.4f}  (target: >0.90)")
    print(f"  TorchScript     : {OUTPUT_SCRIPT}")
    print(f"  Tiếp theo       : python inference_pipeline.py --image test.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()
