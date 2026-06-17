"""
Training Script — Fine-tune all DR classifiers
===============================================
Trains EfficientNet-B4, RETFound ViT, and DenseNet-121 on your dataset.

Expected dataset layout (APTOS / EyePACS compatible):
  data/
    train/
      images/  *.png
      labels.csv   (image_name, diagnosis)   diagnosis: 0-4
    val/
      images/
      labels.csv
    test/
      images/
      labels.csv

Usage
-----
  python train.py --model efficientnet --data-dir data/ --epochs 30
  python train.py --model retfound     --data-dir data/ --epochs 20
  python train.py --model densenet     --data-dir data/ --epochs 25

  # After training, calibrate temperature on val set:
  python train.py --model efficientnet --calibrate-only --checkpoint weights/efficientnet_best.pth

FIXES vs original:
  - FocalLoss (gamma=2.0) replaces CrossEntropyLoss → handles rare classes like PDR
  - CLAHE preprocessing applied in __getitem__ → better vessel contrast before augmentation
  - Stronger augmentation: RandomAffine, RandomErasing, wider ColorJitter
  - Mixup training in train_epoch → improved generalisation
  - Layer-wise LR decay → pretrained backbone trains 10× slower than classifier head
  - Post-training temperature calibration saved to weights/temperature.json
  - Gradient accumulation support for small batch sizes
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from stages.stage4_classify import (
    EfficientNetDR,
    RETFoundViT,
    DenseNetDR,
    NUM_DR_CLASSES,
    _LESION_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("Train")


# ─────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    Focuses training on hard/misclassified examples — critical for
    rare classes like Proliferative DR where CrossEntropy under-trains.

    gamma=2.0 : standard value; increase to 3.0 for very imbalanced data
    """

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per sample (no reduction)
        ce = F.cross_entropy(
            logits, targets,
            weight          = self.weight,
            label_smoothing = self.label_smoothing,
            reduction       = 'none',
        )
        # Focal weight: (1 - p_t)^gamma
        pt     = torch.exp(-ce)
        loss   = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────
#  CLAHE preprocessing (fundus-specific)
# ─────────────────────────────────────────────

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Apply CLAHE on the LAB L-channel + green-channel boost.
    Applied in __getitem__ so augmentation acts on enhanced images.
    Green blend weight 0.70 matches stage2_preprocess.py.
    """
    img = np.array(pil_img.convert("RGB"))

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green    = img[:, :, 1]
    green_eq = clahe.apply(green)

    lab          = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b      = cv2.split(lab)
    l_eq         = clahe.apply(l)
    lab_eq       = cv2.merge([l_eq, a, b])
    rgb_eq       = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    result = rgb_eq.copy().astype(np.float32)
    result[:, :, 1] = 0.30 * rgb_eq[:, :, 1] + 0.70 * green_eq
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class FundusDataset(Dataset):
    """
    CSV columns required: image_name, diagnosis (0-4)
    Optional lesion_* columns for DenseNet multi-label training.
    """

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),                          # increased from 15
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,                       # increased from 0.2
            saturation=0.2, hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),                               # new: slight spatial shifts
            scale=(0.9, 1.1),
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),    # new: occlusion robustness
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, csv_path: str, img_dir: str, split: str = "train", use_clahe: bool = True):
        self.df          = pd.read_csv(csv_path)
        self.img_dir     = Path(img_dir)
        self.split       = split
        self.use_clahe   = use_clahe
        self.tfm         = self.TRAIN_TRANSFORM if split == "train" else self.VAL_TRANSFORM
        self.lesion_cols = [c for c in self.df.columns if c.startswith("lesion_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        name = row["image_name"]

        # Support .png, .jpg, .jpeg with or without extension
        img_path = None
        for ext in ["", ".png", ".jpg", ".jpeg"]:
            p = self.img_dir / (str(name) + ext)
            if p.exists():
                img_path = p
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for '{name}' in {self.img_dir}")

        img = Image.open(img_path).convert("RGB")

        # CLAHE applied before augmentation transforms
        if self.use_clahe:
            img = apply_clahe(img)

        img_t = self.tfm(img)
        grade = int(row["diagnosis"])

        lesions = torch.zeros(len(_LESION_LABELS))
        for i, col in enumerate(self.lesion_cols):
            if col in row.index:
                lesions[i] = float(row[col])

        return img_t, grade, lesions


# ─────────────────────────────────────────────
#  Class-balanced sampler
# ─────────────────────────────────────────────

def make_weighted_sampler(dataset: FundusDataset) -> WeightedRandomSampler:
    grades  = dataset.df["diagnosis"].values
    counts  = np.bincount(grades, minlength=NUM_DR_CLASSES).astype(float)
    weights = 1.0 / np.maximum(counts, 1)
    logger.info(f"Class counts: {counts.astype(int).tolist()} → sample weights: {np.round(weights, 4).tolist()}")
    sample_weights = torch.tensor([weights[g] for g in grades], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ─────────────────────────────────────────────
#  Mixup augmentation
# ─────────────────────────────────────────────

def mixup_batch(imgs: torch.Tensor, grades: torch.Tensor, alpha: float = 0.4):
    """
    Mixup: linearly interpolate two random samples and mix their labels.
    Returns: mixed_imgs, labels_a, labels_b, lambda
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1.0 - lam) * imgs[idx]
    return mixed, grades, grades[idx], lam


# ─────────────────────────────────────────────
#  Layer-wise LR
# ─────────────────────────────────────────────

def get_layerwise_optimizer(model, model_name: str, base_lr: float = 1e-4, weight_decay: float = 1e-4):
    """
    Pretrained backbone trains at base_lr × 0.1 (10× slower).
    Classifier head trains at base_lr.
    This prevents destroying learned ImageNet/retinal features early in training.
    """
    try:
        if model_name == "efficientnet":
            backbone_params = list(model.backbone.parameters())
            head_params     = list(model.classifier.parameters())
        elif model_name == "retfound":
            backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
            head_params     = [p for n, p in model.named_parameters() if "head"     in n]
        else:  # densenet
            backbone_params = list(model.backbone.parameters())
            head_params     = list(model.classifier.parameters()) + list(model.lesion_head.parameters())
    except AttributeError:
        # Fallback: last 10 param groups = head, rest = backbone
        all_params      = list(model.parameters())
        backbone_params = all_params[:-10]
        head_params     = all_params[-10:]
        logger.warning("Could not find named backbone/classifier attributes; using parameter index split.")

    return AdamW([
        {"params": backbone_params, "lr": base_lr * 0.1},
        {"params": head_params,     "lr": base_lr},
    ], weight_decay=weight_decay)


# ─────────────────────────────────────────────
#  Temperature calibration
# ─────────────────────────────────────────────

def calibrate_temperature(model, val_loader: DataLoader, device: torch.device,
                           save_path: str = "weights/temperature.json",
                           model_name: str = "model"):
    """
    Fit a single temperature scalar T on the validation set.
    Minimises NLL of (logits / T) vs true labels using LBFGS.
    Saves T to JSON for use by pipeline.TemperatureScaler.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for imgs, grades, _ in val_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            all_logits.append(logits.cpu())
            all_labels.append(grades)

    all_logits = torch.cat(all_logits).to(device)
    all_labels = torch.cat(all_labels).to(device)

    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    nll         = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        loss = nll(all_logits / temperature.clamp(min=0.05), all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = float(temperature.item())
    logger.info(f"Temperature calibrated: T={T:.4f} for {model_name}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({"temperature": T, "model": model_name}, f, indent=2)
    logger.info(f"Saved to {save_path}")
    return T


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, model, device, model_name: str, out_dir: str,
                 lr: float = 1e-4, l2: float = 1e-4,
                 grad_accum_steps: int = 1, use_mixup: bool = True):
        self.model              = model.to(device)
        self.device             = device
        self.model_name         = model_name
        self.out_dir            = Path(out_dir)
        self.grad_accum_steps   = grad_accum_steps
        self.use_mixup          = use_mixup
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.opt = get_layerwise_optimizer(model, model_name, base_lr=lr, weight_decay=l2)

        # FocalLoss replaces CrossEntropyLoss — better for class-imbalanced DR data
        self.grade_loss  = FocalLoss(gamma=2.0, label_smoothing=0.1)
        self.lesion_loss = nn.BCEWithLogitsLoss()

        self.best_val_acc = 0.0

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        self.opt.zero_grad()

        for step, (imgs, grades, lesions) in enumerate(loader):
            imgs    = imgs.to(self.device)
            grades  = grades.to(self.device)
            lesions = lesions.to(self.device)

            # Mixup augmentation
            if self.use_mixup:
                imgs_in, g_a, g_b, lam = mixup_batch(imgs, grades)
            else:
                imgs_in, g_a, g_b, lam = imgs, grades, grades, 1.0

            if self.model_name == "densenet":
                grade_logits, lesion_logits = self.model(imgs_in)
                grade_loss = (
                    lam * self.grade_loss(grade_logits, g_a)
                    + (1 - lam) * self.grade_loss(grade_logits, g_b)
                )
                loss   = grade_loss * 0.7 + self.lesion_loss(lesion_logits, lesions) * 0.3
                preds  = grade_logits.argmax(dim=1)
            else:
                logits = self.model(imgs_in)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss   = (
                    lam * self.grade_loss(logits, g_a)
                    + (1 - lam) * self.grade_loss(logits, g_b)
                )
                # For accuracy tracking use original (non-mixed) grades
                with torch.no_grad():
                    orig_logits = self.model(imgs)
                    if isinstance(orig_logits, tuple):
                        orig_logits = orig_logits[0]
                    preds = orig_logits.argmax(dim=1)

            # Gradient accumulation
            (loss / self.grad_accum_steps).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == grades).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, grades, lesions in loader:
            imgs   = imgs.to(self.device)
            grades = grades.to(self.device)

            if self.model_name == "densenet":
                grade_logits, _ = self.model(imgs)
                loss  = self.grade_loss(grade_logits, grades)
                preds = grade_logits.argmax(dim=1)
            else:
                logits = self.model(imgs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss  = self.grade_loss(logits, grades)
                preds = logits.argmax(dim=1)

            total_loss += loss.item() * imgs.size(0)
            correct    += (preds == grades).sum().item()
            total      += imgs.size(0)

        return total_loss / total, correct / total

    def save_checkpoint(self, epoch: int, val_acc: float):
        path = self.out_dir / f"{self.model_name}_best.pth"
        torch.save({
            "epoch":   epoch,
            "model":   self.model.state_dict(),
            "val_acc": val_acc,
        }, path)
        logger.info(f"  ✓ Checkpoint saved → {path}  (val_acc={val_acc:.4f})")


# ─────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────

def build_model(name: str, retfound_weights=None):
    if name == "efficientnet":
        return EfficientNetDR(pretrained=True)
    elif name == "retfound":
        return RETFoundViT(weights_path=retfound_weights)
    elif name == "densenet":
        return DenseNetDR(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["efficientnet", "retfound", "densenet"], required=True)
    parser.add_argument("--data-dir",    default="data",    help="Root dir with train/val subdirs")
    parser.add_argument("--output-dir",  default="weights", help="Where to save checkpoints")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--no-mixup",    action="store_true", help="Disable Mixup training")
    parser.add_argument("--no-clahe",    action="store_true", help="Disable CLAHE preprocessing")
    parser.add_argument("--retfound-pretrain", default=None,
                        help="Path to RETFound MAE checkpoint (only for --model retfound)")

    # Calibration
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Skip training; load checkpoint and calibrate temperature on val set")
    parser.add_argument("--checkpoint",     default=None,
                        help="Checkpoint path for --calibrate-only")
    parser.add_argument("--temperature-out", default=None,
                        help="Where to save temperature JSON (default: <output-dir>/temperature.json)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {args.model}")

    data_dir = Path(args.data_dir)
    use_clahe = not args.no_clahe

    # ── Datasets ──────────────────────────────
    val_ds = FundusDataset(
        data_dir / "val/labels.csv",
        data_dir / "val/images",
        split     = "val",
        use_clahe = use_clahe,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────
    model = build_model(args.model, retfound_weights=args.retfound_pretrain)

    # ── Calibrate-only mode ───────────────────
    if args.calibrate_only:
        if not args.checkpoint:
            parser.error("--checkpoint is required with --calibrate-only")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        temp_out = args.temperature_out or str(Path(args.output_dir) / "temperature.json")
        calibrate_temperature(model, val_loader, device, save_path=temp_out, model_name=args.model)
        return

    # ── Training mode ─────────────────────────
    train_ds = FundusDataset(
        data_dir / "train/labels.csv",
        data_dir / "train/images",
        split     = "train",
        use_clahe = use_clahe,
    )
    train_sampler = make_weighted_sampler(train_ds)
    train_loader  = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    logger.info(f"CLAHE: {'on' if use_clahe else 'off'} | Mixup: {'off' if args.no_mixup else 'on'} | "
                f"Grad accum: {args.grad_accum}")

    trainer = Trainer(
        model,
        device,
        args.model,
        args.output_dir,
        lr              = args.lr,
        grad_accum_steps= args.grad_accum,
        use_mixup       = not args.no_mixup,
    )
    scheduler = CosineAnnealingLR(trainer.opt, T_max=args.epochs, eta_min=1e-6)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss,   val_acc   = trainer.val_epoch(val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"| train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val loss={val_loss:.4f} acc={val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_acc > trainer.best_val_acc:
            trainer.best_val_acc = val_acc
            trainer.save_checkpoint(epoch, val_acc)

    logger.info(f"Training complete. Best val acc: {trainer.best_val_acc:.4f}")

    # Auto-calibrate temperature after training
    logger.info("Running temperature calibration on val set …")
    best_ckpt = Path(args.output_dir) / f"{args.model}_best.pth"
    ckpt      = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    temp_out  = args.temperature_out or str(Path(args.output_dir) / "temperature.json")
    calibrate_temperature(model, val_loader, device, save_path=temp_out, model_name=args.model)


if __name__ == "__main__":
    main()                                   