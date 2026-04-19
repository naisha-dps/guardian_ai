"""
Guardian AI - YOLOv8-Nano Training Script
==========================================
Full training pipeline with:
- YOLOv8-nano model
- SGD optimizer (lr=0.01, momentum=0.937)
- CIoU + BCE + DFL losses
- Custom callbacks and logging
- Checkpointing
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# YOLOv8 is from ultralytics package
# pip install ultralytics
from ultralytics import YOLO
from ultralytics.utils.callbacks.base import add_integration_callbacks


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "model": "yolov8n.pt",          # Nano variant (smallest, edge-optimized)
    "data": "guardian_dataset.yaml",

    # Training params
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,                    # Reduce to 8 if OOM on edge training machine
    "workers": 4,

    # Optimizer (SGD as specified)
    "optimizer": "SGD",
    "lr0": 0.01,                    # Initial learning rate
    "lrf": 0.01,                    # Final LR = lr0 * lrf
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,

    # Augmentation (built-in YOLOv8 + extras)
    "mosaic": 1.0,                  # Mosaic augmentation probability
    "flipud": 0.0,
    "fliplr": 0.5,                  # Horizontal flip 50%
    "hsv_h": 0.015,                 # Hue jitter
    "hsv_s": 0.7,                   # Saturation jitter
    "hsv_v": 0.4,                   # Value/brightness jitter
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,

    # Output
    "project": "runs/train",
    "name": "guardian_wildlife",
    "save": True,
    "save_period": 10,              # Save checkpoint every 10 epochs
    "exist_ok": True,

    # Hardware
    "device": "0" if torch.cuda.is_available() else "cpu",
    "amp": True,                    # Automatic mixed precision

    # Target metrics
    "patience": 20,                 # Early stopping patience
    "close_mosaic": 10,             # Disable mosaic last N epochs
}


# ─── Loss Configuration ────────────────────────────────────────────────────────
# YOLOv8 uses these losses by default:
#   box loss   → CIoU (Complete IoU)
#   cls loss   → BCE (Binary Cross-Entropy)
#   dfl loss   → DFL (Distribution Focal Loss) for precise localization
#
# Weights are configured via box/cls/dfl hyperparameters:

LOSS_WEIGHTS = {
    "box": 7.5,     # CIoU box regression weight
    "cls": 0.5,     # BCE classification weight
    "dfl": 1.5,     # DFL localization weight
}


# ─── Custom Callback: Live Logging ────────────────────────────────────────────

class GuardianTrainingLogger:
    """
    Custom callback to log metrics, save best model,
    and display progress in a Guardian AI friendly format.
    """

    def __init__(self, log_file: str = "training_log.json"):
        self.log_file = log_file
        self.history = []
        self.best_map50 = 0.0
        self.start_time = datetime.now()

    def on_train_epoch_end(self, trainer):
        """Called at end of each training epoch."""
        metrics = trainer.metrics
        epoch = trainer.epoch

        log_entry = {
            "epoch": epoch,
            "train/box_loss": float(trainer.loss_items[0]) if hasattr(trainer, 'loss_items') else 0,
            "train/cls_loss": float(trainer.loss_items[1]) if hasattr(trainer, 'loss_items') else 0,
            "train/dfl_loss": float(trainer.loss_items[2]) if hasattr(trainer, 'loss_items') else 0,
            "val/mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
            "val/mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            "val/precision": float(metrics.get("metrics/precision(B)", 0)),
            "val/recall": float(metrics.get("metrics/recall(B)", 0)),
            "lr": float(trainer.optimizer.param_groups[0]["lr"]),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(log_entry)

        # Track best mAP
        if log_entry["val/mAP50"] > self.best_map50:
            self.best_map50 = log_entry["val/mAP50"]

        # Log to console
        if epoch % 5 == 0 or log_entry["val/mAP50"] >= 0.90:
            print(f"\n[Guardian AI Trainer] Epoch {epoch:3d}")
            print(f"  Box Loss : {log_entry['train/box_loss']:.4f}")
            print(f"  Cls Loss : {log_entry['train/cls_loss']:.4f}")
            print(f"  DFL Loss : {log_entry['train/dfl_loss']:.4f}")
            print(f"  mAP@50   : {log_entry['val/mAP50']:.4f} (best: {self.best_map50:.4f})")
            print(f"  LR       : {log_entry['lr']:.6f}")

            if log_entry["val/mAP50"] >= 0.90:
                print("  🎯 TARGET mAP ≥ 90% REACHED!")

        # Save log
        import json
        with open(self.log_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def on_train_end(self, trainer):
        """Final summary after training."""
        elapsed = datetime.now() - self.start_time
        print("\n" + "=" * 60)
        print("  Guardian AI - Training Complete")
        print("=" * 60)
        print(f"  Total Time   : {elapsed}")
        print(f"  Best mAP@50  : {self.best_map50:.4f}")
        print(f"  Target Met   : {'✓ YES' if self.best_map50 >= 0.90 else '✗ Keep training'}")
        print(f"  Log saved    : {self.log_file}")
        print("=" * 60)


# ─── Training Function ────────────────────────────────────────────────────────

def train_guardian_model(
    resume: bool = False,
    resume_path: str = None,
    use_pretrained: bool = True,
):
    """
    Main training entry point.

    Args:
        resume: Resume from last checkpoint
        resume_path: Path to specific checkpoint .pt file
        use_pretrained: Start from COCO pretrained weights (recommended)
    """
    print("=" * 60)
    print("  Guardian AI - YOLOv8-Nano Training")
    print("=" * 60)
    print(f"  Device  : {CONFIG['device']}")
    print(f"  Epochs  : {CONFIG['epochs']}")
    print(f"  Batch   : {CONFIG['batch']}")
    print(f"  ImgSz   : {CONFIG['imgsz']}")
    print(f"  Classes : {['deer', 'boar', 'wolf', 'cattle', 'dog']}")
    print()

    # Initialize logger
    logger = GuardianTrainingLogger()

    # Load model
    if resume and resume_path:
        print(f"[✓] Resuming from: {resume_path}")
        model = YOLO(resume_path)
    else:
        model_path = CONFIG["model"] if use_pretrained else "yolov8n.yaml"
        print(f"[✓] Loading base model: {model_path}")
        model = YOLO(model_path)

    # Register custom callbacks
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_train_end", logger.on_train_end)

    # Build training arguments
    train_args = {
        **CONFIG,
        **LOSS_WEIGHTS,
    }

    # Start training
    print("[→] Starting training...\n")
    results = model.train(**train_args)

    return model, results


# ─── Validation / Evaluation ──────────────────────────────────────────────────

def evaluate_model(model_path: str, data_yaml: str = "guardian_dataset.yaml"):
    """
    Run full validation on the trained model.
    Prints per-class mAP, precision, recall, F1.
    """
    print(f"\n[→] Evaluating model: {model_path}")
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=640,
        batch=8,
        conf=0.25,
        iou=0.6,
        device=CONFIG["device"],
        verbose=True,
    )

    print("\n[✓] Evaluation Results:")
    print(f"  mAP@50      : {results.box.map50:.4f}")
    print(f"  mAP@50-95   : {results.box.map:.4f}")
    print(f"  Precision   : {results.box.mp:.4f}")
    print(f"  Recall      : {results.box.mr:.4f}")

    # Per-class results
    classes = ["deer", "boar", "wolf", "cattle", "dog"]
    print("\n  Per-Class mAP@50:")
    for cls, ap in zip(classes, results.box.ap50):
        status = "✓" if ap >= 0.90 else "✗"
        print(f"    {status} {cls:10s}: {ap:.4f}")

    return results


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardian AI Training")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--model", type=str, default=None, help="Path to model for eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        model, results = train_guardian_model(
            resume=args.resume,
            resume_path=args.resume_path,
        )
        # Save final model
        best_path = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
        print(f"\n[✓] Best model: {best_path}")

    elif args.mode == "eval":
        assert args.model, "Provide --model path for evaluation"
        evaluate_model(args.model)
