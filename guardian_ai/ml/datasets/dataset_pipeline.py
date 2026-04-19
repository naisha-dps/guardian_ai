"""
Guardian AI - Dataset Pipeline
================================
Downloads, prepares, and splits dataset for YOLOv8 training.
Supports: COCO subsets, Roboflow, custom animal datasets.
Classes: deer, boar, wolf, cattle, dog
"""

import os
import shutil
import random
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import requests
import zipfile


# ─── Configuration ────────────────────────────────────────────────────────────

CLASSES = ["deer", "boar", "wolf", "cattle", "dog"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}

# COCO category IDs mapped to our classes
COCO_MAPPING = {
    19: "cattle",   # COCO 'cow'
    18: "dog",      # COCO 'dog'
    # deer, boar, wolf come from custom/Roboflow datasets
}

DATASET_DIR = Path("data")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

TARGET_SIZE = (640, 640)  # YOLOv8 standard input


# ─── Step 1: Directory Setup ───────────────────────────────────────────────────

def create_dataset_structure():
    """Create YOLO-format directory structure."""
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    print("[✓] Dataset directory structure created.")


# ─── Step 2: Letterbox Preprocessing ──────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize image with letterboxing (preserves aspect ratio).
    Pads with gray to fill target_size.

    Returns:
        image: resized+padded image
        ratio: (scale_w, scale_h)
        pad: (pad_x, pad_y)
    """
    h, w = image.shape[:2]
    tw, th = target_size

    # Compute scale ratio (maintain aspect ratio)
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized,
        pad_y, th - new_h - pad_y,
        pad_x, tw - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, (scale, scale), (pad_x, pad_y)


def adjust_labels_for_letterbox(
    labels: np.ndarray,
    orig_size: Tuple[int, int],
    scale: Tuple[float, float],
    pad: Tuple[int, int],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> np.ndarray:
    """
    Adjust YOLO bounding box labels after letterboxing.
    Labels format: [class_id, cx, cy, w, h] (normalized 0-1)
    """
    if len(labels) == 0:
        return labels

    orig_w, orig_h = orig_size
    tw, th = target_size
    pad_x, pad_y = pad
    sx, sy = scale

    adjusted = labels.copy()
    # Denormalize
    adjusted[:, 1] *= orig_w
    adjusted[:, 2] *= orig_h
    adjusted[:, 3] *= orig_w
    adjusted[:, 4] *= orig_h
    # Scale
    adjusted[:, 1] = adjusted[:, 1] * sx + pad_x
    adjusted[:, 2] = adjusted[:, 2] * sy + pad_y
    adjusted[:, 3] *= sx
    adjusted[:, 4] *= sy
    # Re-normalize
    adjusted[:, 1] /= tw
    adjusted[:, 2] /= th
    adjusted[:, 3] /= tw
    adjusted[:, 4] /= th

    return adjusted


# ─── Step 3: Data Augmentation ────────────────────────────────────────────────

class Augmentor:
    """
    Applies training-time augmentations to images and labels.
    """

    @staticmethod
    def horizontal_flip(image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flip image and adjust box cx."""
        flipped = cv2.flip(image, 1)
        if len(labels):
            labels = labels.copy()
            labels[:, 1] = 1.0 - labels[:, 1]  # cx = 1 - cx
        return flipped, labels

    @staticmethod
    def color_jitter(
        image: np.ndarray,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1
    ) -> np.ndarray:
        """Random color jitter in HSV space."""
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue
        img_hsv[:, :, 0] += random.uniform(-hue * 180, hue * 180)
        img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 180)

        # Saturation
        img_hsv[:, :, 1] *= random.uniform(1 - saturation, 1 + saturation)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

        # Brightness (Value channel)
        img_hsv[:, :, 2] *= random.uniform(1 - brightness, 1 + brightness)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)

        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def low_light_simulation(image: np.ndarray, gamma: float = None) -> np.ndarray:
        """Simulate low-light / night conditions using gamma correction."""
        if gamma is None:
            gamma = random.uniform(0.2, 0.6)  # Dark
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(image, table)

    @staticmethod
    def weather_simulation(image: np.ndarray, mode: str = "rain") -> np.ndarray:
        """Simulate rain, fog, or noise weather effects."""
        h, w = image.shape[:2]

        if mode == "rain":
            # Add rain streaks
            rain = np.zeros_like(image)
            num_drops = random.randint(300, 800)
            for _ in range(num_drops):
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                length = random.randint(10, 30)
                x2 = x1 + random.randint(-5, 5)
                y2 = min(y1 + length, h - 1)
                cv2.line(rain, (x1, y1), (x2, y2), (200, 200, 200), 1)
            return cv2.addWeighted(image, 0.85, rain, 0.15, 0)

        elif mode == "fog":
            fog = np.ones_like(image) * 200
            alpha = random.uniform(0.3, 0.6)
            return cv2.addWeighted(image, 1 - alpha, fog, alpha, 0)

        elif mode == "noise":
            noise = np.random.normal(0, 25, image.shape).astype(np.int16)
            return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def mosaic_augmentation(
        images: List[np.ndarray],
        labels_list: List[np.ndarray],
        target_size: Tuple[int, int] = TARGET_SIZE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        YOLOv8-style mosaic: combines 4 images into one.
        Each quadrant gets one image; labels are adjusted accordingly.
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        tw, th = target_size
        cx, cy = tw // 2, th // 2

        mosaic_img = np.full((th, tw, 3), 114, dtype=np.uint8)
        all_labels = []

        positions = [
            (0, 0, cx, cy),       # top-left
            (cx, 0, tw, cy),      # top-right
            (0, cy, cx, th),      # bottom-left
            (cx, cy, tw, th),     # bottom-right
        ]

        for i, (x1, y1, x2, y2) in enumerate(positions):
            img = cv2.resize(images[i], (x2 - x1, y2 - y1))
            mosaic_img[y1:y2, x1:x2] = img

            if len(labels_list[i]):
                lbl = labels_list[i].copy()
                # Convert normalized coords to mosaic-space
                lbl[:, 1] = (lbl[:, 1] * (x2 - x1) + x1) / tw
                lbl[:, 2] = (lbl[:, 2] * (y2 - y1) + y1) / th
                lbl[:, 3] *= (x2 - x1) / tw
                lbl[:, 4] *= (y2 - y1) / th
                all_labels.append(lbl)

        combined_labels = np.concatenate(all_labels) if all_labels else np.array([])
        return mosaic_img, combined_labels


# ─── Step 4: Dataset Split ─────────────────────────────────────────────────────

def split_dataset(image_paths: List[Path], label_paths: List[Path]):
    """
    Splits images/labels into train/val/test sets.
    Ratio: 70/20/10
    """
    assert len(image_paths) == len(label_paths)
    combined = list(zip(image_paths, label_paths))
    random.shuffle(combined)

    n = len(combined)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": combined[:n_train],
        "val": combined[n_train:n_train + n_val],
        "test": combined[n_train + n_val:],
    }

    for split, pairs in splits.items():
        for img_path, lbl_path in pairs:
            shutil.copy(img_path, IMAGES_DIR / split / img_path.name)
            shutil.copy(lbl_path, LABELS_DIR / split / lbl_path.name)
        print(f"[✓] {split}: {len(pairs)} samples")

    print(f"\n[✓] Dataset split complete. Total: {n} samples")


# ─── Step 5: YAML Config ───────────────────────────────────────────────────────

def generate_dataset_yaml(output_path: str = "guardian_dataset.yaml"):
    """Generate YOLOv8-compatible dataset.yaml config."""
    config = {
        "path": str(DATASET_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[✓] Dataset YAML saved to: {output_path}")
    return output_path


# ─── Step 6: Sample Generation (for testing pipeline) ─────────────────────────

def generate_synthetic_samples(num_samples: int = 50, output_dir: Path = Path("data/raw")):
    """
    Generates synthetic labeled samples for pipeline testing.
    In production, replace with real images from Roboflow/COCO.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    for i in range(num_samples):
        # Create a random 640x640 BGR image
        img = np.random.randint(30, 200, (640, 640, 3), dtype=np.uint8)

        # Simulate a detection box
        class_id = random.randint(0, len(CLASSES) - 1)
        cx = random.uniform(0.2, 0.8)
        cy = random.uniform(0.2, 0.8)
        bw = random.uniform(0.1, 0.4)
        bh = random.uniform(0.1, 0.4)

        # Draw a colored rectangle (simulated animal silhouette)
        x1 = int((cx - bw / 2) * 640)
        y1 = int((cy - bh / 2) * 640)
        x2 = int((cx + bw / 2) * 640)
        y2 = int((cy + bh / 2) * 640)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save image
        img_name = f"sample_{i:04d}.jpg"
        cv2.imwrite(str(output_dir / "images" / img_name), img)

        # Save label
        lbl_name = f"sample_{i:04d}.txt"
        with open(output_dir / "labels" / lbl_name, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[✓] Generated {num_samples} synthetic samples in {output_dir}")
    return output_dir


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Guardian AI - Dataset Pipeline")
    print("=" * 60)

    # 1. Create structure
    create_dataset_structure()

    # 2. Generate synthetic samples (replace with real data download)
    raw_dir = generate_synthetic_samples(num_samples=100)

    # 3. Split dataset
    img_paths = sorted(list((raw_dir / "images").glob("*.jpg")))
    lbl_paths = sorted(list((raw_dir / "labels").glob("*.txt")))
    split_dataset(img_paths, lbl_paths)

    # 4. Generate config
    yaml_path = generate_dataset_yaml()

    print("\n[✓] Pipeline complete. Ready for training!")
    print(f"     Dataset config: {yaml_path}")
