"""
Guardian AI - Model Export & Optimization
==========================================
Exports YOLOv8-nano to:
  1. ONNX (cross-platform)
  2. OpenVINO (Intel Raspberry Pi / NCS2)
  3. TensorRT (NVIDIA Jetson)
  4. INT8 Quantized versions for edge deployment

Target: ~15W power, 20-30 FPS inference
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# pip install ultralytics openvino-dev
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────

EXPORT_CONFIG = {
    "imgsz": 640,
    "half": False,      # FP16 (enable for GPU/TensorRT)
    "int8": True,       # INT8 quantization for edge devices
    "dynamic": False,   # Static shape for edge deployment
    "simplify": True,   # Simplify ONNX graph
    "opset": 12,        # ONNX opset version
    "batch": 1,         # Single-frame inference on edge
}


# ─── 1. ONNX Export ───────────────────────────────────────────────────────────

def export_onnx(model_path: str, output_dir: str = "models/exported") -> str:
    """
    Export YOLOv8 model to ONNX format.
    ONNX is universal - runs on CPU/GPU, Raspberry Pi, etc.
    """
    print("\n[→] Exporting to ONNX...")
    model = YOLO(model_path)

    export_path = model.export(
        format="onnx",
        imgsz=EXPORT_CONFIG["imgsz"],
        simplify=EXPORT_CONFIG["simplify"],
        opset=EXPORT_CONFIG["opset"],
        dynamic=EXPORT_CONFIG["dynamic"],
        half=EXPORT_CONFIG["half"],
    )

    # Move to output dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "guardian_wildlife.onnx"
    shutil.copy(export_path, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] ONNX model saved: {out_path} ({size_mb:.1f} MB)")
    return str(out_path)


# ─── 2. OpenVINO Export ───────────────────────────────────────────────────────

def export_openvino(model_path: str, output_dir: str = "models/exported") -> str:
    """
    Export to OpenVINO IR format (XML + BIN).
    Optimized for Intel hardware: Raspberry Pi, NCS2, i-series CPUs.
    INT8 quantization dramatically reduces size and increases FPS.
    """
    print("\n[→] Exporting to OpenVINO (INT8)...")
    model = YOLO(model_path)

    export_path = model.export(
        format="openvino",
        imgsz=EXPORT_CONFIG["imgsz"],
        int8=EXPORT_CONFIG["int8"],
        dynamic=EXPORT_CONFIG["dynamic"],
    )

    out_dir = Path(output_dir) / "openvino_int8"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy OpenVINO files (XML + BIN)
    src = Path(export_path)
    if src.is_dir():
        shutil.copytree(src, out_dir, dirs_exist_ok=True)
    else:
        shutil.copy(src, out_dir)

    print(f"[✓] OpenVINO model saved: {out_dir}")
    return str(out_dir)


# ─── 3. TensorRT Export ───────────────────────────────────────────────────────

def export_tensorrt(model_path: str, output_dir: str = "models/exported") -> Optional[str]:
    """
    Export to TensorRT engine (NVIDIA Jetson Nano / Xavier).
    Requires: CUDA + TensorRT installed on target device.
    INT8 + FP16 for maximum throughput on Jetson.
    """
    try:
        import tensorrt  # noqa
    except ImportError:
        print("[!] TensorRT not available. Skipping TRT export.")
        print("    Install on Jetson: pip install tensorrt")
        return None

    print("\n[→] Exporting to TensorRT (INT8)...")
    model = YOLO(model_path)

    export_path = model.export(
        format="engine",
        imgsz=EXPORT_CONFIG["imgsz"],
        half=True,           # FP16 for Jetson
        int8=True,           # INT8 quantization
        device=0,            # GPU required
        workspace=4,         # 4GB workspace for Jetson
    )

    out_path = Path(output_dir) / "guardian_wildlife.engine"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(export_path, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] TensorRT engine saved: {out_path} ({size_mb:.1f} MB)")
    return str(out_path)


# ─── 4. Model Size Benchmark ──────────────────────────────────────────────────

def benchmark_model(model_path: str, format: str = "onnx"):
    """
    Run inference speed benchmark.
    Reports latency, throughput, and memory usage.
    """
    print(f"\n[→] Benchmarking {format.upper()} model...")

    try:
        from ultralytics.utils.benchmarks import benchmark
        results = benchmark(
            model=model_path,
            data="guardian_dataset.yaml",
            imgsz=640,
            half=False,
            device="cpu",   # Simulate Raspberry Pi CPU
        )
        print(f"[✓] Benchmark complete:\n{results}")
    except Exception as e:
        print(f"[!] Benchmark error: {e}")


# ─── 5. INT8 Calibration Script ───────────────────────────────────────────────

def calibrate_int8(onnx_path: str, calibration_data_dir: str, output_path: str):
    """
    Post-training INT8 quantization using calibration dataset.
    Uses ONNX Runtime quantization for CPU deployment.

    Args:
        onnx_path: Path to FP32 ONNX model
        calibration_data_dir: Directory of calibration images (100-200 images)
        output_path: Output path for quantized INT8 model
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
        )
        import onnxruntime as ort
        import cv2
        import numpy as np
    except ImportError:
        print("[!] onnxruntime not installed. Run: pip install onnxruntime-tools")
        return

    class AnimalCalibrationReader(CalibrationDataReader):
        """Feeds calibration images to INT8 quantizer."""

        def __init__(self, data_dir: str, input_name: str = "images"):
            self.data_dir = Path(data_dir)
            self.input_name = input_name
            self.image_files = list(self.data_dir.glob("*.jpg"))[:200]
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.image_files):
                return None
            img_path = self.image_files[self.idx]
            self.idx += 1

            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)[np.newaxis]  # BHWC→BCHW

            return {self.input_name: img}

    print(f"\n[→] INT8 Quantization of {onnx_path}...")
    reader = AnimalCalibrationReader(calibration_data_dir)

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_type=QuantType.QInt8,
    )

    # Size comparison
    orig_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quant_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    print(f"[✓] INT8 model saved: {output_path}")
    print(f"    Original  : {orig_size:.1f} MB")
    print(f"    Quantized : {quant_size:.1f} MB ({reduction:.1f}% reduction)")


# ─── 6. Power Optimization Notes ─────────────────────────────────────────────

POWER_OPTIMIZATION_GUIDE = """
Guardian AI - Power Optimization for ~15W Operation
=====================================================

Raspberry Pi 4 (typical ~6.5W idle, ~8W under load):
  ✓ Use OpenVINO INT8 model → reduces compute ~4x
  ✓ Disable HDMI: vcgencmd display_power 0
  ✓ Use camera trigger (PIR) instead of continuous capture
  ✓ CPU governor: performance mode only during inference
      echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  ✓ Disable WiFi when using LoRa/GSM:
      rfkill block wifi
  ✓ Disable Bluetooth: rfkill block bluetooth
  ✓ Reduce polling rate: process at 5 FPS normally, 25 FPS on PIR trigger
  ✓ Use lightweight MJPEG stream instead of H264 for local preview

Total estimated: ~7-10W (well under 15W budget)
"""


# ─── Main Export Pipeline ─────────────────────────────────────────────────────

def run_full_export(model_path: str = "runs/train/guardian_wildlife/weights/best.pt"):
    """Run the complete export pipeline."""
    print("=" * 60)
    print("  Guardian AI - Model Export Pipeline")
    print("=" * 60)

    output_dir = "models/exported"

    # 1. ONNX
    onnx_path = export_onnx(model_path, output_dir)

    # 2. OpenVINO (for Raspberry Pi with Intel NCS2 or ARM)
    ov_path = export_openvino(model_path, output_dir)

    # 3. TensorRT (for Jetson, skip if no GPU)
    trt_path = export_tensorrt(model_path, output_dir)

    # 4. INT8 ONNX
    int8_path = str(Path(output_dir) / "guardian_wildlife_int8.onnx")
    calibrate_int8(
        onnx_path=onnx_path,
        calibration_data_dir="data/images/val",
        output_path=int8_path,
    )

    # Print guide
    print(POWER_OPTIMIZATION_GUIDE)

    print("\n[✓] Export pipeline complete!")
    print(f"    ONNX        : {onnx_path}")
    print(f"    OpenVINO    : {ov_path}")
    print(f"    TensorRT    : {trt_path or 'N/A'}")
    print(f"    INT8 ONNX   : {int8_path}")

    return {
        "onnx": onnx_path,
        "openvino": ov_path,
        "tensorrt": trt_path,
        "int8_onnx": int8_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/train/guardian_wildlife/weights/best.pt")
    parser.add_argument("--format", choices=["all", "onnx", "openvino", "trt"], default="all")
    args = parser.parse_args()

    if args.format == "all":
        run_full_export(args.model)
    elif args.format == "onnx":
        export_onnx(args.model)
    elif args.format == "openvino":
        export_openvino(args.model)
    elif args.format == "trt":
        export_tensorrt(args.model)
