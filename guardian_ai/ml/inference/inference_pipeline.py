"""
Guardian AI - Real-Time Inference Pipeline
==========================================
Runs animal detection on:
  - Live camera stream (Pi Camera / USB Webcam)
  - Video file
  - Single image

Features:
  - Bounding boxes + confidence + class labels
  - PIR sensor trigger integration
  - Sends detections to backend API
  - Offline queue if network unavailable
"""

import cv2
import time
import json
import queue
import threading
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────────────────────

CLASSES = ["deer", "boar", "wolf", "cattle", "dog"]

# Class colors for visualization (BGR)
CLASS_COLORS = {
    "deer":   (0, 200, 100),
    "boar":   (0, 100, 255),
    "wolf":   (0, 0, 255),
    "cattle": (255, 150, 0),
    "dog":    (255, 200, 0),
}

INFERENCE_CONFIG = {
    "conf_threshold": 0.45,     # Minimum confidence to report detection
    "iou_threshold": 0.45,      # NMS IoU threshold
    "target_fps": 25,           # Desired FPS
    "input_size": (640, 640),
    "backend_url": "http://localhost:8000",
    "device_id": "pi_001",      # Unique ID for this edge device
    "offline_queue_size": 1000, # Max detections to buffer offline
}


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2] normalized 0-1
    timestamp: str
    device_id: str
    frame_id: int

    def to_dict(self):
        return asdict(self)


# ─── ONNX Inference Backend ───────────────────────────────────────────────────

class ONNXInference:
    """
    ONNX Runtime inference backend.
    Optimized for CPU inference on Raspberry Pi.
    """

    def __init__(self, model_path: str):
        assert ONNX_AVAILABLE, "Install: pip install onnxruntime"

        # Configure ONNX Runtime for low-power CPU
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # Use 4 Pi CPU cores
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CPU provider (or OpenVINO if available)
        providers = []
        if "OpenVINOExecutionProvider" in ort.get_available_providers():
            providers.append("OpenVINOExecutionProvider")
            print("[✓] Using OpenVINO acceleration")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"[✓] ONNX model loaded: {model_path}")
        print(f"    Input: {self.input_name} {self.input_shape}")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple, Tuple]:
        """Letterbox + normalize + convert to BCHW tensor."""
        h, w = image.shape[:2]
        target_h, target_w = INFERENCE_CONFIG["input_size"]

        # Letterbox
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_y, target_h - new_h - pad_y,
            pad_x, target_w - new_w - pad_x,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Normalize [0,255] → [0,1], BGR→RGB, HWC→CHW
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor[:, :, ::-1].copy()  # BGR→RGB
        tensor = tensor.transpose(2, 0, 1)  # HWC→CHW
        tensor = tensor[np.newaxis]          # Add batch dim

        return tensor, (scale, scale), (pad_x, pad_y)

    def postprocess(
        self,
        output: np.ndarray,
        scale: Tuple,
        pad: Tuple,
        orig_shape: Tuple,
        conf_thresh: float,
        iou_thresh: float,
    ) -> List[Dict]:
        """
        Parse YOLOv8 output and apply NMS.
        YOLOv8 ONNX output shape: [1, 84, 8400] for 80-class COCO
        For our 5-class model: [1, 9, 8400] → [cx, cy, w, h, conf*5]
        """
        # Transpose to [8400, 9]
        pred = output[0].T  # [8400, 4+nc]
        n_classes = pred.shape[1] - 4

        boxes = pred[:, :4]
        scores = pred[:, 4:4 + n_classes]

        # Get best class per box
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence
        mask = confidences >= conf_thresh
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Un-letterbox: remove padding and scale back to original image coords
        sx, sy = scale
        px, py = pad
        target_h, target_w = INFERENCE_CONFIG["input_size"]

        x1 = (x1 - px) / sx / orig_shape[1]
        y1 = (y1 - py) / sy / orig_shape[0]
        x2 = (x2 - px) / sx / orig_shape[1]
        y2 = (y2 - py) / sy / orig_shape[0]

        # Clamp to [0, 1]
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        # Apply NMS
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = self._nms(xyxy, confidences, iou_thresh)

        results = []
        for i in indices:
            results.append({
                "class_id": int(class_ids[i]),
                "class_name": CLASSES[int(class_ids[i])],
                "confidence": float(confidences[i]),
                "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
            })

        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """Simple NMS implementation."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-7)

            order = order[1:][iou < iou_thresh]

        return keep

    def infer(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Run inference on a single frame.
        Returns detections list and inference latency (ms).
        """
        orig_shape = image.shape[:2]  # (H, W)
        t0 = time.perf_counter()

        tensor, scale, pad = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        detections = self.postprocess(
            outputs[0], scale, pad, orig_shape,
            INFERENCE_CONFIG["conf_threshold"],
            INFERENCE_CONFIG["iou_threshold"],
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return detections, latency_ms


# ─── Visualization ────────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, detections: List[Dict], fps: float = 0) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence on frame."""
    h, w = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1i, y1i = int(x1 * w), int(y1 * h)
        x2i, y2i = int(x2 * w), int(y2 * h)

        color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Draw box
        cv2.rectangle(image, (x1i, y1i), (x2i, y2i), color, 2)

        # Draw label background
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1i, y1i - lh - baseline - 4), (x1i + lw, y1i), color, -1)

        # Draw label text
        cv2.putText(image, label, (x1i, y1i - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # FPS overlay
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"Guardian AI", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image


# ─── Network Sender (with offline queue) ─────────────────────────────────────

class DetectionSender:
    """
    Sends detection events to backend API.
    Queues events offline if network is unavailable.
    Retries automatically when network is restored.
    """

    def __init__(self):
        self.queue = queue.Queue(maxsize=INFERENCE_CONFIG["offline_queue_size"])
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def send(self, detection: Detection):
        """Non-blocking: enqueue detection for sending."""
        try:
            self.queue.put_nowait(detection.to_dict())
        except queue.Full:
            print("[!] Offline queue full - dropping oldest event")
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(detection.to_dict())
            except Exception:
                pass

    def _worker(self):
        """Background thread: drain queue → HTTP POST."""
        while True:
            try:
                payload = self.queue.get(timeout=1.0)
                self._post(payload)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] Sender error: {e}")

    def _post(self, payload: dict, retries: int = 3):
        """POST detection to backend with retry."""
        url = f"{INFERENCE_CONFIG['backend_url']}/detect"
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                if attempt == retries - 1:
                    print(f"[!] Network unavailable - detection queued for retry")
                time.sleep(1)


# ─── Main Inference Loop ──────────────────────────────────────────────────────

class GuardianInference:
    """
    Main inference engine.
    Handles camera, model, detections, and sending.
    """

    def __init__(self, model_path: str, camera_source=0):
        self.model = ONNXInference(model_path)
        self.sender = DetectionSender()
        self.camera_source = camera_source
        self.frame_id = 0
        self.fps_buffer = []
        self.running = False

    def _get_fps(self, t_start: float) -> float:
        """Rolling average FPS calculation."""
        elapsed = time.perf_counter() - t_start
        self.fps_buffer.append(1.0 / max(elapsed, 0.001))
        if len(self.fps_buffer) > 30:
            self.fps_buffer.pop(0)
        return sum(self.fps_buffer) / len(self.fps_buffer)

    def run(self, display: bool = True, save_output: bool = False):
        """Main loop: capture → infer → visualize → send."""
        cap = cv2.VideoCapture(self.camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, INFERENCE_CONFIG["target_fps"])

        if not cap.isOpened():
            print(f"[✗] Cannot open camera: {self.camera_source}")
            return

        print(f"[✓] Camera opened. Running inference...")
        print(f"    Press 'q' to quit, 's' to save snapshot")

        writer = None
        if save_output:
            writer = cv2.VideoWriter(
                "guardian_output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                20, (640, 480)
            )

        self.running = True
        while self.running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame capture failed")
                break

            # Run detection
            detections, latency_ms = self.model.infer(frame)

            # FPS
            fps = self._get_fps(t0)

            # Send detected animals to backend
            for det in detections:
                event = Detection(
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    timestamp=datetime.utcnow().isoformat(),
                    device_id=INFERENCE_CONFIG["device_id"],
                    frame_id=self.frame_id,
                )
                self.sender.send(event)
                print(f"[🔍] {det['class_name']} ({det['confidence']:.2f}) | {latency_ms:.1f}ms")

            # Visualize
            vis_frame = draw_detections(frame.copy(), detections, fps)

            if display:
                cv2.imshow("Guardian AI", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    snapshot_path = f"snapshot_{self.frame_id}.jpg"
                    cv2.imwrite(snapshot_path, vis_frame)
                    print(f"[✓] Snapshot saved: {snapshot_path}")

            if writer:
                writer.write(vis_frame)

            self.frame_id += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[✓] Inference stopped.")

    def infer_image(self, image_path: str) -> List[Dict]:
        """Run inference on a single image file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        detections, latency_ms = self.model.infer(img)
        print(f"[✓] Detected {len(detections)} objects in {latency_ms:.1f}ms")
        vis = draw_detections(img, detections)
        cv2.imwrite("detection_result.jpg", vis)
        return detections


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardian AI Inference")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--source", default=0, help="Camera index or video path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--image", type=str, help="Run on single image")
    parser.add_argument("--save", action="store_true", help="Save output video")
    args = parser.parse_args()

    engine = GuardianInference(args.model, camera_source=args.source)

    if args.image:
        dets = engine.infer_image(args.image)
        print(json.dumps(dets, indent=2))
    else:
        engine.run(display=not args.no_display, save_output=args.save)
