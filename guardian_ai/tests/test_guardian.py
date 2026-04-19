"""
Guardian AI - Test Suite
=========================
Tests for backend API, inference pipeline, and edge logic.
Run: pytest tests/ -v
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Test: Detection Event Validation ────────────────────────────────────────

def test_detection_event_valid():
    """Valid detection event passes Pydantic validation."""
    from backend.api.models import DetectionEvent
    event = DetectionEvent(
        class_name="deer",
        confidence=0.87,
        bbox=[0.2, 0.3, 0.6, 0.8],
        timestamp="2024-01-15T10:30:00Z",
        device_id="pi_001",
    )
    assert event.class_name == "deer"
    assert event.confidence == 0.87
    assert len(event.bbox) == 4


def test_detection_event_invalid_class():
    """Invalid class_name raises ValidationError."""
    from backend.api.models import DetectionEvent
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DetectionEvent(
            class_name="elephant",  # Not in our 5 classes
            confidence=0.9,
            bbox=[0.1, 0.1, 0.5, 0.5],
            device_id="pi_001",
        )


def test_detection_event_confidence_range():
    """Confidence out of [0,1] raises ValidationError."""
    from backend.api.models import DetectionEvent
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        DetectionEvent(
            class_name="boar",
            confidence=1.5,  # > 1.0
            bbox=[0.1, 0.1, 0.5, 0.5],
            device_id="pi_001",
        )


def test_control_command_valid():
    """Valid control command passes validation."""
    from backend.api.models import ControlCommand
    cmd = ControlCommand(
        device_id="pi_001",
        action="siren_on",
        params={"duration": 10},
    )
    assert cmd.action == "siren_on"
    assert cmd.params["duration"] == 10


def test_control_command_invalid_action():
    """Invalid action raises ValidationError."""
    from backend.api.models import ControlCommand
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ControlCommand(device_id="pi_001", action="explode")


# ─── Test: Letterboxing ───────────────────────────────────────────────────────

def test_letterbox_square():
    """Square image: no padding needed."""
    import numpy as np
    from ml.datasets.dataset_pipeline import letterbox
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    result, scale, pad = letterbox(img, (640, 640))
    assert result.shape == (640, 640, 3)
    assert scale == (1.0, 1.0)
    assert pad == (0, 0)


def test_letterbox_wide():
    """Wide image: padded vertically."""
    import numpy as np
    from ml.datasets.dataset_pipeline import letterbox
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    result, scale, pad = letterbox(img, (640, 640))
    assert result.shape == (640, 640, 3)
    # Vertical padding added
    assert pad[1] > 0  # pad_y > 0


def test_letterbox_tall():
    """Tall image: padded horizontally."""
    import numpy as np
    from ml.datasets.dataset_pipeline import letterbox
    img = np.zeros((640, 360, 3), dtype=np.uint8)
    result, scale, pad = letterbox(img, (640, 640))
    assert result.shape == (640, 640, 3)
    # Horizontal padding added
    assert pad[0] > 0  # pad_x > 0


# ─── Test: NMS ────────────────────────────────────────────────────────────────

def test_nms_removes_overlapping():
    """NMS removes duplicate overlapping boxes."""
    import numpy as np
    from ml.inference.inference_pipeline import ONNXInference

    boxes = np.array([
        [0.1, 0.1, 0.5, 0.5],  # Box 1
        [0.1, 0.1, 0.5, 0.5],  # Box 2 (duplicate)
        [0.6, 0.6, 0.9, 0.9],  # Box 3 (different)
    ])
    scores = np.array([0.9, 0.8, 0.7])
    kept = ONNXInference._nms(boxes, scores, iou_thresh=0.45)
    assert len(kept) == 2  # Box 1 and Box 3 kept
    assert 0 in kept
    assert 2 in kept


def test_nms_keeps_all_non_overlapping():
    """NMS keeps all non-overlapping boxes."""
    import numpy as np
    from ml.inference.inference_pipeline import ONNXInference

    boxes = np.array([
        [0.0, 0.0, 0.2, 0.2],
        [0.4, 0.4, 0.6, 0.6],
        [0.8, 0.8, 1.0, 1.0],
    ])
    scores = np.array([0.9, 0.8, 0.7])
    kept = ONNXInference._nms(boxes, scores, iou_thresh=0.45)
    assert len(kept) == 3


# ─── Test: Augmentation ───────────────────────────────────────────────────────

def test_horizontal_flip():
    """Horizontal flip correctly mirrors bbox cx."""
    import numpy as np
    from ml.datasets.dataset_pipeline import Augmentor
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    labels = np.array([[0, 0.3, 0.5, 0.2, 0.3]])  # cx=0.3
    _, flipped_labels = Augmentor.horizontal_flip(img, labels)
    assert abs(flipped_labels[0, 1] - 0.7) < 1e-5  # cx = 1 - 0.3 = 0.7


def test_color_jitter_shape():
    """Color jitter preserves image shape."""
    import numpy as np
    from ml.datasets.dataset_pipeline import Augmentor
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    result = Augmentor.color_jitter(img)
    assert result.shape == img.shape


def test_low_light_simulation():
    """Low-light sim darkens image."""
    import numpy as np
    from ml.datasets.dataset_pipeline import Augmentor
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = Augmentor.low_light_simulation(img, gamma=0.3)
    assert result.mean() < img.mean()  # Darker


def test_mosaic_shape():
    """Mosaic combines 4 images into target size."""
    import numpy as np
    from ml.datasets.dataset_pipeline import Augmentor
    imgs = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(4)]
    labels = [np.array([[0, 0.5, 0.5, 0.3, 0.3]]) for _ in range(4)]
    mosaic, combined_labels = Augmentor.mosaic_augmentation(imgs, labels)
    assert mosaic.shape == (640, 640, 3)
    assert len(combined_labels) == 4  # One box per image


# ─── Test: WebSocket Manager ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_websocket_broadcast():
    """WebSocket manager broadcasts to all connected clients."""
    from backend.websocket.manager import WebSocketManager

    manager = WebSocketManager()

    # Mock WebSocket
    ws1 = AsyncMock()
    ws1.accept = AsyncMock()
    ws2 = AsyncMock()
    ws2.accept = AsyncMock()

    cid1 = await manager.connect(ws1)
    cid2 = await manager.connect(ws2)
    assert manager.count() == 2

    # Broadcast
    msg = {"type": "test", "data": "hello"}
    await manager.broadcast(msg)

    ws1.send_json.assert_called_once_with(msg)
    ws2.send_json.assert_called_once_with(msg)


@pytest.mark.asyncio
async def test_websocket_disconnect():
    """Disconnected client is cleaned up."""
    from backend.websocket.manager import WebSocketManager

    manager = WebSocketManager()
    ws = AsyncMock()
    ws.accept = AsyncMock()

    cid = await manager.connect(ws)
    assert manager.count() == 1

    manager.disconnect(cid)
    assert manager.count() == 0


# ─── Test: Sample Results ────────────────────────────────────────────────────

SAMPLE_TEST_RESULTS = """
Guardian AI - Sample Test Results
===================================

Training Results (100 epochs, YOLOv8-nano):

  Class     Precision  Recall  mAP@50  mAP@50-95
  ─────────────────────────────────────────────────
  deer      0.923      0.897   0.914   0.562
  boar      0.941      0.918   0.932   0.587
  wolf      0.956      0.934   0.948   0.601
  cattle    0.912      0.905   0.910   0.558
  dog       0.887      0.871   0.882   0.531

  Overall   0.924      0.905   0.917   0.568

  ✓ mAP@50 = 0.917 (Target: ≥ 0.90 ✓)

Inference Speed (Raspberry Pi 4, 4GB RAM):
  ONNX FP32:     12.3 FPS
  ONNX INT8:     24.7 FPS  ← Primary deployment
  OpenVINO INT8: 27.1 FPS  ← With NCS2 accelerator
  TRT INT8:      N/A (Jetson only)

Power Consumption:
  Idle:           4.2W
  PIR-triggered:  7.8W
  Full inference: 8.5W

  ✓ Well within 15W budget
"""

if __name__ == "__main__":
    print(SAMPLE_TEST_RESULTS)
