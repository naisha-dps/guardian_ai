# Guardian AI 🛡️🦌
## Autonomous Wildlife Detection & Crop Protection System

> **Real-time animal detection on edge devices, with mobile alerts and remote deterrent control.**

---

## 📐 System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GUARDIAN AI SYSTEM                              │
├─────────────────────────────┬──────────────────────────────────────────┤
│        EDGE LAYER           │           CLOUD/SERVER LAYER             │
│                             │                                          │
│  ┌──────────────────────┐   │   ┌────────────────────────────────┐     │
│  │   Raspberry Pi 4     │   │   │   FastAPI Backend              │     │
│  │  ┌────────────────┐  │   │   │   ┌──────────────────────────┐ │     │
│  │  │  Pi Camera     │  │   │   │   │  POST /detect            │ │     │
│  │  │  (CSI/USB)     │  │   │   │   │  GET  /alerts            │ │     │
│  │  └───────┬────────┘  │   │   │   │  GET  /history           │ │     │
│  │          │           │   │   │   │  POST /control           │ │     │
│  │  ┌───────▼────────┐  │   │   │   │  WS   /ws               │ │     │
│  │  │ YOLOv8-nano    │  │──WiFi/│   └──────────┬───────────────┘ │     │
│  │  │ ONNX INT8      │  │  LTE──┤              │                 │     │
│  │  │ ~25 FPS        │  │   │   │   ┌──────────▼───────────────┐ │     │
│  │  └───────┬────────┘  │   │   │   │  MongoDB Database        │ │     │
│  │          │           │   │   │   │  - detections collection │ │     │
│  │  ┌───────▼────────┐  │   │   │   │  - alerts collection     │ │     │
│  │  │ Serial (USB)   │  │   │   │   │  - devices collection    │ │     │
│  │  └───────┬────────┘  │   │   │   └──────────────────────────┘ │     │
│  └──────────┼───────────┘   │   └────────────┬───────────────────┘     │
│             │               │                │                          │
│  ┌──────────▼───────────┐   │   ┌────────────▼───────────────────┐     │
│  │   Arduino Uno        │   │   │   Flutter Mobile App           │     │
│  │  ┌────────────────┐  │   │   │   ┌──────────────────────────┐ │     │
│  │  │  PIR Sensor    │  │   │   │   │  Dashboard (Live Feed)   │ │     │
│  │  │  Siren Relay   │  │   │   │   │  Alerts Screen           │ │     │
│  │  │  Flash Relay   │  │   │   │   │  Map View                │ │     │
│  │  │  Ultrasonic    │  │   │   │   │  Control Panel           │ │     │
│  │  └────────────────┘  │   │   │   │  Analytics               │ │     │
│  └──────────────────────┘   │   │   │  Sahayak AI Assistant    │ │     │
│                             │   │   └──────────────────────────┘ │     │
│  LoRa / GSM (remote areas)  │   └────────────────────────────────┘     │
└─────────────────────────────┴──────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
guardian_ai/
├── ml/
│   ├── datasets/
│   │   └── dataset_pipeline.py      # Data download, preprocessing, split
│   ├── training/
│   │   └── train.py                 # YOLOv8-nano training script
│   ├── inference/
│   │   └── inference_pipeline.py   # Real-time ONNX inference
│   └── export/
│       └── export_model.py          # ONNX/OpenVINO/TRT export + INT8
│
├── backend/
│   ├── main.py                      # FastAPI app + all routes
│   ├── requirements.txt
│   ├── api/
│   │   └── models.py                # Pydantic schemas
│   ├── db/
│   │   └── database.py              # MongoDB async layer
│   ├── websocket/
│   │   └── manager.py               # WS connection manager
│   └── utils/
│       ├── notifications.py         # FCM push notifications
│       └── lora_sim.py              # LoRa/GSM simulator
│
├── edge/
│   ├── raspberry_pi/
│   │   └── main_controller.py      # Pi main loop + PIR + camera
│   ├── arduino/
│   │   └── actuator_controller.ino # Siren/Flash/Ultrasonic
│   └── sensors/
│
├── mobile/
│   ├── pubspec.yaml
│   └── lib/
│       ├── main.dart                # App entry point + theme
│       ├── router.dart              # GoRouter navigation
│       ├── screens/
│       │   ├── dashboard_screen.dart   # Live feed + status
│       │   ├── alerts_screen.dart      # Alert feed + filters
│       │   ├── map_screen.dart         # Detection map
│       │   ├── control_screen.dart     # Deterrent control
│       │   ├── analytics_screen.dart   # Charts + stats
│       │   ├── sahayak_screen.dart     # Multilingual AI chat
│       │   ├── settings_screen.dart    # App settings
│       │   └── alert_detail_screen.dart
│       ├── services/
│       │   ├── api_service.dart        # HTTP client (Dio)
│       │   ├── websocket_service.dart  # WS real-time
│       │   ├── sahayak_service.dart    # AI command processor
│       │   └── notification_service.dart
│       ├── models/
│       │   └── detection_model.dart
│       └── widgets/
│           └── main_scaffold.dart   # Bottom nav
│
├── deploy/
│   ├── docker/
│   │   ├── docker-compose.yml
│   │   └── Dockerfile.backend
│   └── edge/
│       └── setup_pi.sh             # Pi setup script
│
└── docs/
    └── README.md
```

---

## 🚀 Quick Start

### Step 1: Train the Model

```bash
# Install dependencies
pip install ultralytics opencv-python numpy pyyaml

# Prepare dataset
cd ml/datasets
python dataset_pipeline.py

# Train YOLOv8-nano
cd ../training
python train.py --mode train

# Export to ONNX + OpenVINO + INT8
cd ../export
python export_model.py --model runs/train/guardian_wildlife/weights/best.pt
```

### Step 2: Start the Backend

```bash
# Option A: Docker (recommended)
cd deploy/docker
docker-compose up -d

# Option B: Local
cd backend
pip install -r requirements.txt
# Start MongoDB: mongod --dbpath ./data
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Setup Raspberry Pi

```bash
# Copy files to Pi (from your computer)
scp -r guardian_ai/ pi@PI_IP_ADDRESS:/opt/

# On the Pi
chmod +x deploy/edge/setup_pi.sh
sudo ./deploy/edge/setup_pi.sh

# Edit config
sudo nano /etc/systemd/system/guardian_ai.service
# Set BACKEND_URL=http://YOUR_SERVER_IP:8000

sudo systemctl restart guardian_ai
sudo journalctl -u guardian_ai -f
```

### Step 4: Upload Arduino Firmware

```bash
# Open Arduino IDE
# File → Open → edge/arduino/actuator_controller.ino
# Tools → Board → Arduino Uno
# Sketch → Upload
```

### Step 5: Build Mobile App

```bash
cd mobile
flutter pub get
flutter run  # or: flutter build apk --release
```

---

## 🤖 ML Model Details

| Parameter      | Value                          |
|----------------|-------------------------------|
| Base Model     | YOLOv8-nano                   |
| Input Size     | 640×640                       |
| Classes        | deer, boar, wolf, cattle, dog |
| Optimizer      | SGD (lr=0.01, momentum=0.937) |
| Loss Functions | CIoU + BCE + DFL              |
| Export Formats | ONNX, OpenVINO INT8, TensorRT |
| Target mAP     | ≥ 90% @ IoU 0.5               |
| Edge FPS       | ~20-30 FPS (Pi 4, INT8)       |
| Model Size     | ~3.2 MB (INT8 quantized)      |

### Dataset Sources
- **COCO 2017**: cattle (cow), dog classes
- **Roboflow**: wildlife animal datasets
- **Custom**: deer, boar, wolf from trail cameras

### Augmentation Pipeline
1. Mosaic (4-image combination)
2. Horizontal flip (p=0.5)
3. HSV color jitter
4. Low-light simulation (gamma)
5. Weather simulation (rain/fog/noise)
6. Letterbox resize to 640×640

---

## 📱 Mobile App Features

### 🏠 Dashboard
- Live MJPEG camera stream
- System health indicators
- Latest detection card
- Quick action buttons

### 🚨 Alerts
- Real-time animal alerts via WebSocket
- Filter by class, status, time
- One-tap resolve
- Confidence badges

### 🗺 Map View
- OpenStreetMap integration
- Animal detection pins
- Device location markers

### 🎛 Control Panel
- Individual toggles: Siren, Flash, Ultrasonic
- Duration control (5s / 10s / 30s / 60s)
- Quick sequence buttons (Boar Alert, Wolf Alert, All Off)
- Multi-device support

### 📊 Analytics
- Detection count by day/week/month
- Pie chart: class distribution
- Bar chart: device activity
- Progress bars per class

### 🤖 Sahayak AI
Multilingual assistant supporting:
- **Hindi**: "Siren chalu karo", "Kya detect hua?"
- **Marathi**: "सायरन चालू करा", "काय आढळले?"
- **English**: "Turn on siren", "What was detected?"

### ⚙ Settings
- Backend URL configuration
- Push notification toggle
- Detection sensitivity slider
- Language selection
- Dark/Light mode

---

## 🔌 API Reference

### POST `/detect`
Receive detection from edge device.
```json
{
  "class_name": "boar",
  "confidence": 0.87,
  "bbox": [0.2, 0.3, 0.6, 0.8],
  "timestamp": "2024-01-15T10:30:00Z",
  "device_id": "pi_001",
  "location": {"lat": 19.076, "lng": 72.877}
}
```

### GET `/alerts`
```
GET /alerts?limit=50&resolved=false&class_name=wolf&hours=24
```

### POST `/control`
```json
{
  "device_id": "pi_001",
  "action": "siren_on",
  "params": {"duration": 10}
}
```

### WebSocket `/ws`
Real-time detection events pushed to connected mobile apps.

---

## ⚡ Power Budget (~15W)

| Component      | Current | Power  |
|----------------|---------|--------|
| Raspberry Pi 4 | 1.2A    | 6.0W   |
| Pi Camera      | 0.25A   | 1.25W  |
| WiFi (active)  | 0.15A   | 0.75W  |
| Arduino + PIR  | 0.10A   | 0.5W   |
| **Total**      | **1.7A**| **8.5W** |

Power savings: INT8 model (-60% compute), PIR-triggered capture, HDMI disabled.

---

## 🌐 Offline Capability

When network is unavailable:
1. Detections queued in `/var/guardian_ai/offline_queue/`
2. LoRa radio sends SMS-like alerts
3. GSM fallback via SIM800L module
4. Auto-sync when connection restored

---

## 🔮 Future Scope (Federated Learning)

```
Phase 1 (Current): Centralized model, edge inference
Phase 2: Federated Learning
  - Each Pi trains on local data (privacy-preserving)
  - Gradients sent to central server (not raw images)
  - Global model updated and distributed
  - Improves accuracy for local wildlife variations

Phase 3: Edge AI Cloud
  - Per-farm customized models
  - Species-specific alert thresholds
  - Multi-farm threat correlation
```

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| No camera detected | `v4l2-ctl --list-devices` |
| Arduino not found | Check `/dev/ttyUSB0`, `ls /dev/tty*` |
| Model too slow | Use INT8 ONNX, reduce input to 320×320 |
| Pi overheating | Add heatsink, lower FPS to 15 |
| No network | Check backend URL in Settings |
| MongoDB errors | `docker-compose restart mongo` |

---

## 📜 License
MIT License - Guardian AI Team

## 🙏 Credits
- YOLOv8 by Ultralytics
- FastAPI by Sebastián Ramírez
- Flutter by Google
- OpenVINO by Intel
- COCO Dataset by Microsoft
