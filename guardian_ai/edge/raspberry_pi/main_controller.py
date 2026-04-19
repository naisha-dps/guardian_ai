"""
Guardian AI - Raspberry Pi Main Controller
==========================================
Dual-layer edge system:
  Layer 1: Raspberry Pi (Python) - ML inference, camera, WiFi/LoRa
  Layer 2: Arduino (C++) - Sensors, actuators via Serial

PIR sensor → Camera ON → YOLOv8 inference → Actuators + API

Hardware connections:
  Pi GPIO 17  ← PIR sensor OUT
  Pi GPIO 18  → LED status indicator
  Pi USB      ↔ Arduino (Serial)
  Pi CSI/USB  ← Pi Camera / USB webcam
"""

import os
import sys
import time
import json
import serial
import logging
import threading
import requests
import RPi.GPIO as GPIO  # type: ignore  # Only available on Pi
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.inference.inference_pipeline import GuardianInference, INFERENCE_CONFIG

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/var/log/guardian_ai.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("guardian_edge")


# ─── Hardware Configuration ───────────────────────────────────────────────────

GPIO_CONFIG = {
    "PIR_PIN": 17,          # GPIO 17 = physical pin 11
    "LED_PIN": 18,          # Status LED
    "ARDUINO_PORT": "/dev/ttyUSB0",
    "ARDUINO_BAUD": 9600,
}

EDGE_CONFIG = {
    "MODEL_PATH": "/opt/guardian_ai/models/guardian_wildlife_int8.onnx",
    "BACKEND_URL": os.getenv("BACKEND_URL", "http://192.168.1.100:8000"),
    "DEVICE_ID": os.getenv("DEVICE_ID", "pi_001"),
    "CAMERA_SOURCE": 0,
    "PIR_COOLDOWN": 30,      # Seconds between PIR triggers
    "HEARTBEAT_INTERVAL": 30, # Seconds between heartbeat sends
    "OFFLINE_MODE": False,    # True = don't require network
}


# ─── GPIO Setup ───────────────────────────────────────────────────────────────

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(GPIO_CONFIG["PIR_PIN"], GPIO.IN)
    GPIO.setup(GPIO_CONFIG["LED_PIN"], GPIO.OUT)
    GPIO.output(GPIO_CONFIG["LED_PIN"], GPIO.LOW)
    logger.info("GPIO initialized")


def cleanup_gpio():
    GPIO.output(GPIO_CONFIG["LED_PIN"], GPIO.LOW)
    GPIO.cleanup()


def blink_led(times: int = 3, interval: float = 0.2):
    """Blink status LED to indicate detection."""
    for _ in range(times):
        GPIO.output(GPIO_CONFIG["LED_PIN"], GPIO.HIGH)
        time.sleep(interval)
        GPIO.output(GPIO_CONFIG["LED_PIN"], GPIO.LOW)
        time.sleep(interval)


# ─── Arduino Serial Communication ─────────────────────────────────────────────

class ArduinoController:
    """
    Communicates with Arduino via Serial to control actuators:
      - Siren (buzzer/horn)
      - Flash light (high-power LED)
      - Ultrasonic deterrent (40kHz ultrasonic emitter)
    
    Protocol: JSON over Serial
    Pi → Arduino: {"cmd": "siren", "state": 1, "duration": 5}
    Arduino → Pi: {"status": "ok", "sensor": "pir", "value": 1}
    """

    def __init__(self):
        self.ser = None
        self.connected = False
        self._reader_thread = None

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                GPIO_CONFIG["ARDUINO_PORT"],
                GPIO_CONFIG["ARDUINO_BAUD"],
                timeout=1
            )
            time.sleep(2)  # Arduino reset delay
            self.connected = True
            logger.info(f"Arduino connected on {GPIO_CONFIG['ARDUINO_PORT']}")

            # Start reader thread
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._reader_thread.start()
            return True

        except serial.SerialException as e:
            logger.warning(f"Arduino not available: {e}")
            self.connected = False
            return False

    def send_command(self, cmd: str, state: int, duration: int = 0):
        """Send actuator command to Arduino."""
        if not self.connected or not self.ser:
            logger.warning(f"[Arduino SIM] {cmd}={state} duration={duration}s")
            return

        payload = json.dumps({"cmd": cmd, "state": state, "duration": duration})
        self.ser.write((payload + "\n").encode())
        logger.info(f"→ Arduino: {payload}")

    def _read_loop(self):
        """Background thread: read status messages from Arduino."""
        while self.connected and self.ser:
            try:
                line = self.ser.readline().decode().strip()
                if line:
                    try:
                        data = json.loads(line)
                        logger.debug(f"← Arduino: {data}")
                    except json.JSONDecodeError:
                        logger.debug(f"← Arduino raw: {line}")
            except Exception as e:
                logger.error(f"Arduino read error: {e}")
                break

    def siren(self, on: bool, duration: int = 10):
        self.send_command("siren", 1 if on else 0, duration)

    def flash(self, on: bool, duration: int = 5):
        self.send_command("flash", 1 if on else 0, duration)

    def ultrasonic(self, on: bool, duration: int = 15):
        self.send_command("ultrasonic", 1 if on else 0, duration)

    def deterrent_sequence(self, animal_class: str):
        """
        Run a deterrent sequence based on animal type.
        Different animals respond to different deterrents.
        """
        sequences = {
            "deer":   [("flash", 3), ("ultrasonic", 10)],
            "boar":   [("siren", 5), ("flash", 5), ("ultrasonic", 15)],
            "wolf":   [("siren", 10), ("flash", 5)],
            "cattle": [("flash", 3)],
            "dog":    [("ultrasonic", 8)],
        }
        seq = sequences.get(animal_class, [("siren", 5)])
        for actuator, duration in seq:
            getattr(self, actuator)(True, duration)
            logger.info(f"Deterrent: {actuator} for {duration}s")


# ─── Heartbeat ────────────────────────────────────────────────────────────────

class HeartbeatService:
    """Sends periodic status pings to backend."""

    def __init__(self, backend_url: str, device_id: str):
        self.backend_url = backend_url
        self.device_id = device_id
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while True:
            try:
                import psutil
                cpu_temp = None
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp") as f:
                        cpu_temp = int(f.read()) / 1000.0
                except Exception:
                    pass

                status = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                    "cpu_temp": cpu_temp,
                }
                requests.patch(
                    f"{self.backend_url}/devices/{self.device_id}/heartbeat",
                    json=status,
                    timeout=5,
                )
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

            time.sleep(EDGE_CONFIG["HEARTBEAT_INTERVAL"])


# ─── Main Guardian Loop ───────────────────────────────────────────────────────

class GuardianEdge:
    """
    Main edge device controller.
    
    State machine:
      IDLE → PIR trigger → DETECTING → [Detection found] → ACTING → IDLE
                        ↓ [No detection]                  
                        → IDLE
    """

    def __init__(self):
        self.arduino = ArduinoController()
        self.inference = None  # Lazy-loaded when PIR triggers
        self.last_pir_trigger = 0
        self.running = False

        # Register command listener (for remote control from app)
        self._command_thread = threading.Thread(
            target=self._poll_commands, daemon=True
        )

    def start(self):
        """Initialize hardware and start main loop."""
        logger.info("=" * 50)
        logger.info("Guardian AI Edge Controller Starting")
        logger.info("=" * 50)

        # Setup hardware
        setup_gpio()
        self.arduino.connect()

        # Register device with backend
        self._register_device()

        # Start heartbeat
        heartbeat = HeartbeatService(
            EDGE_CONFIG["BACKEND_URL"],
            EDGE_CONFIG["DEVICE_ID"],
        )
        heartbeat.start()

        # Load inference engine
        logger.info("Loading ML model...")
        self.inference = GuardianInference(
            EDGE_CONFIG["MODEL_PATH"],
            camera_source=EDGE_CONFIG["CAMERA_SOURCE"],
        )
        logger.info("Model loaded. Waiting for PIR trigger...")

        # Start command polling
        self._command_thread.start()

        # Main loop
        self.running = True
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.running = False
            cleanup_gpio()

    def _main_loop(self):
        """Monitor PIR sensor and trigger detection pipeline."""
        while self.running:
            pir_state = GPIO.input(GPIO_CONFIG["PIR_PIN"])

            if pir_state == GPIO.HIGH:
                now = time.time()
                # Respect cooldown period
                if now - self.last_pir_trigger >= EDGE_CONFIG["PIR_COOLDOWN"]:
                    self.last_pir_trigger = now
                    logger.info("🔴 PIR Motion Detected! Starting detection...")
                    blink_led(3)
                    self._handle_detection_trigger()

            time.sleep(0.05)  # 50ms polling = 20Hz

    def _handle_detection_trigger(self):
        """
        Triggered by PIR: capture frame, run inference, act.
        """
        import cv2
        import numpy as np

        # Capture a frame
        cap = None
        try:
            cap = cv2.VideoCapture(EDGE_CONFIG["CAMERA_SOURCE"])
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                return

            # Run inference
            detections, latency_ms = self.inference.model.infer(frame)
            logger.info(f"Inference: {len(detections)} objects in {latency_ms:.1f}ms")

            for det in detections:
                logger.info(
                    f"  → {det['class_name']} ({det['confidence']:.2f})"
                )

                # Send to backend
                self._send_detection(det, frame)

                # Trigger deterrents
                if det["confidence"] >= 0.60:
                    self.arduino.deterrent_sequence(det["class_name"])
                    blink_led(5, 0.1)

        finally:
            if cap:
                cap.release()

    def _send_detection(self, det: dict, frame=None):
        """Send detection event to backend API."""
        import base64
        import cv2

        payload = {
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "timestamp": datetime.utcnow().isoformat(),
            "device_id": EDGE_CONFIG["DEVICE_ID"],
            "location": self._get_location(),
        }

        # Optionally attach base64 snapshot
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            payload["image_b64"] = base64.b64encode(buf).decode()

        try:
            resp = requests.post(
                f"{EDGE_CONFIG['BACKEND_URL']}/detect",
                json=payload,
                timeout=10,
            )
            logger.info(f"Detection sent: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning("Network unavailable - detection queued locally")
            self._save_offline(payload)

    def _save_offline(self, payload: dict):
        """Save detection locally when network unavailable."""
        offline_dir = Path("/var/guardian_ai/offline_queue")
        offline_dir.mkdir(parents=True, exist_ok=True)
        fname = offline_dir / f"det_{int(time.time())}.json"
        with open(fname, "w") as f:
            json.dump(payload, f)

    def _get_location(self):
        """
        Get GPS coordinates if GPS module attached.
        Returns None if not available.
        """
        # TODO: Replace with real GPS module (e.g., Neo-6M via serial)
        return {"lat": 19.0760, "lng": 72.8777}  # Default: Mumbai

    def _register_device(self):
        """Register this device with the backend."""
        try:
            requests.post(
                f"{EDGE_CONFIG['BACKEND_URL']}/devices/register",
                json={
                    "device_id": EDGE_CONFIG["DEVICE_ID"],
                    "name": "Guardian Pi Unit 1",
                    "online": True,
                    "firmware_version": "1.0.0",
                },
                timeout=5,
            )
            logger.info("Device registered with backend")
        except Exception as e:
            logger.warning(f"Could not register device: {e}")

    def _poll_commands(self):
        """
        Poll backend for control commands.
        In production, use WebSocket for push-based commands.
        """
        import websocket  # pip install websocket-client

        ws_url = EDGE_CONFIG["BACKEND_URL"].replace("http", "ws") + "/ws"

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if data.get("type") == "control":
                    device_id = data.get("device_id")
                    if device_id in (EDGE_CONFIG["DEVICE_ID"], "all"):
                        action = data.get("action", "")
                        logger.info(f"Remote command received: {action}")
                        self._execute_command(action, data.get("params", {}))
            except Exception as e:
                logger.error(f"Command parse error: {e}")

        def on_error(ws, error):
            logger.debug(f"WS error: {error}")

        def on_close(ws, *args):
            logger.debug("WS connection closed, retrying in 10s...")
            time.sleep(10)

        while self.running:
            try:
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                )
                ws.run_forever()
            except Exception:
                time.sleep(10)

    def _execute_command(self, action: str, params: dict):
        """Execute a remote control command."""
        duration = params.get("duration", 10)
        if action == "siren_on":
            self.arduino.siren(True, duration)
        elif action == "siren_off":
            self.arduino.siren(False)
        elif action == "flash_on":
            self.arduino.flash(True, duration)
        elif action == "flash_off":
            self.arduino.flash(False)
        elif action == "ultrasonic_on":
            self.arduino.ultrasonic(True, duration)
        elif action == "ultrasonic_off":
            self.arduino.ultrasonic(False)
        elif action == "reboot":
            logger.info("Rebooting...")
            os.system("sudo reboot")
        elif action == "status":
            self._register_device()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    controller = GuardianEdge()
    controller.start()
