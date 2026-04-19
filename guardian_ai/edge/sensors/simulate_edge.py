"""
Guardian AI - Edge Sensor Simulator
=====================================
Simulates Raspberry Pi + Arduino hardware for testing
the complete system WITHOUT physical hardware.

Sends synthetic detections to the backend API at random intervals.
Useful for:
  - Backend development and testing
  - Mobile app UI testing
  - Load testing
  - CI/CD pipelines

Usage:
  python edge/sensors/simulate_edge.py --backend http://localhost:8000
"""

import time
import random
import json
import requests
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SIM] %(message)s")
logger = logging.getLogger("guardian_sim")

# ─── Simulation Configuration ─────────────────────────────────────────────────

CLASSES = ["deer", "boar", "wolf", "cattle", "dog"]
DEVICE_IDS = ["pi_001", "pi_002", "pi_003"]

# Probability of each animal appearing (weighted - deer most common)
CLASS_WEIGHTS = [0.35, 0.25, 0.10, 0.20, 0.10]

# Farm locations (lat/lng) - simulating multiple farm fields
FARM_LOCATIONS = [
    {"lat": 19.0760, "lng": 72.8777, "name": "North Field"},
    {"lat": 19.0820, "lng": 72.8830, "name": "South Field"},
    {"lat": 19.0700, "lng": 72.8900, "name": "East Field"},
]

SIM_CONFIG = {
    "min_interval": 5,      # Min seconds between detections
    "max_interval": 30,     # Max seconds between detections
    "min_confidence": 0.55,
    "max_confidence": 0.98,
    "burst_probability": 0.15,  # Probability of burst (multiple animals at once)
    "burst_count": (2, 4),      # Range of burst detection count
}


# ─── Synthetic Detection Generator ───────────────────────────────────────────

def generate_detection(device_id: str = None) -> Dict:
    """Generate a realistic-looking synthetic detection."""
    cls = random.choices(CLASSES, weights=CLASS_WEIGHTS, k=1)[0]
    conf = random.uniform(SIM_CONFIG["min_confidence"], SIM_CONFIG["max_confidence"])

    # High-confidence wolf/boar triggers more urgent alerts
    if cls in ("wolf", "boar"):
        conf = max(conf, 0.70)

    # Bounding box: realistic sizes for each animal
    bbox_sizes = {
        "deer":   (0.08, 0.15, 0.12, 0.25),
        "boar":   (0.10, 0.10, 0.15, 0.15),
        "wolf":   (0.08, 0.08, 0.12, 0.12),
        "cattle": (0.15, 0.20, 0.25, 0.35),
        "dog":    (0.05, 0.06, 0.10, 0.12),
    }
    bw_min, bh_min, bw_max, bh_max = bbox_sizes.get(cls, (0.1, 0.1, 0.2, 0.2))
    bw = random.uniform(bw_min, bw_max)
    bh = random.uniform(bh_min, bh_max)
    cx = random.uniform(bw / 2 + 0.05, 1 - bw / 2 - 0.05)
    cy = random.uniform(bh / 2 + 0.05, 1 - bh / 2 - 0.05)

    # Location: pick a farm field
    loc = random.choice(FARM_LOCATIONS)
    # Add small random offset to simulate movement
    lat = loc["lat"] + random.uniform(-0.001, 0.001)
    lng = loc["lng"] + random.uniform(-0.001, 0.001)

    dev = device_id or random.choice(DEVICE_IDS)

    return {
        "class_name": cls,
        "confidence": round(conf, 3),
        "bbox": [
            round(cx - bw / 2, 4),
            round(cy - bh / 2, 4),
            round(cx + bw / 2, 4),
            round(cy + bh / 2, 4),
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "device_id": dev,
        "frame_id": random.randint(0, 10000),
        "location": {"lat": round(lat, 6), "lng": round(lng, 6)},
    }


# ─── Detection Sender ─────────────────────────────────────────────────────────

def send_detection(backend_url: str, detection: Dict) -> bool:
    """POST a detection event to the backend."""
    try:
        resp = requests.post(
            f"{backend_url}/detect",
            json=detection,
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            alert_flag = "🚨" if result.get("alert_created") else "✓"
            logger.info(
                f"{alert_flag} Sent: {detection['class_name']:8s} "
                f"conf={detection['confidence']:.2f} "
                f"dev={detection['device_id']}"
            )
            return True
        else:
            logger.warning(f"Server error: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to backend: {backend_url}")
        return False
    except Exception as e:
        logger.error(f"Send error: {e}")
        return False


# ─── Register Simulated Devices ──────────────────────────────────────────────

def register_devices(backend_url: str):
    """Register simulated Pi devices with the backend."""
    for device_id in DEVICE_IDS:
        try:
            requests.post(
                f"{backend_url}/devices/register",
                json={
                    "device_id": device_id,
                    "name": f"Simulated Guardian Pi ({device_id})",
                    "online": True,
                    "firmware_version": "1.0.0-sim",
                    "model_version": "yolov8n-int8-v1",
                    "location": random.choice(FARM_LOCATIONS),
                },
                timeout=5,
            )
        except Exception:
            pass
    logger.info(f"Registered {len(DEVICE_IDS)} simulated devices")


# ─── Heartbeat Simulation ────────────────────────────────────────────────────

def send_heartbeats(backend_url: str):
    """Simulate device heartbeats."""
    import threading

    def hb_loop():
        while True:
            for device_id in DEVICE_IDS:
                try:
                    requests.patch(
                        f"{backend_url}/devices/{device_id}/heartbeat",
                        json={
                            "cpu_percent": random.uniform(20, 60),
                            "memory_percent": random.uniform(30, 70),
                            "cpu_temp": random.uniform(45, 65),
                        },
                        timeout=3,
                    )
                except Exception:
                    pass
            time.sleep(30)

    t = threading.Thread(target=hb_loop, daemon=True)
    t.start()


# ─── Main Simulation Loop ─────────────────────────────────────────────────────

def run_simulation(
    backend_url: str,
    duration_minutes: float = None,
    detections_per_minute: float = 4.0,
):
    """
    Run the edge simulation loop.
    
    Args:
        backend_url: Backend API URL
        duration_minutes: Run for N minutes (None = forever)
        detections_per_minute: Average detection rate
    """
    logger.info("=" * 55)
    logger.info("  Guardian AI - Edge Simulator")
    logger.info("=" * 55)
    logger.info(f"  Backend : {backend_url}")
    logger.info(f"  Devices : {', '.join(DEVICE_IDS)}")
    logger.info(f"  Rate    : ~{detections_per_minute} detections/min")
    logger.info(f"  Duration: {'Forever' if duration_minutes is None else f'{duration_minutes} min'}")
    logger.info("")

    # Register devices
    register_devices(backend_url)
    send_heartbeats(backend_url)

    start_time = time.time()
    total_sent = 0
    total_alerts = 0

    base_interval = 60.0 / detections_per_minute

    try:
        while True:
            # Check duration limit
            if duration_minutes and (time.time() - start_time) > duration_minutes * 60:
                logger.info(f"Duration limit reached. Sent {total_sent} detections.")
                break

            # Burst: simulate multiple animals appearing at once
            if random.random() < SIM_CONFIG["burst_probability"]:
                burst_n = random.randint(*SIM_CONFIG["burst_count"])
                logger.info(f"  [BURST] Sending {burst_n} detections at once")
                for _ in range(burst_n):
                    det = generate_detection()
                    success = send_detection(backend_url, det)
                    if success:
                        total_sent += 1
                        if det["confidence"] >= 0.60:
                            total_alerts += 1
                    time.sleep(0.3)
            else:
                # Single detection
                det = generate_detection()
                success = send_detection(backend_url, det)
                if success:
                    total_sent += 1

            # Wait for next detection
            interval = random.uniform(
                base_interval * 0.5,
                base_interval * 1.5,
            )
            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info(f"\nSimulation stopped by user.")

    logger.info(f"\n[Summary] Total sent: {total_sent} | Alerts created: {total_alerts}")


# ─── Historical Data Generator ────────────────────────────────────────────────

def generate_historical_data(backend_url: str, days: int = 7):
    """
    Generate historical detection data for the past N days.
    Useful for populating analytics charts in the mobile app.
    """
    logger.info(f"Generating {days} days of historical data...")

    total = 0
    now = datetime.utcnow()

    for day_offset in range(days, 0, -1):
        # More detections at dawn/dusk (animals are most active)
        base_date = now - timedelta(days=day_offset)

        # Generate 10-40 detections per day
        daily_count = random.randint(10, 40)

        for _ in range(daily_count):
            # Peak hours: 5-7am and 6-8pm
            hour = random.choices(
                range(24),
                weights=[
                    3, 2, 2, 2, 3,   # 0-4
                    8, 10, 6, 4, 3,  # 5-9
                    3, 3, 3, 3, 3,   # 10-14
                    3, 4, 6, 8, 10,  # 15-19
                    7, 5, 4, 3,      # 20-23
                ],
                k=1,
            )[0]
            minute = random.randint(0, 59)
            timestamp = base_date.replace(
                hour=hour, minute=minute, second=random.randint(0, 59),
                microsecond=0,
            )

            det = generate_detection()
            det["timestamp"] = timestamp.isoformat()

            send_detection(backend_url, det)
            total += 1

        logger.info(f"Day {day_offset}/{days}: {daily_count} detections generated")
        time.sleep(0.1)  # Don't hammer the server

    logger.info(f"\n[✓] Historical data generated: {total} total detections over {days} days")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian AI Edge Simulator")
    parser.add_argument(
        "--backend",
        default="http://localhost:8000",
        help="Backend API URL",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "historical", "both"],
        default="both",
        help="Simulation mode",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run duration in minutes (default: forever)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=4.0,
        help="Detections per minute (default: 4.0)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of historical data to generate",
    )
    args = parser.parse_args()

    if args.mode in ("historical", "both"):
        generate_historical_data(args.backend, days=args.days)

    if args.mode in ("live", "both"):
        run_simulation(
            args.backend,
            duration_minutes=args.duration,
            detections_per_minute=args.rate,
        )
