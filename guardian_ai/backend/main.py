"""
Guardian AI - FastAPI Backend
==============================
Production-ready backend with:
  - /detect     : Receive detection events from edge device
  - /alerts     : Get recent alerts
  - /history    : Detection history with filters
  - /control    : Send commands to edge device
  - WebSocket   : Real-time push to mobile app
  - MongoDB     : Persistent storage
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware

# Assuming your app is named 'app', paste this right below it:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # The '*' means "allow anyone, including localhost"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from db.database import Database
from websocket.manager import WebSocketManager
from api.models import (
    DetectionEvent,
    Alert,
    ControlCommand,
    DeviceStatus,
    HistoryFilter,
)
from utils.notifications import NotificationService
from utils.lora_sim import LoRaSimulator
# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("guardian_ai")


# ─── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("🦌 Guardian AI Backend starting...")
    await db.connect()
    logger.info("✓ Database connected")
    yield
    await db.disconnect()
    logger.info("Guardian AI Backend stopped")


# ─── Initialize Services ──────────────────────────────────────────────────────

db = Database()
ws_manager = WebSocketManager()
notifier = NotificationService()
lora = LoRaSimulator()

app = FastAPI(
    title="Guardian AI API",
    description="Autonomous Wildlife Detection & Crop Protection System",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow mobile app and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Guardian AI",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "db": await db.ping(),
        "ws_connections": ws_manager.count(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── POST /detect ──────────────────────────────────────────────────────────────

@app.post("/detect", response_model=dict)
async def receive_detection(event: DetectionEvent):
    """
    Receives detection from edge device (Raspberry Pi).
    
    Flow:
      1. Save to DB
      2. Create alert if high-confidence animal detected
      3. Broadcast to all WebSocket clients (mobile app)
      4. Send push notification
      5. Trigger LoRa/SMS if configured
    """
    logger.info(
        f"Detection: {event.class_name} ({event.confidence:.2f}) "
        f"from device {event.device_id}"
    )

    # 1. Persist detection
    detection_id = await db.save_detection(event.dict())

    # 2. Create alert for high-confidence detections
    alert = None
    if event.confidence >= 0.60:
        alert = Alert(
            detection_id=detection_id,
            class_name=event.class_name,
            confidence=event.confidence,
            timestamp=event.timestamp,
            device_id=event.device_id,
            location=event.location,
            resolved=False,
        )
        await db.save_alert(alert.dict())

    # 3. Broadcast via WebSocket to mobile apps
    payload = {
        "type": "detection",
        "data": {
            **event.dict(),
            "detection_id": detection_id,
            "alert": alert.dict() if alert else None,
        }
    }
    await ws_manager.broadcast(payload)

    # 4. Push notification
    if alert:
        await notifier.send_push(
            title=f"🚨 {event.class_name.title()} Detected!",
            body=f"Confidence: {event.confidence:.0%} | Device: {event.device_id}",
            data={"detection_id": detection_id, "class": event.class_name},
        )

    # 5. LoRa/SMS simulation for remote areas
    if event.confidence >= 0.75:
        await lora.send_alert(event.class_name, event.device_id)

    return {
        "status": "ok",
        "detection_id": detection_id,
        "alert_created": alert is not None,
    }


# ─── GET /alerts ──────────────────────────────────────────────────────────────

@app.get("/alerts", response_model=List[dict])
async def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    resolved: Optional[bool] = Query(None),
    class_name: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=720),
):
    """
    Get recent alerts with optional filters.
    
    Args:
        limit: Max number of results
        resolved: Filter by resolved status
        class_name: Filter by animal class (deer, boar, etc.)
        device_id: Filter by specific device
        hours: Look back N hours (default: 24)
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    alerts = await db.get_alerts(
        since=since,
        limit=limit,
        resolved=resolved,
        class_name=class_name,
        device_id=device_id,
    )
    return alerts


@app.patch("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mark an alert as resolved."""
    success = await db.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    await ws_manager.broadcast({"type": "alert_resolved", "alert_id": alert_id})
    return {"status": "resolved"}


# ─── GET /history ─────────────────────────────────────────────────────────────

@app.get("/history", response_model=List[dict])
async def get_history(
    start_date: Optional[str] = Query(None, description="ISO date string"),
    end_date: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    page: int = Query(1, ge=1),
):
    """
    Paginated detection history with date range and class filters.
    Used for analytics graphs in mobile app.
    """
    history = await db.get_history(
        start_date=start_date,
        end_date=end_date,
        class_name=class_name,
        device_id=device_id,
        limit=limit,
        skip=(page - 1) * limit,
    )
    return history


@app.get("/history/stats")
async def get_stats(days: int = Query(7, ge=1, le=90)):
    """
    Aggregated detection statistics for analytics dashboard.
    Returns: daily counts, class distribution, peak hours.
    """
    stats = await db.get_statistics(days=days)
    return stats


# ─── POST /control ────────────────────────────────────────────────────────────

@app.post("/control")
async def send_control(cmd: ControlCommand):
    """
    Send control command to edge device.
    Commands: siren_on, siren_off, flash_on, flash_off,
              ultrasonic_on, ultrasonic_off, camera_start, camera_stop
              
    The command is broadcast via WebSocket; the Pi subscribes and acts on it.
    """
    logger.info(f"Control command: {cmd.action} → device {cmd.device_id}")

    # Save command log
    await db.save_command(cmd.dict())

    # Broadcast to device WebSocket channel
    payload = {
        "type": "control",
        "device_id": cmd.device_id,
        "action": cmd.action,
        "params": cmd.params or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    await ws_manager.broadcast(payload, channel=f"device_{cmd.device_id}")

    return {"status": "sent", "command": cmd.action}


# ─── GET /devices ─────────────────────────────────────────────────────────────

@app.get("/devices")
async def list_devices():
    """List all registered edge devices and their last-seen status."""
    devices = await db.get_devices()
    return devices


@app.post("/devices/register")
async def register_device(device: DeviceStatus):
    """Register or update an edge device."""
    await db.upsert_device(device.dict())
    return {"status": "registered", "device_id": device.device_id}


@app.patch("/devices/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, status: dict):
    """Called by Pi every 30s to report it's alive."""
    await db.update_device_heartbeat(device_id, status)
    return {"status": "ok"}


# ─── WebSocket /ws ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.
    Mobile app connects here to receive live detections.
    """
    client_id = await ws_manager.connect(websocket)
    logger.info(f"WebSocket client connected: {client_id}")

    try:
        # Send recent alerts on connect
        recent = await db.get_alerts(
            since=datetime.utcnow() - timedelta(hours=1),
            limit=10,
        )
        await websocket.send_json({
            "type": "init",
            "recent_alerts": recent,
            "client_id": client_id,
        })

        # Keep alive - listen for messages from app
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "control":
                # App can also send control commands via WS
                cmd = ControlCommand(**data.get("payload", {}))
                await send_control(cmd)

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        logger.info(f"WebSocket client disconnected: {client_id}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
