"""
Guardian AI - API Data Models (Pydantic)
=========================================
Defines request/response schemas for all endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class DetectionEvent(BaseModel):
    """Incoming detection from Raspberry Pi edge device."""
    class_name: str = Field(..., description="Animal class: deer/boar/wolf/cattle/dog")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="[x1,y1,x2,y2] normalized 0-1")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    device_id: str = Field(..., description="Unique edge device ID")
    frame_id: Optional[int] = None
    location: Optional[Dict[str, float]] = Field(
        None, description="{'lat': ..., 'lng': ...}"
    )
    image_b64: Optional[str] = Field(
        None, description="Base64 encoded snapshot (optional)"
    )

    @validator("class_name")
    def validate_class(cls, v):
        valid = {"deer", "boar", "wolf", "cattle", "dog"}
        if v not in valid:
            raise ValueError(f"class_name must be one of {valid}")
        return v


class Alert(BaseModel):
    """High-confidence detection alert."""
    detection_id: Optional[str] = None
    class_name: str
    confidence: float
    timestamp: str
    device_id: str
    location: Optional[Dict[str, float]] = None
    resolved: bool = False
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    notes: Optional[str] = None


class ControlCommand(BaseModel):
    """Control command to send to edge device."""
    device_id: str
    action: str = Field(
        ...,
        description="Action: siren_on/siren_off/flash_on/flash_off/"
                    "ultrasonic_on/ultrasonic_off/camera_start/camera_stop"
    )
    params: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[int] = Field(
        None, description="Auto-stop after N seconds (optional)"
    )

    @validator("action")
    def validate_action(cls, v):
        valid_actions = {
            "siren_on", "siren_off",
            "flash_on", "flash_off",
            "ultrasonic_on", "ultrasonic_off",
            "camera_start", "camera_stop",
            "reboot", "status",
        }
        if v not in valid_actions:
            raise ValueError(f"action must be one of {valid_actions}")
        return v


class DeviceStatus(BaseModel):
    """Edge device registration and status."""
    device_id: str
    name: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    firmware_version: Optional[str] = None
    model_version: Optional[str] = None
    battery_level: Optional[float] = None
    cpu_temp: Optional[float] = None
    online: bool = True
    last_seen: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HistoryFilter(BaseModel):
    """Filters for detection history query."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    class_name: Optional[str] = None
    device_id: Optional[str] = None
    min_confidence: float = 0.0
    limit: int = 100
    page: int = 1
