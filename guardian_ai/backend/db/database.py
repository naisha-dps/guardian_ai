"""
Guardian AI - Database Layer (MongoDB via Motor)
=================================================
Async MongoDB operations for all Guardian AI data.
Uses Motor (async MongoDB driver) for FastAPI compatibility.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from bson import ObjectId

import motor.motor_asyncio
from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "guardian_ai")


def serialize_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string for JSON serialization."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


class Database:
    """
    Manages MongoDB connection and all CRUD operations.
    Collections:
      - detections : All raw detection events
      - alerts     : High-confidence alert records
      - commands   : Control command log
      - devices    : Edge device registry
    """

    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        """Establish MongoDB connection and create indexes."""
        self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        await self._create_indexes()
        logger.info(f"Connected to MongoDB: {MONGO_URI}/{DB_NAME}")

    async def disconnect(self):
        if self.client:
            self.client.close()

    async def ping(self) -> str:
        try:
            await self.db.command("ping")
            return "ok"
        except Exception as e:
            return f"error: {e}"

    async def _create_indexes(self):
        """Create indexes for query performance."""
        # Detections: sorted by time, class, device
        await self.db.detections.create_index([
            ("timestamp", DESCENDING),
            ("class_name", ASCENDING),
            ("device_id", ASCENDING),
        ])
        await self.db.detections.create_index("device_id")

        # Alerts: time + resolved status
        await self.db.alerts.create_index([
            ("timestamp", DESCENDING),
            ("resolved", ASCENDING),
        ])

        # TTL: auto-delete raw detections after 90 days
        await self.db.detections.create_index(
            "timestamp",
            expireAfterSeconds=90 * 24 * 3600,
        )

        logger.info("Database indexes created")

    # ─── Detections ───────────────────────────────────────────────────────────

    async def save_detection(self, event: dict) -> str:
        """Insert detection event, return its ID."""
        event["created_at"] = datetime.utcnow()
        result = await self.db.detections.insert_one(event)
        return str(result.inserted_id)

    async def get_history(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        class_name: Optional[str],
        device_id: Optional[str],
        limit: int,
        skip: int,
    ) -> List[dict]:
        """Query detection history with filters and pagination."""
        query = {}

        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        if class_name:
            query["class_name"] = class_name

        if device_id:
            query["device_id"] = device_id

        cursor = self.db.detections.find(query).sort(
            "timestamp", DESCENDING
        ).skip(skip).limit(limit)

        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def get_statistics(self, days: int = 7) -> dict:
        """
        Aggregate detection statistics for analytics dashboard.
        Returns:
          - total_detections
          - by_class: {class: count}
          - by_day: [{date, count}]
          - by_hour: [{hour, count}] (peak hours)
          - top_device: most active device
        """
        since = datetime.utcnow() - timedelta(days=days)
        since_str = since.isoformat()

        # Total count
        total = await self.db.detections.count_documents(
            {"timestamp": {"$gte": since_str}}
        )

        # By class
        by_class_pipeline = [
            {"$match": {"timestamp": {"$gte": since_str}}},
            {"$group": {"_id": "$class_name", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
        ]
        by_class_cursor = self.db.detections.aggregate(by_class_pipeline)
        by_class = {doc["_id"]: doc["count"] async for doc in by_class_cursor}

        # By device
        by_device_pipeline = [
            {"$match": {"timestamp": {"$gte": since_str}}},
            {"$group": {"_id": "$device_id", "count": {"$sum": 1}}},
            {"$sort": {"count": DESCENDING}},
            {"$limit": 5},
        ]
        by_device_cursor = self.db.detections.aggregate(by_device_pipeline)
        by_device = [
            {"device_id": d["_id"], "count": d["count"]}
            async for d in by_device_cursor
        ]

        return {
            "period_days": days,
            "total_detections": total,
            "by_class": by_class,
            "by_device": by_device,
        }

    # ─── Alerts ───────────────────────────────────────────────────────────────

    async def save_alert(self, alert: dict) -> str:
        alert["created_at"] = datetime.utcnow()
        result = await self.db.alerts.insert_one(alert)
        return str(result.inserted_id)

    async def get_alerts(
        self,
        since: datetime,
        limit: int,
        resolved: Optional[bool] = None,
        class_name: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> List[dict]:
        query = {"timestamp": {"$gte": since.isoformat()}}
        if resolved is not None:
            query["resolved"] = resolved
        if class_name:
            query["class_name"] = class_name
        if device_id:
            query["device_id"] = device_id

        cursor = self.db.alerts.find(query).sort("timestamp", DESCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [serialize_doc(d) for d in docs]

    async def resolve_alert(self, alert_id: str) -> bool:
        result = await self.db.alerts.update_one(
            {"_id": ObjectId(alert_id)},
            {"$set": {"resolved": True, "resolved_at": datetime.utcnow().isoformat()}},
        )
        return result.modified_count > 0

    # ─── Control Commands ─────────────────────────────────────────────────────

    async def save_command(self, cmd: dict) -> str:
        cmd["created_at"] = datetime.utcnow()
        result = await self.db.commands.insert_one(cmd)
        return str(result.inserted_id)

    # ─── Devices ──────────────────────────────────────────────────────────────

    async def upsert_device(self, device: dict):
        device["updated_at"] = datetime.utcnow()
        await self.db.devices.update_one(
            {"device_id": device["device_id"]},
            {"$set": device},
            upsert=True,
        )

    async def get_devices(self) -> List[dict]:
        cursor = self.db.devices.find({})
        docs = await cursor.to_list(length=100)
        return [serialize_doc(d) for d in docs]

    async def update_device_heartbeat(self, device_id: str, status: dict):
        await self.db.devices.update_one(
            {"device_id": device_id},
            {"$set": {
                "online": True,
                "last_seen": datetime.utcnow().isoformat(),
                **{f"status.{k}": v for k, v in status.items()},
            }},
            upsert=True,
        )
