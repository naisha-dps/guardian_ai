"""
Guardian AI - Notification Service
=====================================
Handles:
  - Push notifications (FCM / Firebase)
  - SMS simulation
  - In-app notification formatting
"""

import os
import logging
import aiohttp
from typing import Optional

logger = logging.getLogger(__name__)

FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", "YOUR_FCM_KEY_HERE")
FCM_URL = "https://fcm.googleapis.com/fcm/send"


class NotificationService:
    """Sends push notifications via Firebase Cloud Messaging."""

    async def send_push(
        self,
        title: str,
        body: str,
        data: dict = None,
        topic: str = "guardian_alerts",
    ):
        """Send FCM push notification to all subscribed devices."""
        if FCM_SERVER_KEY == "YOUR_FCM_KEY_HERE":
            logger.debug(f"[PUSH SIMULATED] {title}: {body}")
            return

        payload = {
            "to": f"/topics/{topic}",
            "notification": {
                "title": title,
                "body": body,
                "sound": "alert",
                "badge": "1",
            },
            "data": data or {},
            "priority": "high",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    FCM_URL,
                    json=payload,
                    headers={
                        "Authorization": f"key={FCM_SERVER_KEY}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Push sent: {title}")
                    else:
                        logger.warning(f"Push failed: {resp.status}")
        except Exception as e:
            logger.error(f"Push notification error: {e}")
