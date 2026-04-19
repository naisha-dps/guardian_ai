"""
Guardian AI - WebSocket Connection Manager
============================================
Manages multiple concurrent WebSocket connections.
Supports: broadcast, per-device channels, client tracking.
"""

import uuid
import logging
import asyncio
from typing import Dict, Optional, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections from mobile app clients.
    
    Features:
      - Multi-client broadcast
      - Per-device channels (only Pi's device clients get its commands)
      - Graceful disconnect handling
      - Message queuing for slow clients
    """

    def __init__(self):
        # All active connections: client_id → WebSocket
        self.connections: Dict[str, WebSocket] = {}

        # Channel subscriptions: channel_name → set of client_ids
        self.channels: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, channels: list = None) -> str:
        """
        Accept a new WebSocket connection.
        Returns: unique client_id
        """
        await websocket.accept()
        client_id = str(uuid.uuid4())[:8]
        self.connections[client_id] = websocket

        # Subscribe to default + any custom channels
        default_channels = ["all", "alerts"]
        for ch in (channels or []) + default_channels:
            if ch not in self.channels:
                self.channels[ch] = set()
            self.channels[ch].add(client_id)

        logger.info(f"WS client {client_id} connected. Total: {len(self.connections)}")
        return client_id

    def disconnect(self, client_id: str):
        """Remove client from all channels."""
        self.connections.pop(client_id, None)
        for subscribers in self.channels.values():
            subscribers.discard(client_id)
        logger.info(f"WS client {client_id} disconnected. Total: {len(self.connections)}")

    def count(self) -> int:
        """Number of active connections."""
        return len(self.connections)

    async def broadcast(self, message: dict, channel: str = "all"):
        """
        Send message to all clients in a channel.
        Skips disconnected clients gracefully.
        """
        disconnected = []
        targets = self.channels.get(channel, set()).copy()

        # "all" channel: send to every connected client
        if channel == "all":
            targets = set(self.connections.keys())

        for client_id in targets:
            ws = self.connections.get(client_id)
            if ws is None:
                disconnected.append(client_id)
                continue
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up dead connections
        for cid in disconnected:
            self.disconnect(cid)

    async def send_to(self, client_id: str, message: dict):
        """Send message to a specific client."""
        ws = self.connections.get(client_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)
