from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Manages active streaming WebSocket subscribers with room channels and heartbeat ping."""

    def __init__(self) -> None:
        self._active_connections: set[WebSocket] = set()
        self._channel_subscriptions: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str = "telemetry") -> None:
        await websocket.accept()
        self._active_connections.add(websocket)
        if channel not in self._channel_subscriptions:
            self._channel_subscriptions[channel] = set()
        self._channel_subscriptions[channel].add(websocket)
        logger.debug("WebSocket client connected to channel '%s'. Total: %d", channel, len(self._active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._active_connections.discard(websocket)
        for clients in self._channel_subscriptions.values():
            clients.discard(websocket)

    async def broadcast_json(self, data: dict, channel: str = "telemetry") -> None:
        clients = self._channel_subscriptions.get(channel, set())
        if not clients:
            return

        message = json.dumps(data)
        stale_clients = []
        for client in list(clients):
            try:
                await client.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                stale_clients.append(client)

        for stale in stale_clients:
            self.disconnect(stale)

    @property
    def total_connections(self) -> int:
        return len(self._active_connections)


ws_manager = WebSocketConnectionManager()
