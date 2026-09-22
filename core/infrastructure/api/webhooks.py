from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WebhookPayload:
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    delivery_id: str


class WebhookDispatcher:
    """Delivers security alerts and detection events via HMAC-SHA256 signed HTTP POST webhooks."""

    def __init__(self, secret: str = "webhook_default_hmac_secret", max_retries: int = 3) -> None:
        self.secret = secret.encode("utf-8")
        self.max_retries = max_retries
        self.dead_letter_queue: List[Dict[str, Any]] = []

    def compute_signature(self, payload_bytes: bytes) -> str:
        """HMAC-SHA256 signature for payload verification."""
        return hmac.new(self.secret, payload_bytes, hashlib.sha256).hexdigest()

    async def dispatch(self, target_url: str, event_type: str, data: Dict[str, Any]) -> bool:
        payload = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
            "delivery_id": f"dlv_{int(time.time() * 1000)}",
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        signature = self.compute_signature(body_bytes)

        headers = {
            "Content-Type": "application/json",
            "X-SmartVision-Signature": signature,
            "X-SmartVision-Delivery": payload["delivery_id"],
        }

        async with httpx.AsyncClient(timeout=3.0) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(target_url, content=body_bytes, headers=headers)
                    if resp.status_code in (200, 201, 202, 204):
                        return True
                except Exception as exc:
                    logger.debug("Webhook delivery attempt %d failed: %s", attempt, exc)

                await asyncio.sleep(0.05 * (2 ** attempt))

        # Failed after retries: append to Dead Letter Queue (DLQ)
        self.dead_letter_queue.append({
            "target_url": target_url,
            "payload": payload,
            "failed_at": time.time(),
        })
        return False


webhook_dispatcher = WebhookDispatcher()
