from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WebhookPayload:
    event_type: str
    data: dict[str, Any]
    timestamp: float
    delivery_id: str


def _load_webhook_secret() -> str:
    """Load webhook signing secret from environment or generate a random one.

    Never hardcodes a default secret. Falls back to a cryptographically
    random token if WEBHOOK_SECRET is not set.
    """
    secret = os.getenv("WEBHOOK_SECRET")
    if not secret:
        secret = secrets.token_urlsafe(32)
        logger.warning(
            "WEBHOOK_SECRET not set; using randomly generated secret. "
            "Set WEBHOOK_SECRET in production for stable webhook signatures."
        )
    return secret


class WebhookDispatcher:
    """Delivers security alerts and detection events via HMAC-SHA256 signed HTTP POST webhooks."""

    def __init__(self, secret: str | None = None, max_retries: int = 3) -> None:
        self.secret = (secret or _load_webhook_secret()).encode("utf-8")
        self.max_retries = max_retries
        self.dead_letter_queue: list[dict[str, Any]] = []

    def compute_signature(self, payload_bytes: bytes) -> str:
        """HMAC-SHA256 signature for payload verification."""
        return hmac.new(self.secret, payload_bytes, hashlib.sha256).hexdigest()

    async def dispatch(self, target_url: str, event_type: str, data: dict[str, Any]) -> bool:
        payload = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
            "delivery_id": f"dlv_{int(time.time() * 1000)}",
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        signature = self.compute_signature(body_bytes)
        delivery_id = str(payload["delivery_id"])

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-SmartVision-Signature": signature,
            "X-SmartVision-Delivery": delivery_id,
        }

        async with httpx.AsyncClient(timeout=3.0) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = await client.post(target_url, content=body_bytes, headers=headers)
                    if resp.status_code in (200, 201, 202, 204):
                        return True
                except httpx.InvalidURL as exc:
                    logger.debug("Webhook target URL is invalid: %s", exc)
                    break
                except httpx.HTTPError as exc:
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
