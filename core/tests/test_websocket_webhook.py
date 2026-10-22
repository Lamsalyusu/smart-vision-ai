import hashlib
import hmac
import json
import unittest

from core.infrastructure.api.webhooks import WebhookDispatcher
from core.infrastructure.api.websocket_stream import WebSocketConnectionManager


class MockWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent_messages = []

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        self.sent_messages.append(text)


class WebSocketAndWebhookTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_connection_manager(self):
        manager = WebSocketConnectionManager()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, channel="alerts")
        await manager.connect(ws2, channel="alerts")
        self.assertEqual(manager.total_connections, 2)

        # Broadcast message
        payload = {"alert": "perimeter_breached", "zone": "vault"}
        await manager.broadcast_json(payload, channel="alerts")

        self.assertEqual(len(ws1.sent_messages), 1)
        self.assertEqual(json.loads(ws1.sent_messages[0])["alert"], "perimeter_breached")
        self.assertEqual(len(ws2.sent_messages), 1)

        # Disconnect ws1
        manager.disconnect(ws1)
        self.assertEqual(manager.total_connections, 1)

    def test_webhook_hmac_signature(self):
        dispatcher = WebhookDispatcher(secret="test-webhook-secret-do-not-use-in-production")
        data = b'{"event":"ZONE_BREACH","track_id":42}'

        sig = dispatcher.compute_signature(data)
        self.assertIsInstance(sig, str)

        # Manually compute expected HMAC-SHA256
        expected = hmac.new(b"test-webhook-secret-do-not-use-in-production", data, hashlib.sha256).hexdigest()
        self.assertEqual(sig, expected)

    async def test_invalid_webhook_url_is_added_to_dead_letter_queue(self):
        dispatcher = WebhookDispatcher(max_retries=2)

        delivered = await dispatcher.dispatch("http://[invalid", "test.event", {})

        self.assertFalse(delivered)
        self.assertEqual(len(dispatcher.dead_letter_queue), 1)
        self.assertEqual(dispatcher.dead_letter_queue[0]["target_url"], "http://[invalid")


if __name__ == "__main__":
    unittest.main()
