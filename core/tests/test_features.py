"""Tests for frame telemetry, alerts, live broadcast, email verification,
administration endpoints, rate limits and database migrations."""

import os
import re
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from sqlalchemy import create_engine, inspect, text

from core.infrastructure.api import main as api
from core.infrastructure.api.main import app
from core.infrastructure.database.data_manager import data_manager
from core.infrastructure.database.engine import SessionLocal
from core.infrastructure.database.models import EmailVerification, User

TOKEN = "features-test-token"
HEADERS = {"X-Internal-Token": TOKEN}


class ApiTestCase(unittest.IsolatedAsyncioTestCase):
    env: dict[str, str] = {}

    async def asyncSetUp(self):
        self.env_patch = patch.dict(os.environ, {"INTERNAL_API_TOKEN": TOKEN, **self.env})
        self.env_patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=HEADERS
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.env_patch.stop()

    async def signup(self, password: str = "feature-password-1") -> tuple[str, httpx.Response]:
        username = f"f_{uuid4().hex[:10]}"
        response = await self.client.post("/signup", json={
            "full_name": "Feature Test", "email": f"{username}@example.com",
            "username": username, "password": password,
        })
        return username, response


class FrameTelemetryTests(ApiTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        data_manager.clear_history()

    async def test_frames_without_detections_are_counted_and_timed(self):
        timings = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0), (70.0, 80.0)]
        for inference, roundtrip in timings:
            res = await self.client.post("/detections/batch", json={
                "detections": [],
                "frame": {"inference_ms": inference, "total_ms": inference + 2, "roundtrip_ms": roundtrip},
            })
            self.assertEqual(res.status_code, 200)
        await self.client.post("/detections/batch", json={
            "detections": [{"class_name": "can", "confidence": 0.8}, {"class_name": "pen", "confidence": 0.6}],
            "frame": {"inference_ms": 90.0, "roundtrip_ms": 100.0},
        })

        perf = (await self.client.get("/stats")).json()["performance"]
        self.assertEqual(perf["sample_frames"], 5)
        self.assertEqual(perf["frames_in_window"], 5)
        self.assertEqual(perf["frames_per_minute"], 1.0)
        self.assertEqual(perf["inference_ms_p50"], 50.0)
        self.assertEqual(perf["roundtrip_ms_p50"], 60.0)
        self.assertAlmostEqual(perf["inference_ms_p95"], 86.0)
        self.assertEqual(perf["detections_per_frame"], 0.4)
        # Detections are still counted separately from frames.
        self.assertEqual((await self.client.get("/stats")).json()["total_detections"], 2)

    async def test_no_frames_reports_nulls_not_invented_values(self):
        perf = (await self.client.get("/stats")).json()["performance"]
        self.assertEqual(perf["sample_frames"], 0)
        self.assertIsNone(perf["inference_ms_p50"])
        self.assertIsNone(perf["detections_per_frame"])

    async def test_clear_history_removes_frame_timings(self):
        await self.client.post("/detections/batch", json={"detections": [], "frame": {"inference_ms": 5.0}})
        await self.client.post("/history/clear")
        self.assertEqual((await self.client.get("/stats")).json()["performance"]["sample_frames"], 0)

    async def test_invalid_timings_are_rejected(self):
        res = await self.client.post("/detections/batch", json={"detections": [], "frame": {"inference_ms": -1}})
        self.assertEqual(res.status_code, 422)


class AlertWebhookTests(ApiTestCase):
    env = {"ALERT_WEBHOOK_URL": "https://alerts.example.com/hook", "ALERT_CLASSES": "knife,fire"}

    async def asyncSetUp(self):
        await super().asyncSetUp()
        api._alert_last_sent.clear()
        self.dispatch = patch.object(api.webhook_dispatcher, "dispatch", new=AsyncMock(return_value=True))
        self.validate = patch.object(api, "_validate_webhook_target")
        self.dispatch_mock = self.dispatch.start()
        self.validate.start()

    async def asyncTearDown(self):
        self.dispatch.stop()
        self.validate.stop()
        api._alert_last_sent.clear()
        await super().asyncTearDown()

    async def post(self, *detections):
        return await self.client.post("/detections/batch", json={
            "detections": [{"class_name": c, "confidence": conf} for c, conf in detections],
        })

    async def test_watched_class_sends_one_alert_with_the_best_detection(self):
        res = await self.post(("knife", 0.7), ("knife", 0.9), ("can", 0.99))
        self.assertEqual(res.json()["alerts"], 1)
        self.dispatch_mock.assert_awaited_once()
        kwargs = self.dispatch_mock.await_args.kwargs
        self.assertEqual(kwargs["event_type"], "detection.alert")
        self.assertEqual(kwargs["data"], {"class_name": "knife", "confidence": 0.9, "distance_m": None, "has_crop": False})

    async def test_cooldown_and_thresholds(self):
        await self.post(("fire", 0.8))
        again = await self.post(("fire", 0.95))
        self.assertEqual(again.json()["alerts"], 0)
        low = await self.post(("knife", 0.3))
        self.assertEqual(low.json()["alerts"], 0)
        unwatched = await self.post(("helmet", 0.99))
        self.assertEqual(unwatched.json()["alerts"], 0)
        self.assertEqual(self.dispatch_mock.await_count, 1)

    async def test_no_alerts_without_configured_url(self):
        with patch.dict(os.environ, {"ALERT_WEBHOOK_URL": ""}):
            res = await self.post(("knife", 0.99))
        self.assertEqual(res.json()["alerts"], 0)
        self.dispatch_mock.assert_not_awaited()


class LiveBroadcastTests(ApiTestCase):
    async def test_batch_is_broadcast_to_subscribers(self):
        with patch.object(type(api.ws_manager), "total_connections", new=1), \
             patch.object(api.ws_manager, "broadcast_json", new=AsyncMock()) as broadcast:
            await self.client.post("/detections/batch", json={
                "detections": [{"class_name": "pen", "confidence": 0.5}],
                "frame": {"inference_ms": 12.5},
            })
        broadcast.assert_awaited_once()
        message = broadcast.await_args.args[0]
        self.assertEqual(message["type"], "detections")
        self.assertEqual(message["detections"][0]["class_name"], "pen")
        self.assertEqual(message["frame"]["inference_ms"], 12.5)

    async def test_subscribed_websocket_receives_detection_events(self):
        import json as jsonlib

        class FakeSocket:
            def __init__(self):
                self.accept = AsyncMock()
                self.send_text = AsyncMock()

        socket = FakeSocket()
        await api.ws_manager.connect(socket, channel="telemetry")
        try:
            await self.client.post("/detections/batch", json={
                "detections": [{"class_name": "can", "confidence": 0.7}],
            })
        finally:
            api.ws_manager.disconnect(socket)
        socket.send_text.assert_awaited_once()
        event = jsonlib.loads(socket.send_text.await_args.args[0])
        self.assertEqual(event["type"], "detections")
        self.assertEqual(event["detections"][0]["class_name"], "can")


class EmailVerificationTests(ApiTestCase):
    env = {
        "EMAIL_HOST_USER": "sender@example.com",
        "EMAIL_HOST_PASSWORD": "smtp-password",
        "APP_BASE_URL": "https://app.example.com",
    }

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.sent: list[tuple[str, str, str]] = []
        self.mailer = patch.object(api, "send_email", side_effect=lambda *a: self.sent.append(a))
        self.mailer.start()

    async def asyncTearDown(self):
        self.mailer.stop()
        await super().asyncTearDown()

    def token(self) -> str:
        bodies = [body for _to, subject, body in self.sent if subject.startswith("Verify")]
        self.assertEqual(len(bodies), 1)
        match = re.search(r"/verify-email\?token=(\S+)", bodies[0])
        assert match is not None
        return match.group(1)

    async def test_signup_sends_link_and_confirming_marks_verified(self):
        username, res = await self.signup()
        self.assertTrue(res.json()["verification_sent"])
        self.assertFalse(res.json()["user"]["email_verified"])
        token = self.token()

        self.assertEqual((await self.client.post("/email-verification/confirm", json={"token": token})).status_code, 200)
        session = await self.client.get("/session", headers={"X-Session-Token": res.json()["token"]})
        self.assertTrue(session.json()["user"]["email_verified"])
        # Single use.
        self.assertEqual((await self.client.post("/email-verification/confirm", json={"token": token})).status_code, 400)
        with SessionLocal() as db:
            record = db.get(EmailVerification, data_manager.find_user_by_identifier(username).id)
            self.assertIsNone(record.token_hash)

    async def test_resend_is_throttled_silent_for_unknown_and_skips_verified(self):
        username, _ = await self.signup()
        self.sent.clear()
        await self.client.post("/email-verification/resend", json={"email": f"{username}@example.com"})
        self.assertEqual(self.sent, [])  # within the one-minute throttle

        unknown = await self.client.post("/email-verification/resend", json={"email": "nobody@example.com"})
        self.assertEqual(unknown.json(), {"success": True})

        data_manager.mark_email_verified(data_manager.find_user_by_identifier(username).id)
        with SessionLocal() as db:
            db.query(EmailVerification).update({EmailVerification.created_at: EmailVerification.created_at})
            db.commit()
        await self.client.post("/email-verification/resend", json={"email": f"{username}@example.com"})
        self.assertEqual(self.sent, [])

    async def test_required_verification_blocks_login_until_confirmed(self):
        with patch.dict(os.environ, {"REQUIRE_EMAIL_VERIFICATION": "true"}):
            username, res = await self.signup()
            self.assertTrue(res.json()["verification_required"])
            self.assertNotIn("token", res.json())

            login = {"email_or_username": username, "password": "feature-password-1"}
            blocked = await self.client.post("/login", json=login)
            self.assertEqual(blocked.status_code, 403)
            wrong = await self.client.post("/login", json={**login, "password": "wrong-password"})
            self.assertEqual(wrong.status_code, 401)

            await self.client.post("/email-verification/confirm", json={"token": self.token()})
            self.assertEqual((await self.client.post("/login", json=login)).status_code, 200)

    async def test_required_verification_is_not_enforced_without_email(self):
        with patch.dict(os.environ, {"REQUIRE_EMAIL_VERIFICATION": "true", "EMAIL_HOST_USER": ""}):
            username, res = await self.signup()
            self.assertIn("token", res.json())
            login = await self.client.post(
                "/login", json={"email_or_username": username, "password": "feature-password-1"}
            )
            self.assertEqual(login.status_code, 200)


class AdminEndpointTests(ApiTestCase):
    async def test_list_change_role_and_verify(self):
        username, _ = await self.signup()
        users = (await self.client.get("/admin/users")).json()["users"]
        me = next(u for u in users if u["username"] == username)
        self.assertEqual(me["role"], "operator")
        self.assertFalse(me["email_verified"])

        res = await self.client.post(f"/admin/users/{me['id']}/role", json={"role": "analyst", "actor_id": "usr_admin"})
        self.assertEqual(res.json(), {"success": True, "previous": "operator", "role": "analyst"})
        await self.client.post(f"/admin/users/{me['id']}/verify-email", params={"actor_id": "usr_admin"})
        refreshed = next(u for u in (await self.client.get("/admin/users")).json()["users"] if u["id"] == me["id"])
        self.assertEqual(refreshed["role"], "analyst")
        self.assertTrue(refreshed["email_verified"])

        events = (await self.client.get("/admin/audit")).json()["events"]
        changed = next(e for e in events if e["event_type"] == "ROLE_CHANGED" and e["user_id"] == me["id"])
        self.assertEqual(changed["details"]["actor"], "usr_admin")

    async def test_unknown_user_and_invalid_role(self):
        self.assertEqual((await self.client.post("/admin/users/usr_missing/role", json={"role": "viewer"})).status_code, 404)
        self.assertEqual((await self.client.post("/admin/users/usr_missing/role", json={"role": "root"})).status_code, 422)
        self.assertEqual((await self.client.post("/admin/users/usr_missing/verify-email")).status_code, 404)

    async def test_last_admin_cannot_be_demoted(self):
        username, _ = await self.signup()
        user_id = data_manager.find_user_by_identifier(username).id
        with SessionLocal() as db:
            db.query(User).filter(User.role == "admin").update({User.role: "operator"})
            db.query(User).filter(User.id == user_id).update({User.role: "admin"})
            db.commit()
        res = await self.client.post(f"/admin/users/{user_id}/role", json={"role": "viewer"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("last administrator", res.json()["detail"])

    async def test_admin_endpoints_need_the_internal_token(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as anon:
            for path in ("/admin/users", "/admin/audit"):
                self.assertEqual((await anon.get(path)).status_code, 401)


class RateLimitTests(ApiTestCase):
    async def test_login_is_limited_per_client(self):
        def body():
            # A new account name each time, so the per-account lockout never applies.
            return {"email_or_username": f"nobody_{uuid4().hex}", "password": "x"}

        for _ in range(30):
            res = await self.client.post("/login", json=body(), headers={"X-Client-IP": "203.0.113.7"})
            self.assertEqual(res.status_code, 401)
        limited = await self.client.post("/login", json=body(), headers={"X-Client-IP": "203.0.113.7"})
        self.assertEqual(limited.status_code, 429)
        self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)
        other = await self.client.post("/login", json=body(), headers={"X-Client-IP": "203.0.113.8"})
        self.assertEqual(other.status_code, 401)

    async def test_signup_limit_and_switch(self):
        for _ in range(10):
            _, res = await self.signup()
            self.assertEqual(res.status_code, 200)
        _, limited = await self.signup()
        self.assertEqual(limited.status_code, 429)
        with patch.dict(os.environ, {"RATE_LIMITS_ENABLED": "false"}):
            _, allowed = await self.signup()
        self.assertEqual(allowed.status_code, 200)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.url = f"sqlite:///{self.tmp.name}/m.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_database_matches_models(self):
        from alembic.autogenerate import compare_metadata
        from alembic.migration import MigrationContext

        from core.infrastructure.database.engine import Base
        from core.infrastructure.database.migrate import upgrade

        engine = create_engine(self.url)
        upgrade(engine)
        with engine.connect() as conn:
            self.assertEqual(compare_metadata(MigrationContext.configure(conn), Base.metadata), [])
            self.assertEqual(conn.execute(text("select version_num from alembic_version")).scalar(), "0002_email_verifications")

    def test_database_created_before_migrations_is_upgraded_in_place(self):
        from core.infrastructure.database.migrate import upgrade

        engine = create_engine(self.url)
        with engine.begin() as conn:
            conn.execute(text("create table users (id varchar primary key, full_name varchar not null, "
                              "email varchar not null unique, username varchar not null unique, "
                              "password_hash varchar not null, auth_provider varchar, role varchar, created_at datetime)"))
            conn.execute(text("insert into users (id, full_name, email, username, password_hash) "
                              "values ('usr_old', 'Old', 'old@example.com', 'old', 'x')"))
        upgrade(engine)
        tables = set(inspect(engine).get_table_names())
        self.assertTrue({"users", "sessions", "email_verifications", "alembic_version"} <= tables)
        with engine.connect() as conn:
            self.assertEqual(conn.execute(text("select count(*) from users")).scalar(), 1)
        upgrade(engine)  # running again is a no-op


if __name__ == "__main__":
    unittest.main()
