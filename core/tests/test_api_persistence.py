import os
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from core.infrastructure.api.main import app
from core.infrastructure.database.data_manager import DataManager, data_manager
from core.infrastructure.database.engine import SessionLocal, _init_engine
from core.infrastructure.database.models import Session as DatabaseSession


class ApiPersistenceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.internal_token_patch = patch.dict(os.environ, {"INTERNAL_API_TOKEN": "test-internal-token"})
        self.internal_token_patch.start()
        self.transport = httpx.ASGITransport(app=app)
        self.internal_headers = {"X-Internal-Token": "test-internal-token"}
        self.client = httpx.AsyncClient(
            transport=self.transport, base_url="http://test", headers=self.internal_headers
        )
        self.anonymous_client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.anonymous_client.aclose()
        self.internal_token_patch.stop()

    async def test_01_health_endpoint(self):
        res = await self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")

    async def test_liveness_probe_returns_timestamp(self):
        res = await self.client.get("/health/live")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "alive")
        self.assertIsInstance(res.json().get("timestamp"), (int, float))

    async def test_readiness_probe_returns_service_status(self):
        res = await self.client.get("/health/ready")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("status"), "ready")

    async def test_02_signup_login_session_flow(self):
        email = f"int_test_{id(self)}@smartvision.ai"
        username = f"intuser_{id(self)}"
        password = "test-password-do-not-use-in-production"

        # 1. Signup
        signup_res = await self.client.post(
            "/signup",
            json={
                "full_name": "Integration Test User",
                "email": email,
                "username": username,
                "password": password,
                "role": "admin",
            },
        )
        self.assertEqual(signup_res.status_code, 200)
        self.assertTrue(signup_res.json().get("success"))

        # 2. Login
        login_res = await self.client.post(
            "/login",
            json={"email_or_username": email, "password": password},
        )
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertTrue(login_data.get("success"))
        self.assertEqual(login_data["user"]["email"], email)
        self.assertEqual(login_data["user"]["role"], "operator")

        # Cookie was set
        self.assertIn("session_token", login_res.cookies)
        token = login_res.cookies["session_token"]

        # 3. Session validation
        session_res = await self.client.get("/session", cookies={"session_token": token})
        self.assertEqual(session_res.status_code, 200)
        self.assertEqual(session_res.json()["user"]["username"], username)

        with SessionLocal() as db:
            # Only the SHA-256 digest of the token is stored.
            self.assertIsNone(db.query(DatabaseSession).filter_by(token=token).first())
            stored_session = (
                db.query(DatabaseSession)
                .filter_by(token=DataManager.hash_session_token(token))
                .first()
            )
            self.assertIsNotNone(stored_session)
            stored_session.expires_at = None
            db.commit()

        missing_expiry_res = await self.client.get(
            "/session", cookies={"session_token": token}
        )
        self.assertEqual(missing_expiry_res.status_code, 401)

        # 4. Logout
        logout_res = await self.client.post("/logout", cookies={"session_token": token})
        self.assertEqual(logout_res.status_code, 200)

        # Session revoked
        revoked_res = await self.client.get("/session", cookies={"session_token": token})
        self.assertEqual(revoked_res.status_code, 401)

    async def test_03_crop_and_telemetry_persistence(self):
        # 1. Save crop
        crop_res = await self.client.post(
            "/crops",
            json={
                "object_class": "knife",
                "confidence": 0.94,
                "distance_m": 1.25,
                "has_crop": True,
                "payload": {"zone": "restricted_area_A"},
            },
        )
        self.assertEqual(crop_res.status_code, 200)
        crop_data = crop_res.json()
        self.assertTrue(crop_data["success"])
        self.assertEqual(crop_data["crop"]["object_class"], "knife")

        # 2. Retrieve latest crop
        latest_res = await self.client.get("/crops/latest", params={"object_class": "knife"})
        self.assertEqual(latest_res.status_code, 200)
        latest_data = latest_res.json()
        self.assertIsNotNone(latest_data["crop"])
        self.assertEqual(latest_data["crop"]["object_class"], "knife")

        # 3. Record detection
        det_res = await self.client.post(
            "/detections/batch",
            json={
                "detections": [
                    {
                        "class_name": "helmet",
                        "confidence": 0.88,
                        "distance_m": 2.1,
                        "has_crop": False,
                    },
                    {
                        "class_name": "can",
                        "confidence": 0.76,
                        "has_crop": False,
                    },
                ]
            },
        )
        self.assertEqual(det_res.status_code, 200)
        self.assertEqual(det_res.json()["count"], 2)

        # 4. Check stats
        stats_res = await self.client.get("/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats = stats_res.json()
        self.assertGreater(stats["total_detections"], 0)
        self.assertIn("class_distribution", stats)
        self.assertEqual(stats["class_distribution"].get("helmet"), 1)
        self.assertEqual(stats["class_distribution"].get("can"), 1)
        self.assertEqual(DataManager().get_stats()["total_detections"], stats["total_detections"])

        # 5. Clear history
        clear_res = await self.client.post("/history/clear")
        self.assertEqual(clear_res.status_code, 200)
        self.assertEqual(DataManager().get_stats()["total_detections"], 0)

    async def test_login_lockout_persists_across_manager_instances(self):
        identifier = f"lockout_{id(self)}"
        for _ in range(5):
            result = data_manager.record_failed_login(identifier)

        self.assertTrue(result["locked"])
        self.assertTrue(DataManager().is_account_locked(identifier))
        DataManager().clear_failed_logins(identifier)
        self.assertFalse(data_manager.is_account_locked(identifier))

    async def test_replay_frames_persist_are_session_scoped_and_can_be_cleared(self):
        first_session = str(uuid4())
        second_session = str(uuid4())
        payload = {
            "captured_at": 1_798_000_000_000,
            "detections": [
                {
                    "class": "can",
                    "confidence": 0.81,
                    "bbox": [1, 2, 30, 40],
                    "track_id": 9,
                    "spatial_3d": {"distance_m": 2.4},
                }
            ],
        }

        first_save = await self.client.post(
            "/replay/frames",
            json={"session_id": first_session, **payload},
            headers=self.internal_headers,
        )
        second_save = await self.client.post(
            "/replay/frames",
            json={"session_id": second_session, "captured_at": payload["captured_at"], "detections": []},
            headers=self.internal_headers,
        )
        self.assertEqual(first_save.status_code, 200)
        self.assertEqual(second_save.status_code, 200)

        loaded = await self.client.get(
            "/replay/frames", params={"session_id": first_session}, headers=self.internal_headers
        )
        self.assertEqual(loaded.status_code, 200)
        expected_frame = {
            "capturedAt": float(payload["captured_at"]),
            "detections": [{**payload["detections"][0], "bbox": [1.0, 2.0, 30.0, 40.0]}],
        }
        self.assertEqual(loaded.json()["frames"], [expected_frame])
        self.assertEqual(DataManager().get_replay_frames(first_session), [expected_frame])

        cleared = await self.client.delete(
            "/replay/frames", params={"session_id": first_session}, headers=self.internal_headers
        )
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(
            (await self.client.get("/replay/frames", params={"session_id": first_session}, headers=self.internal_headers)).json()["frames"],
            [],
        )
        self.assertEqual(
            len((await self.client.get("/replay/frames", params={"session_id": second_session}, headers=self.internal_headers)).json()["frames"]),
            1,
        )

    async def test_replay_api_rejects_missing_internal_token(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INTERNAL_API_TOKEN", None)
            response = await self.client.get("/replay/frames", params={"session_id": str(uuid4())})

        self.assertEqual(response.status_code, 503)

    async def test_replay_storage_keeps_only_the_latest_frames_per_session(self):
        session_id = str(uuid4())
        manager = DataManager()
        for captured_at in (1.0, 2.0, 3.0):
            manager.save_replay_frame(
                session_id,
                captured_at,
                [],
                max_frames=2,
            )

        self.assertEqual(
            [frame["capturedAt"] for frame in manager.get_replay_frames(session_id)],
            [2.0, 3.0],
        )


class ApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.internal_token_patch = patch.dict(os.environ, {"INTERNAL_API_TOKEN": "test-internal-token"})
        self.internal_token_patch.start()
        transport = httpx.ASGITransport(app=app)
        self.headers = {"X-Internal-Token": "test-internal-token"}
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test", headers=self.headers)
        self.anonymous = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.anonymous.aclose()
        self.internal_token_patch.stop()

    async def _signup(self, username: str, password: str = "correct-horse-battery") -> httpx.Response:
        return await self.client.post(
            "/signup",
            json={
                "full_name": "Security Test",
                "email": f"{username.lower()}@example.com",
                "username": username,
                "password": password,
            },
        )

    async def test_endpoints_reject_requests_without_internal_token(self):
        calls = [
            ("POST", "/login", {"email_or_username": "x", "password": "y"}),
            ("POST", "/signup", {"full_name": "a", "email": "a@b.co", "username": "abc", "password": "12345678"}),
            ("GET", "/session", None),
            ("POST", "/crops", {"object_class": "can", "confidence": 0.5}),
            ("GET", "/crops/latest", None),
            ("POST", "/detections/batch", {"detections": []}),
            ("GET", "/stats", None),
            ("POST", "/history/clear", None),
            ("GET", "/metrics", None),
            ("POST", "/webhooks/dispatch", {"target_url": "https://example.com", "event_type": "x", "data": {}}),
            ("POST", "/mesh/reconstruct", {}),
        ]
        for method, path, body in calls:
            with self.subTest(path=path):
                response = await self.anonymous.request(method, path, json=body)
                self.assertEqual(response.status_code, 401)

        wrong = await self.anonymous.get("/stats", headers={"X-Internal-Token": "wrong"})
        self.assertEqual(wrong.status_code, 401)

    async def test_health_probes_stay_public(self):
        for path in ("/health", "/health/live", "/health/ready"):
            with self.subTest(path=path):
                self.assertEqual((await self.anonymous.get(path)).status_code, 200)

    async def test_signup_returns_session_and_ignores_requested_role(self):
        username = f"Sec_{uuid4().hex[:8]}"
        response = await self.client.post(
            "/signup",
            json={
                "full_name": "Role Escalation",
                "email": f"{username.lower()}@example.com",
                "username": username,
                "password": "correct-horse-battery",
                "role": "admin",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user"]["role"], "operator")
        session = await self.client.get("/session", headers={"X-Session-Token": body["token"]})
        self.assertEqual(session.status_code, 200)
        self.assertEqual(session.json()["user"]["username"], username)

    async def test_signup_validates_input(self):
        bad_payloads = [
            {"full_name": "a", "email": "not-an-email", "username": "valid_name", "password": "12345678"},
            {"full_name": "a", "email": "a@b.co", "username": "ab", "password": "12345678"},
            {"full_name": "a", "email": "a@b.co", "username": "valid_name", "password": "short"},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.assertEqual((await self.client.post("/signup", json=payload)).status_code, 422)

    async def test_duplicate_signup_is_rejected_case_insensitively(self):
        username = f"Dup_{uuid4().hex[:8]}"
        self.assertEqual((await self._signup(username)).status_code, 200)
        again = await self.client.post(
            "/signup",
            json={
                "full_name": "Dup",
                "email": f"other_{username.lower()}@example.com",
                "username": username.upper(),
                "password": "correct-horse-battery",
            },
        )
        self.assertEqual(again.status_code, 400)

    async def test_login_is_case_insensitive_and_wrong_password_is_401(self):
        username = f"Mixed_{uuid4().hex[:8]}"
        await self._signup(username)
        ok = await self.client.post(
            "/login", json={"email_or_username": username.lower(), "password": "correct-horse-battery"}
        )
        self.assertEqual(ok.status_code, 200)
        bad = await self.client.post(
            "/login", json={"email_or_username": username, "password": "wrong-password"}
        )
        self.assertEqual(bad.status_code, 401)
        unknown = await self.client.post(
            "/login", json={"email_or_username": f"nobody_{uuid4().hex}", "password": "whatever1"}
        )
        self.assertEqual(unknown.status_code, 401)

    async def test_session_tokens_are_unpredictable(self):
        username = f"Tok_{uuid4().hex[:8]}"
        await self._signup(username)
        tokens = set()
        for _ in range(3):
            response = await self.client.post(
                "/login", json={"email_or_username": username, "password": "correct-horse-battery"}
            )
            tokens.add(response.json()["token"])
        self.assertEqual(len(tokens), 3)
        for token in tokens:
            self.assertGreaterEqual(len(token), 40)

    async def test_crops_saved_in_the_same_second_do_not_collide(self):
        for _ in range(3):
            response = await self.client.post("/crops", json={"object_class": "pen", "confidence": 0.7})
            self.assertEqual(response.status_code, 200)

    async def test_webhook_rejects_internal_targets(self):
        for url in ("http://127.0.0.1:8000/history/clear", "http://169.254.169.254/latest/meta-data", "ftp://example.com"):
            with self.subTest(url=url):
                response = await self.client.post(
                    "/webhooks/dispatch", json={"target_url": url, "event_type": "test", "data": {}}
                )
                self.assertEqual(response.status_code, 400)

    async def test_metrics_report_only_measured_values(self):
        body = (await self.client.get("/metrics")).text
        self.assertIn("smart_vision_total_detections", body)
        self.assertNotIn("tracking_efficiency", body)

    async def test_mesh_rejects_malformed_points(self):
        response = await self.client.post("/mesh/reconstruct", json={"points": [[0, 0], [1, 1], [2, 2], [3, 3]]})
        self.assertEqual(response.status_code, 422)


class ProductionDatabaseConfigurationTests(unittest.TestCase):
    def test_production_requires_database_configuration(self):
        with (
            patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "production DATABASE_URL"),
        ):
            _init_engine()

    def test_production_does_not_fall_back_when_database_is_unavailable(self):
        with (
            patch.dict(
                os.environ,
                {"NODE_ENV": "production", "DATABASE_URL": "postgresql://db/app"},
                clear=True,
            ),
            patch("core.infrastructure.database.engine.create_engine", side_effect=OSError("unavailable")),
            self.assertRaisesRegex(RuntimeError, "production database is unavailable"),
        ):
            _init_engine()


if __name__ == "__main__":
    unittest.main()
