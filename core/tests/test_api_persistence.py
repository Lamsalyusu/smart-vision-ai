import os
import unittest
from unittest.mock import patch

import httpx

from core.infrastructure.api.main import app
from core.infrastructure.database.data_manager import DataManager, data_manager
from core.infrastructure.database.engine import SessionLocal, _init_engine
from core.infrastructure.database.models import Session as DatabaseSession


class ApiPersistenceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

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
        password = "SecurePassword123!"

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
            stored_session = db.query(DatabaseSession).filter_by(token=token).first()
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
