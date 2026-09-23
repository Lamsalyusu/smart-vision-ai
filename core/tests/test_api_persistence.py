import unittest
import httpx

from core.infrastructure.api.main import app
from core.infrastructure.database.data_manager import data_manager


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
                "role": "analyst",
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

        # Cookie was set
        self.assertIn("session_token", login_res.cookies)
        token = login_res.cookies["session_token"]

        # 3. Session validation
        session_res = await self.client.get("/session", cookies={"session_token": token})
        self.assertEqual(session_res.status_code, 200)
        self.assertEqual(session_res.json()["user"]["username"], username)

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
            "/detections",
            json={
                "class_name": "helmet",
                "confidence": 0.88,
                "distance_m": 2.1,
                "has_crop": False,
            },
        )
        self.assertEqual(det_res.status_code, 200)

        # 4. Check stats
        stats_res = await self.client.get("/stats")
        self.assertEqual(stats_res.status_code, 200)
        stats = stats_res.json()
        self.assertGreater(stats["total_detections"], 0)
        self.assertIn("class_distribution", stats)

        # 5. Clear history
        clear_res = await self.client.post("/history/clear")
        self.assertEqual(clear_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
