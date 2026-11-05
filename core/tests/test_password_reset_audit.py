import os
import re
import unittest
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

import httpx

from core.infrastructure.api.main import app
from core.infrastructure.database.data_manager import data_manager
from core.infrastructure.database.engine import SessionLocal
from core.infrastructure.database.models import PasswordResetToken
from core.infrastructure.security.jwt_auth import jwt_service

PASSWORD = "original-password-1"


class PasswordResetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict(os.environ, {
            "INTERNAL_API_TOKEN": "reset-test-token",
            "EMAIL_HOST_USER": "sender@example.com",
            "EMAIL_HOST_PASSWORD": "smtp-password",
            "APP_BASE_URL": "https://app.example.com",
        })
        self.env.start()
        self.sent: list[tuple[str, str, str]] = []
        self.mailer = patch(
            "core.infrastructure.api.main.send_email",
            side_effect=lambda to, subject, body: self.sent.append((to, subject, body)),
        )
        self.mailer.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"X-Internal-Token": "reset-test-token"},
        )
        self.username = f"reset_{uuid4().hex[:8]}"
        self.email = f"{self.username}@example.com"
        signup = await self.client.post("/signup", json={
            "full_name": "Reset User", "email": self.email,
            "username": self.username, "password": PASSWORD,
        })
        self.assertEqual(signup.status_code, 200)
        self.session_token = signup.json()["token"]

    async def asyncTearDown(self):
        await self.client.aclose()
        self.mailer.stop()
        self.env.stop()

    def _token_from_email(self) -> str:
        self.assertEqual(len(self.sent), 1)
        to, _subject, body = self.sent[0]
        self.assertEqual(to, self.email)
        match = re.search(r"https://app\.example\.com/reset-password\?token=(\S+)", body)
        assert match is not None, body
        return match.group(1)

    async def _login(self, password: str) -> int:
        res = await self.client.post("/login", json={"email_or_username": self.username, "password": password})
        return res.status_code

    async def test_full_reset_flow(self):
        res = await self.client.post("/password-reset/request", json={"email": self.email.upper()})
        self.assertEqual(res.status_code, 200)
        token = self._token_from_email()

        confirm = await self.client.post(
            "/password-reset/confirm", json={"token": token, "new_password": "brand-new-password"}
        )
        self.assertEqual(confirm.status_code, 200)
        self.assertEqual(await self._login("brand-new-password"), 200)
        self.assertEqual(await self._login(PASSWORD), 401)

        # Existing sessions are revoked and the token cannot be reused.
        session = await self.client.get("/session", headers={"X-Session-Token": self.session_token})
        self.assertEqual(session.status_code, 401)
        again = await self.client.post(
            "/password-reset/confirm", json={"token": token, "new_password": "another-password"}
        )
        self.assertEqual(again.status_code, 400)

    async def test_unknown_email_gets_the_same_response_and_no_mail(self):
        res = await self.client.post("/password-reset/request", json={"email": f"nobody_{uuid4().hex}@example.com"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"success": True})
        self.assertEqual(self.sent, [])

    async def test_username_does_not_trigger_reset(self):
        await self.client.post("/password-reset/request", json={"email": self.username})
        self.assertEqual(self.sent, [])

    async def test_requests_are_throttled(self):
        await self.client.post("/password-reset/request", json={"email": self.email})
        await self.client.post("/password-reset/request", json={"email": self.email})
        self.assertEqual(len(self.sent), 1)

    async def test_expired_and_bogus_tokens_are_rejected(self):
        await self.client.post("/password-reset/request", json={"email": self.email})
        token = self._token_from_email()
        with SessionLocal() as db:
            record = db.query(PasswordResetToken).filter_by(
                token_hash=data_manager.hash_session_token(token)
            ).one()
            record.expires_at = record.created_at - timedelta(seconds=1)
            db.commit()
        for candidate in (token, "x" * 43):
            with self.subTest(token=candidate[:6]):
                res = await self.client.post(
                    "/password-reset/confirm", json={"token": candidate, "new_password": "brand-new-password"}
                )
                self.assertEqual(res.status_code, 400)
        self.assertEqual(await self._login(PASSWORD), 200)

    async def test_token_is_stored_hashed(self):
        await self.client.post("/password-reset/request", json={"email": self.email})
        token = self._token_from_email()
        with SessionLocal() as db:
            self.assertIsNone(db.get(PasswordResetToken, token))

    async def test_request_reports_missing_email_configuration(self):
        with patch.dict(os.environ, {"EMAIL_HOST_USER": ""}):
            res = await self.client.post("/password-reset/request", json={"email": self.email})
        self.assertEqual(res.status_code, 503)

    async def test_audit_log_records_auth_events(self):
        await self._login("wrong-password")
        await self._login(PASSWORD)
        admin_jwt = jwt_service.create_token("usr_admin", "admin", "admin")
        res = await self.client.get(
            "/admin/audit-logs", headers={"Authorization": f"Bearer {admin_jwt}"}
        )
        self.assertEqual(res.status_code, 200)
        user_id = data_manager.find_user_by_identifier(self.username).id
        mine = [e["event_type"] for e in res.json()["events"] if e["user_id"] == user_id]
        self.assertIn("USER_REGISTERED", mine)
        self.assertIn("LOGIN_FAILURE", mine)
        self.assertIn("LOGIN_SUCCESS", mine)

        operator_jwt = jwt_service.create_token("usr_op", "op", "operator")
        denied = await self.client.get(
            "/admin/audit-logs", headers={"Authorization": f"Bearer {operator_jwt}"}
        )
        self.assertEqual(denied.status_code, 403)


if __name__ == "__main__":
    unittest.main()
