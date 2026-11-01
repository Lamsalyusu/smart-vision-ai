import time
import unittest
from unittest.mock import patch

from core.infrastructure.security.jwt_auth import JWTService
from core.infrastructure.security.rbac import (
    ROLE_PERMISSIONS_MAP,
    AppPermission,
    AppRole,
)


class SecurityRbacJwtTests(unittest.TestCase):
    def setUp(self):
        self.jwt = JWTService(
            secret_key="test_super_secure_secret_key_1234567890",
            access_token_ttl_seconds=1.0,  # short ttl for expiry test
        )

    def test_jwt_create_and_verify(self):
        token = self.jwt.create_token(user_id="usr_001", username="alice", role="analyst")
        self.assertIsInstance(token, str)
        self.assertEqual(len(token.split(".")), 3)

        payload = self.jwt.verify_token(token)
        self.assertEqual(payload.sub, "usr_001")
        self.assertEqual(payload.username, "alice")
        self.assertEqual(payload.role, "analyst")

    def test_jwt_requires_configured_secret_in_production(self):
        with (
            patch.dict("os.environ", {"NODE_ENV": "production"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "SECRET_KEY must be configured"),
        ):
            JWTService()

    def test_development_jwt_secret_is_ephemeral_and_long_enough(self):
        with patch.dict("os.environ", {}, clear=True):
            first = JWTService()
            second = JWTService()

        self.assertGreaterEqual(len(first.secret), 32)
        self.assertNotEqual(first.secret, second.secret)

    def test_jwt_rejects_short_configured_secret(self):
        with self.assertRaisesRegex(ValueError, "at least 32 bytes"):
            JWTService(secret_key="too-short")

    def test_jwt_tampering_rejected(self):
        token = self.jwt.create_token(user_id="usr_002", username="bob", role="viewer")
        parts = token.split(".")
        # Tamper payload
        tampered_token = f"{parts[0]}.eyJob2dlIjoidmFsIn0.{parts[2]}"
        with self.assertRaises(ValueError):
            self.jwt.verify_token(tampered_token)

    def test_jwt_revocation(self):
        token = self.jwt.create_token(user_id="usr_003", username="charlie", role="operator")
        self.assertIsNotNone(self.jwt.verify_token(token))

        self.jwt.revoke_token(token)
        with self.assertRaises(ValueError):
            self.jwt.verify_token(token)

    def test_jwt_expiration(self):
        token = self.jwt.create_token(user_id="usr_004", username="david", role="viewer")
        time.sleep(1.1)  # wait for TTL 1.0s to expire
        with self.assertRaises(ValueError):
            self.jwt.verify_token(token)

    def test_rbac_permissions_hierarchy(self):
        # Admin has all permissions including ADMIN_USERS
        admin_perms = ROLE_PERMISSIONS_MAP[AppRole.ADMIN]
        self.assertIn(AppPermission.ADMIN_USERS, admin_perms)
        self.assertIn(AppPermission.MANAGE_ZONES, admin_perms)

        # Analyst can view telemetry and manage zones but not admin users
        analyst_perms = ROLE_PERMISSIONS_MAP[AppRole.ANALYST]
        self.assertIn(AppPermission.VIEW_TELEMETRY, analyst_perms)
        self.assertNotIn(AppPermission.ADMIN_USERS, analyst_perms)

        # Viewer can only view telemetry
        viewer_perms = ROLE_PERMISSIONS_MAP[AppRole.VIEWER]
        self.assertIn(AppPermission.VIEW_TELEMETRY, viewer_perms)
        self.assertNotIn(AppPermission.MANAGE_ZONES, viewer_perms)


if __name__ == "__main__":
    unittest.main()
