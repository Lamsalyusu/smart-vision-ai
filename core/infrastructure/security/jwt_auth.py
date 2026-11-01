from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenPayload:
    sub: str       # user_id
    username: str
    role: str
    exp: float
    iat: float
    iss: str = "smart-vision-ai"
    aud: str = "smart-vision-ai-api"


class JWTService:
    """Zero-external-dependency RFC 7519 HMAC-SHA256 JWT Engine with token revocation blocklist."""

    def __init__(
        self,
        secret_key: str | None = None,
        issuer: str = "smart-vision-ai",
        audience: str = "smart-vision-ai-api",
        access_token_ttl_seconds: float = 3600.0,      # 1 hour
        refresh_token_ttl_seconds: float = 604800.0,   # 7 days
    ) -> None:
        configured_secret = secret_key or os.getenv("SECRET_KEY")
        if not configured_secret:
            environment = os.getenv("APP_ENV", os.getenv("NODE_ENV", "development")).lower()
            if environment in {"production", "prod"}:
                raise RuntimeError("SECRET_KEY must be configured in production")
            configured_secret = secrets.token_urlsafe(32)

        self.secret = configured_secret.encode("utf-8")
        if len(self.secret) < 32:
            raise ValueError("SECRET_KEY must contain at least 32 bytes")
        self.issuer = issuer
        self.audience = audience
        self.access_ttl = access_token_ttl_seconds
        self.refresh_ttl = refresh_token_ttl_seconds
        self._revocation_blocklist: set[str] = set()

    @staticmethod
    def _b64_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    @staticmethod
    def _b64_decode(data_str: str) -> bytes:
        padding = "=" * ((4 - len(data_str) % 4) % 4)
        return base64.urlsafe_b64decode((data_str + padding).encode("utf-8"))

    def create_token(self, user_id: str, username: str, role: str, is_refresh: bool = False) -> str:
        now = time.time()
        ttl = self.refresh_ttl if is_refresh else self.access_ttl
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "username": username,
            "role": role,
            "iat": now,
            "exp": now + ttl,
            "iss": self.issuer,
            "aud": self.audience,
            "type": "refresh" if is_refresh else "access",
        }

        h_b64 = self._b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_b64 = self._b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        message = f"{h_b64}.{p_b64}".encode()
        signature = hmac.new(self.secret, message, hashlib.sha256).digest()
        sig_b64 = self._b64_encode(signature)

        return f"{h_b64}.{p_b64}.{sig_b64}"

    def verify_token(self, token: str) -> TokenPayload:
        if token in self._revocation_blocklist:
            raise ValueError("Token has been revoked")

        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Malformed JWT token structure")

        h_b64, p_b64, sig_b64 = parts
        message = f"{h_b64}.{p_b64}".encode()
        expected_sig = hmac.new(self.secret, message, hashlib.sha256).digest()
        actual_sig = self._b64_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid JWT signature")

        payload_dict = json.loads(self._b64_decode(p_b64).decode("utf-8"))
        now = time.time()
        if now > payload_dict.get("exp", 0):
            raise ValueError("Token has expired")
        if payload_dict.get("iss") != self.issuer:
            raise ValueError("Invalid token issuer")
        if payload_dict.get("aud") != self.audience:
            raise ValueError("Invalid token audience")

        return TokenPayload(
            sub=payload_dict["sub"],
            username=payload_dict["username"],
            role=payload_dict["role"],
            exp=payload_dict["exp"],
            iat=payload_dict["iat"],
            iss=payload_dict["iss"],
            aud=payload_dict["aud"],
        )

    def revoke_token(self, token: str) -> None:
        self._revocation_blocklist.add(token)


jwt_service = JWTService()
