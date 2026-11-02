from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import hashlib
import hmac
import secrets
import base64
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(Enum):
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"
    ECIES = "ecies"


class HashAlgorithm(Enum):
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    token_length: int = 32
    session_timeout: float = 3600.0
    max_failed_attempts: int = 5
    lockout_duration: float = 300.0


@dataclass(frozen=True, slots=True)
class EncryptedData:
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes] = None
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeyPair:
    private_key: bytes
    public_key: bytes
    algorithm: str
    created_at: float = field(default_factory=time.time)


class EncryptionManager:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

    def generate_key(self, algorithm: EncryptionAlgorithm = None) -> bytes:
        algorithm = algorithm or self.config.encryption_algorithm

        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return secrets.token_bytes(32)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return secrets.token_bytes(32)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)
        return secrets.token_bytes(32)

    def generate_key_pair(self, algorithm: str = "RSA") -> KeyPair:
        if algorithm == "RSA":
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return KeyPair(private_pem, public_pem, "RSA")
        elif algorithm == "ECDSA":
            private_key = ec.generate_private_key(ec.SECP256R1())
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return KeyPair(private_pem, public_pem, "ECDSA")

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def encrypt(self, data: bytes, key: bytes, algorithm: EncryptionAlgorithm = None) -> EncryptedData:
        algorithm = algorithm or self.config.encryption_algorithm

        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(data, key)

        raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

    def decrypt(self, encrypted: EncryptedData, key: bytes) -> bytes:
        if encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted, key)
        elif encrypted.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted, key)
        elif encrypted.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encrypted, key)

        raise ValueError(f"Unsupported encryption algorithm: {encrypted.algorithm}")

    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> EncryptedData:
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return EncryptedData(ciphertext, nonce, encryptor.tag, EncryptionAlgorithm.AES_256_GCM)

    def _decrypt_aes_gcm(self, encrypted: EncryptedData, key: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.GCM(encrypted.nonce, encrypted.tag))
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted.ciphertext) + decryptor.finalize()

    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> EncryptedData:
        iv = secrets.token_bytes(16)
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return EncryptedData(ciphertext, iv, None, EncryptionAlgorithm.AES_256_CBC)

    def _decrypt_aes_cbc(self, encrypted: EncryptedData, key: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(key), modes.CBC(encrypted.nonce))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted.ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()

    def _encrypt_chacha20(self, data: bytes, key: bytes) -> EncryptedData:
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return EncryptedData(ciphertext, nonce, None, EncryptionAlgorithm.CHACHA20_POLY1305)

    def _decrypt_chacha20(self, encrypted: EncryptedData, key: bytes) -> bytes:
        cipher = Cipher(algorithms.ChaCha20(key, encrypted.nonce), mode=None)
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted.ciphertext) + decryptor.finalize()


class HashManager:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

    def hash(self, data: bytes, algorithm: HashAlgorithm = None) -> bytes:
        algorithm = algorithm or self.config.hash_algorithm

        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA384:
            return hashlib.sha384(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).digest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).digest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).digest()

        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def hmac(self, data: bytes, key: bytes, algorithm: HashAlgorithm = None) -> bytes:
        algorithm = algorithm or self.config.hash_algorithm

        if algorithm == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).digest()
        elif algorithm == HashAlgorithm.SHA384:
            return hmac.new(key, data, hashlib.sha384).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).digest()

        raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")

    def derive_key(
        self,
        password: bytes,
        salt: bytes = None,
        algorithm: HashAlgorithm = None,
        iterations: int = None
    ) -> bytes:
        salt = salt or secrets.token_bytes(self.config.salt_length)
        iterations = iterations or self.config.key_derivation_iterations
        algorithm = algorithm or self.config.hash_algorithm

        if algorithm == HashAlgorithm.SHA256:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
        elif algorithm == HashAlgorithm.SHA512:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA512(), length=64, salt=salt, iterations=iterations)
        else:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)

        return kdf.derive(password)

    def verify_password(self, password: bytes, salt: bytes, key: bytes, iterations: int = None, algorithm: HashAlgorithm = None) -> bool:
        derived = self.derive_key(password, salt, algorithm, iterations)
        return hmac.compare_digest(derived, key)


class TokenManager:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

    def generate_token(self, length: int = None) -> str:
        length = length or self.config.token_length
        return secrets.token_urlsafe(length)

    def generate_api_key(self, prefix: str = "sk") -> str:
        token = self.generate_token()
        return f"{prefix}_{token}"

    def generate_session_id(self) -> str:
        return self.generate_token(32)

    def generate_csrf_token(self) -> str:
        return self.generate_token(32)

    def hash_token(self, token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()


class RateLimiter:
    def __init__(self, max_requests: int = 100, window: float = 60.0):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        with self.lock:
            self.requests[identifier] = [
                t for t in self.requests[identifier] if now - t < self.window
            ]

            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(now)
                return True
            return False

    def get_remaining(self, identifier: str) -> int:
        now = time.time()
        with self.lock:
            self.requests[identifier] = [
                t for t in self.requests[identifier] if now - t < self.window
            ]
            return max(0, self.max_requests - len(self.requests[identifier]))

    def reset(self, identifier: str):
        with self.lock:
            if identifier in self.requests:
                del self.requests[identifier]


class InputValidator:
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, allowed_chars: str = None) -> str:
        if not isinstance(value, str):
            return ""

        value = value.strip()

        if max_length > 0 and len(value) > max_length:
            value = value[:max_length]

        if allowed_chars:
            value = ''.join(c for c in value if c in allowed_chars)

        return value

    @staticmethod
    def validate_email(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_url(url: str) -> bool:
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))

    @staticmethod
    def validate_ip(ip: str) -> bool:
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        import re
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        filename = filename.strip('. ')
        return filename[:255]


class SecurityAuditLogger:
    def __init__(self, log_path: str = "security_audit.log"):
        self.log_path = log_path
        self.lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        user_id: str = None,
        ip: str = None,
        details: Dict = None,
        severity: SecurityLevel = SecurityLevel.MEDIUM
    ):
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "ip": ip,
            "details": details or {},
            "severity": severity.value
        }

        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_authentication(self, user_id: str, success: bool, ip: str, method: str):
        self.log_event(
            "authentication",
            user_id=user_id,
            ip=ip,
            details={"success": success, "method": method},
            severity=SecurityLevel.LOW if success else SecurityLevel.HIGH
        )

    def log_authorization(self, user_id: str, resource: str, action: str, allowed: bool):
        self.log_event(
            "authorization",
            user_id=user_id,
            details={"resource": resource, "action": action, "allowed": allowed},
            severity=SecurityLevel.LOW if allowed else SecurityLevel.MEDIUM
        )

    def log_data_access(self, user_id: str, data_type: str, operation: str, record_count: int):
        self.log_event(
            "data_access",
            user_id=user_id,
            details={"data_type": data_type, "operation": operation, "record_count": record_count},
            severity=SecurityLevel.LOW
        )

    def log_security_violation(self, violation_type: str, user_id: str = None, ip: str = None, details: Dict = None):
        self.log_event(
            "security_violation",
            user_id=user_id,
            ip=ip,
            details={"violation_type": violation_type, **(details or {})},
            severity=SecurityLevel.CRITICAL
        )


class SecureHeaders:
    @staticmethod
    def get_headers() -> Dict[str, str]:
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


class CertificateManager:
    def __init__(self, cert_dir: str = "./certs"):
        self.cert_dir = cert_dir

    def generate_self_signed_cert(
        self,
        common_name: str,
        san: List[str] = None,
        valid_days: int = 365
    ) -> Tuple[bytes, bytes]:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        from cryptography import x509
        from cryptography.x509.oid import NameOID
        import datetime

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name)
        ])

        builder = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=valid_days)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )

        if san:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName(name) for name in san]),
                critical=False
            )

        certificate = builder.sign(private_key, hashes.SHA256())

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)

        return cert_pem, private_pem


class SecurityMiddleware:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter()
        self.audit_logger = SecurityAuditLogger()
        self.encryption = EncryptionManager(config)
        self.token_manager = TokenManager(config)

    def process_request(self, request: Dict) -> Dict:
        ip = request.get('ip', 'unknown')
        user_id = request.get('user_id')

        if not self.rate_limiter.is_allowed(ip):
            self.audit_logger.log_security_violation("rate_limit_exceeded", user_id, ip)
            return {"error": "Rate limit exceeded", "status": 429}

        return {"allowed": True}

    def process_response(self, response: Dict) -> Dict:
        headers = SecureHeaders.get_headers()
        response["headers"] = {**response.get("headers", {}), **headers}
        return response


import threading
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet