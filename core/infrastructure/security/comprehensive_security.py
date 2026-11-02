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
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography import x509
from cryptography.x509.oid import NameOID


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
    XCHACHA20_POLY1305 = "xchacha20_poly1305"


class HashAlgorithm(Enum):
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"


class KDFAlgorithm(Enum):
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    HKDF = "hkdf"
    ARGON2 = "argon2"


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    kdf_algorithm: KDFAlgorithm = KDFAlgorithm.PBKDF2
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


@dataclass(frozen=True, slots=True)
class Certificate:
    cert_pem: bytes
    key_pem: bytes
    subject: str
    issuer: str
    not_before: float
    not_after: float
    san: List[str] = field(default_factory=list)
    fingerprint: str = ""


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
        elif algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305:
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
        elif algorithm == "ED25519":
            from cryptography.hazmat.primitives.asymmetric import ed25519
            private_key = ed25519.Ed25519PrivateKey.generate()
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            public_pem = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return KeyPair(private_pem, public_pem, "ED25519")

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def encrypt(self, data: bytes, key: bytes, algorithm: EncryptionAlgorithm = None) -> EncryptedData:
        algorithm = algorithm or self.config.encryption_algorithm

        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key)
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(data, key)
        elif algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305:
            return self._encrypt_xchacha20(data, key)

        raise ValueError(f"Unsupported encryption algorithm: {algorithm}")

    def decrypt(self, encrypted: EncryptedData, key: bytes) -> bytes:
        if encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted, key)
        elif encrypted.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted, key)
        elif encrypted.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encrypted, key)
        elif encrypted.algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305:
            return self._decrypt_xchacha20(encrypted, key)

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

    def _encrypt_xchacha20(self, data: bytes, key: bytes) -> EncryptedData:
        nonce = secrets.token_bytes(24)
        cipher = Cipher(algorithms.XChaCha20(key, nonce), mode=None)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return EncryptedData(ciphertext, nonce, None, EncryptionAlgorithm.XCHACHA20_POLY1305)

    def _decrypt_xchacha20(self, encrypted: EncryptedData, key: bytes) -> bytes:
        cipher = Cipher(algorithms.XChaCha20(key, encrypted.nonce), mode=None)
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
        iterations: int = None,
        key_length: int = 32
    ) -> bytes:
        salt = salt or secrets.token_bytes(self.config.salt_length)
        iterations = iterations or self.config.key_derivation_iterations
        algorithm = algorithm or self.config.hash_algorithm

        if algorithm == HashAlgorithm.SHA256:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=key_length, salt=salt, iterations=iterations)
        elif algorithm == HashAlgorithm.SHA512:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA512(), length=key_length, salt=salt, iterations=iterations)
        else:
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=key_length, salt=salt, iterations=iterations)

        return kdf.derive(password)

    def derive_key_scrypt(
        self,
        password: bytes,
        salt: bytes = None,
        n: int = 32768,
        r: int = 8,
        p: int = 1,
        key_length: int = 32
    ) -> bytes:
        salt = salt or secrets.token_bytes(self.config.salt_length)
        kdf = Scrypt(salt=salt, length=key_length, n=n, r=r, p=p)
        return kdf.derive(password)

    def derive_key_hkdf(
        self,
        input_key_material: bytes,
        salt: bytes = None,
        info: bytes = b"",
        algorithm: HashAlgorithm = None,
        key_length: int = 32
    ) -> bytes:
        salt = salt or secrets.token_bytes(self.config.salt_length)
        algorithm = algorithm or self.config.hash_algorithm

        if algorithm == HashAlgorithm.SHA256:
            hash_algo = hashes.SHA256()
        elif algorithm == HashAlgorithm.SHA512:
            hash_algo = hashes.SHA512()
        else:
            hash_algo = hashes.SHA256()

        hkdf = HKDF(algorithm=hash_algo, length=key_length, salt=salt, info=info)
        return hkdf.derive(input_key_material)

    def verify_password(self, password: bytes, salt: bytes, key: bytes, iterations: int = None, algorithm: HashAlgorithm = None) -> bool:
        derived = self.derive_key(password, salt, algorithm, iterations)
        return hmac.compare_digest(derived, key)


class TokenManager:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.tokens: Dict[str, Dict] = {}
        self.lock = threading.RLock()

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

    def create_jwt(self, payload: Dict, secret: str, algorithm: str = "HS256", expires_in: float = 3600) -> str:
        import jwt
        payload = payload.copy()
        payload["exp"] = time.time() + expires_in
        payload["iat"] = time.time()
        return jwt.encode(payload, secret, algorithm=algorithm)

    def verify_jwt(self, token: str, secret: str, algorithms: List[str] = None) -> Optional[Dict]:
        import jwt
        try:
            return jwt.decode(token, secret, algorithms=algorithms or ["HS256"])
        except jwt.InvalidTokenError:
            return None

    def store_token(self, token: str, user_id: str, expires_in: float = None, scopes: List[str] = None):
        with self.lock:
            self.tokens[token] = {
                'user_id': user_id,
                'created_at': time.time(),
                'expires_at': time.time() + (expires_in or self.config.session_timeout),
                'scopes': scopes or []
            }

    def validate_token(self, token: str) -> Optional[Dict]:
        with self.lock:
            if token not in self.tokens:
                return None

            token_data = self.tokens[token]
            if token_data['expires_at'] < time.time():
                del self.tokens[token]
                return None

            return token_data

    def revoke_token(self, token: str) -> bool:
        with self.lock:
            if token in self.tokens:
                del self.tokens[token]
                return True
            return False

    def revoke_all_user_tokens(self, user_id: str):
        with self.lock:
            to_delete = [t for t, d in self.tokens.items() if d['user_id'] == user_id]
            for t in to_delete:
                del self.tokens[t]


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

    @staticmethod
    def sanitize_sql(value: str) -> str:
        return value.replace("'", "''").replace(";", "").replace("--", "")

    @staticmethod
    def sanitize_html(value: str) -> str:
        import html
        return html.escape(value)

    @staticmethod
    def validate_json(value: str, schema: Dict = None) -> Tuple[bool, Optional[Dict]]:
        import json
        try:
            data = json.loads(value)
            if schema:
                # Basic schema validation could be added here
                pass
            return True, data
        except json.JSONDecodeError:
            return False, None


class CertificateManager:
    def __init__(self, cert_dir: str = "./certs"):
        self.cert_dir = cert_dir

    def generate_self_signed_cert(
        self,
        common_name: str,
        san: List[str] = None,
        valid_days: int = 365,
        key_size: int = 2048
    ) -> Certificate:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)

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
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=valid_days)
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

        fingerprint = hashlib.sha256(cert_pem).hexdigest()

        return Certificate(
            cert_pem=cert_pem,
            key_pem=private_pem,
            subject=common_name,
            issuer=common_name,
            not_before=datetime.utcnow().timestamp(),
            not_after=(datetime.utcnow() + timedelta(days=valid_days)).timestamp(),
            san=san or [],
            fingerprint=fingerprint
        )

    def load_certificate(self, cert_path: str, key_path: str) -> Certificate:
        with open(cert_path, 'rb') as f:
            cert_pem = f.read()
        with open(key_path, 'rb') as f:
            key_pem = f.read()

        certificate = x509.load_pem_x509_certificate(cert_pem)
        fingerprint = hashlib.sha256(cert_pem).hexdigest()

        return Certificate(
            cert_pem=cert_pem,
            key_pem=key_pem,
            subject=certificate.subject.rfc4514_string(),
            issuer=certificate.issuer.rfc4514_string(),
            not_before=certificate.not_valid_before.timestamp(),
            not_after=certificate.not_valid_after.timestamp(),
            san=[ext.value for ext in certificate.extensions if ext.oid == x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME],
            fingerprint=fingerprint
        )

    def verify_certificate(self, cert: Certificate, ca_cert: Certificate) -> bool:
        try:
            cert_obj = x509.load_pem_x509_certificate(cert.cert_pem)
            ca_obj = x509.load_pem_x509_certificate(ca_cert.cert_pem)

            ca_public_key = ca_obj.public_key()
            ca_public_key.verify(
                cert_obj.signature,
                cert_obj.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert_obj.signature_hash_algorithm
            )
            return True
        except Exception:
            return False


class SecurityHeaders:
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

    @staticmethod
    def get_cors_headers(origin: str = "*") -> Dict[str, str]:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400"
        }


class SecurityMiddleware:
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter()
        self.token_manager = TokenManager()
        self.input_validator = InputValidator()

    def process_request(self, request: Dict) -> Dict:
        ip = request.get('ip', 'unknown')
        user_id = request.get('user_id')

        if not self.rate_limiter.is_allowed(ip):
            return {"error": "Rate limit exceeded", "status": 429}

        return {"allowed": True}

    def process_response(self, response: Dict) -> Dict:
        headers = SecurityHeaders.get_headers()
        response["headers"] = {**response.get("headers", {}), **headers}
        return response


class AuditLogger:
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


class SecureKeyStorage:
    def __init__(self, master_key: bytes = None):
        self.master_key = master_key or secrets.token_bytes(32)
        self.keys: Dict[str, bytes] = {}
        self.lock = threading.RLock()

    def store_key(self, key_id: str, key: bytes):
        with self.lock:
            encrypted = self._encrypt(key)
            self.keys[key_id] = encrypted

    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        with self.lock:
            if key_id not in self.keys:
                return None
            return self._decrypt(self.keys[key_id])

    def delete_key(self, key_id: str) -> bool:
        with self.lock:
            if key_id in self.keys:
                del self.keys[key_id]
                return True
            return False

    def _encrypt(self, key: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        cipher = Cipher(algorithms.AES(self.master_key), modes.GCM(nonce))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(key) + encryptor.finalize()
        return nonce + encryptor.tag + ciphertext

    def _decrypt(self, data: bytes) -> bytes:
        nonce = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]
        cipher = Cipher(algorithms.AES(self.master_key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()


class SecurityAudit:
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def audit_code_changes(self, changes: List[Dict]):
        for change in changes:
            self.audit_logger.log_event(
                "code_change",
                user_id=change.get("user"),
                details=change
            )

    def audit_config_changes(self, changes: List[Dict]):
        for change in changes:
            self.audit_logger.log_event(
                "config_change",
                user_id=change.get("user"),
                details=change,
                severity=SecurityLevel.HIGH
            )

    def audit_permission_changes(self, changes: List[Dict]):
        for change in changes:
            self.audit_logger.log_event(
                "permission_change",
                user_id=change.get("user"),
                details=change,
                severity=SecurityLevel.CRITICAL
            )


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet