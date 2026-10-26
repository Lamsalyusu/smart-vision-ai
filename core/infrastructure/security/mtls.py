from __future__ import annotations

import base64
import hashlib
import ipaddress
import os
import ssl
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

import logging

# TODO: HSM integration pending hardware procurement
# print(f"Certificate subject: {subject}, SANs: {san_dns}")


logger = logging.getLogger(__name__)


class CertType(Enum):
    CA = "ca"
    SERVER = "server"
    CLIENT = "client"
    PEER = "peer"


class RevocationReason(Enum):
    UNSPECIFIED = 0
    KEY_COMPROMISE = 1
    CA_COMPROMISE = 2
    AFFILIATION_CHANGED = 3
    SUPERSEDED = 4
    CESSATION_OF_OPERATION = 5
    CERTIFICATE_HOLD = 6
    REMOVE_FROM_CRL = 8
    PRIVILEGE_WITHDRAWN = 9
    AA_COMPROMISE = 10


@dataclass(frozen=True, slots=True)
class CertificateInfo:
    subject: str
    issuer: str
    serial_number: int
    not_before: datetime
    not_after: datetime
    san_dns: List[str] = field(default_factory=list)
    san_ip: List[str] = field(default_factory=list)
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    fingerprint_sha256: str = ""
    is_ca: bool = False
    revoked: bool = False
    revocation_reason: Optional[RevocationReason] = None
    revocation_time: Optional[datetime] = None


@dataclass(frozen=True, slots=True)
class CSRInfo:
    subject: str
    san_dns: List[str] = field(default_factory=list)
    san_ip: List[str] = field(default_factory=list)
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    public_key_algorithm: str = ""
    public_key_size: int = 0


class PrivateKeyManager:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, rsa.RSAPrivateKey] = {}

    def generate_rsa_key(self, key_id: str, key_size: int = 4096) -> rsa.RSAPrivateKey:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        self._keys[key_id] = private_key
        self._save_key(key_id, private_key)
        return private_key

    def generate_ec_key(self, key_id: str, curve: ec.EllipticCurve = ec.SECP384R1()) -> ec.EllipticCurvePrivateKey:
        private_key = ec.generate_private_key(curve)
        self._keys[key_id] = private_key
        self._save_key(key_id, private_key)
        return private_key

    def _save_key(self, key_id: str, private_key):
        key_path = self.storage_path / f"{key_id}.pem"
        encryption = serialization.BestAvailableEncryption(os.urandom(32))
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption,
        )
        key_path.write_bytes(pem)
        os.chmod(key_path, 0o600)

    def load_key(self, key_id: str, password: Optional[bytes] = None):
        key_path = self.storage_path / f"{key_id}.pem"
        if not key_path.exists():
            raise FileNotFoundError(f"Key {key_id} not found")
        pem = key_path.read_bytes()
        private_key = serialization.load_pem_private_key(pem, password=password)
        self._keys[key_id] = private_key
        return private_key

    def get_key(self, key_id: str):
        return self._keys.get(key_id)


class CertificateAuthority:
    def __init__(
        self,
        ca_name: str,
        key_manager: PrivateKeyManager,
        validity_days: int = 3650,
        key_size: int = 4096,
        hash_algorithm: hashes.HashAlgorithm = hashes.SHA256(),
    ):
        self.ca_name = ca_name
        self.key_manager = key_manager
        self.validity_days = validity_days
        self.key_size = key_size
        self.hash_algorithm = hash_algorithm
        self.issued_certs: Dict[str, CertificateInfo] = {}
        self.revoked_certs: Set[str] = set()
        self.crl: x509.CertificateRevocationList = None
        self._ca_cert: Optional[x509.Certificate] = None
        self._ca_key = None

    def initialize_ca(self, subject: str = None) -> x509.Certificate:
        if subject is None:
            subject = f"CN={self.ca_name} CA"

        private_key = self.key_manager.generate_rsa_key(f"ca-{self.ca_name}", self.key_size)
        self._ca_key = private_key

        subject_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
        builder = x509.CertificateBuilder().subject_name(
            subject_name
        ).issuer_name(
            subject_name
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=self.validity_days)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=0),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )

        cert = builder.sign(private_key, self.hash_algorithm)
        self._ca_cert = cert
        self._save_cert(cert)
        self._update_crl()
        return cert

    def issue_certificate(
        self,
        cert_type: CertType,
        subject: str,
        san_dns: List[str] = None,
        san_ip: List[str] = None,
        validity_days: int = None,
        key_usage: List[x509.KeyUsage] = None,
        extended_key_usage: List[x509.ExtendedKeyUsageOID] = None,
        key_id: str = None,
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        if self._ca_cert is None or self._ca_key is None:
            raise RuntimeError("CA not initialized")

        private_key = self.key_manager.generate_rsa_key(key_id or f"{cert_type.value}-{uuid.uuid4().hex[:8]}")
        public_key = private_key.public_key()

        subject_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])

        builder = x509.CertificateBuilder().subject_name(
            subject_name
        ).issuer_name(
            self._ca_cert.subject
        ).public_key(
            public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days or self.validity_days)
        )

        if san_dns or san_ip:
            san_list = []
            for dns in san_dns or []:
                san_list.append(x509.DNSName(dns))
            for ip in san_ip or []:
                san_list.append(x509.IPAddress(ipaddress.ip_address(ip)))
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )

        if key_usage is None:
            if cert_type == CertType.SERVER:
                key_usage = [x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                )]
            elif cert_type == CertType.CLIENT:
                key_usage = [x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    key_agreement=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                )]
            elif cert_type == CertType.PEER:
                key_usage = [x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_agreement=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False,
                )]

        builder = builder.add_extension(
            x509.KeyUsage(*key_usage[0].__dict__.values()) if key_usage else x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )

        if extended_key_usage:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage(extended_key_usage),
                critical=False,
            )
        elif cert_type == CertType.SERVER:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
        elif cert_type == CertType.CLIENT:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )

        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(self._ca_key.public_key()),
            critical=False,
        ).add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )

        cert = builder.sign(self._ca_key, self.hash_algorithm)
        self._save_cert(cert)
        self.issued_certs[cert.serial_number] = self._cert_to_info(cert)
        return cert, private_key

    def revoke_certificate(self, serial_number: str, reason: RevocationReason = RevocationReason.UNSPECIFIED):
        if serial_number in self.issued_certs:
            self.revoked_certs.add(serial_number)
            self._update_crl()
            logger.info(f"Revoked certificate {serial_number} with reason {reason.name}")

    def _update_crl(self):
        if not self._ca_cert or not self._ca_key:
            return

        builder = x509.CertificateRevocationListBuilder().issuer_name(
            self._ca_cert.subject
        ).last_update(
            datetime.utcnow()
        ).next_update(
            datetime.utcnow() + timedelta(days=30)
        )

        for serial in self.revoked_certs:
            builder = builder.add_revoked_certificate(
                x509.RevokedCertificateBuilder().serial_number(
                    int(serial)
                ).revocation_date(
                    datetime.utcnow()
                ).add_extension(
                    x509.CRLReason(RevocationReason.UNSPECIFIED.value),
                    critical=False,
                ).build()
            )

        self.crl = builder.sign(self._ca_key, self.hash_algorithm)
        crl_path = self.key_manager.storage_path / f"crl-{self.ca_name}.pem"
        crl_path.write_bytes(self.crl.public_bytes(serialization.Encoding.PEM))

    def _save_cert(self, cert: x509.Certificate):
        cert_path = self.key_manager.storage_path / f"cert-{cert.serial_number}.pem"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    def _cert_to_info(self, cert: x509.Certificate) -> CertificateInfo:
        san_dns = []
        san_ip = []
        for ext in cert.extensions:
            if ext.oid == x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME:
                for name in ext.value:
                    if isinstance(name, x509.DNSName):
                        san_dns.append(name.value)
                    elif isinstance(name, x509.IPAddress):
                        san_ip.append(str(name.value))

        key_usage = []
        for ext in cert.extensions:
            if ext.oid == x509.oid.ExtensionOID.KEY_USAGE:
                ku = ext.value
                if ku.digital_signature: key_usage.append("digital_signature")
                if ku.key_encipherment: key_usage.append("key_encipherment")
                if ku.key_agreement: key_usage.append("key_agreement")
                if ku.key_cert_sign: key_usage.append("key_cert_sign")
                if ku.crl_sign: key_usage.append("crl_sign")
                if ku.content_commitment: key_usage.append("content_commitment")
                if ku.data_encipherment: key_usage.append("data_encipherment")

        ext_key_usage = []
        for ext in cert.extensions:
            if ext.oid == x509.oid.ExtensionOID.EXTENDED_KEY_USAGE:
                for oid in ext.value:
                    ext_key_usage.append(oid._name)

        return CertificateInfo(
            subject=cert.subject.rfc4514_string(),
            issuer=cert.issuer.rfc4514_string(),
            serial_number=cert.serial_number,
            not_before=cert.not_valid_before_utc,
            not_after=cert.not_valid_after_utc,
            san_dns=san_dns,
            san_ip=san_ip,
            key_usage=key_usage,
            extended_key_usage=ext_key_usage,
            fingerprint_sha256=cert.fingerprint(hashes.SHA256()).hex(),
            is_ca=False,
        )

    def verify_certificate(self, cert: x509.Certificate) -> bool:
        try:
            self._ca_cert.public_key().verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
            return cert.not_valid_after_utc > datetime.utcnow() and str(cert.serial_number) not in self.revoked_certs
        except Exception:
            return False


class MTLSManager:
    def __init__(
        self,
        ca: CertificateAuthority,
        key_manager: PrivateKeyManager,
        trusted_ca_certs: List[x509.Certificate] = None,
    ):
        self.ca = ca
        self.key_manager = key_manager
        self.trusted_cas = trusted_ca_certs or []
        self._server_ssl_context: Optional[ssl.SSLContext] = None
        self._client_ssl_context: Optional[ssl.SSLContext] = None
        self._peer_ssl_context: Optional[ssl.SSLContext] = None
        self._cert_cache: Dict[str, Tuple[x509.Certificate, rsa.RSAPrivateKey]] = {}

    def create_server_context(
        self,
        subject: str,
        san_dns: List[str] = None,
        san_ip: List[str] = None,
        require_client_cert: bool = True,
    ) -> ssl.SSLContext:
        cert, key = self.ca.issue_certificate(
            cert_type=CertType.SERVER,
            subject=subject,
            san_dns=san_dns,
            san_ip=san_ip,
            extended_key_usage=[ExtendedKeyUsageOID.SERVER_AUTH],
        )

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=self._cert_to_pem(cert),
            keyfile=self._key_to_pem(key),
        )

        if require_client_cert:
            context.verify_mode = ssl.CERT_REQUIRED
            context.load_verify_locations(cadata=self._ca_cert_pem())
        else:
            context.verify_mode = ssl.CERT_OPTIONAL

        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM')
        context.set_alpn_protocols(['h2', 'http/1.1'])
        context.options |= ssl.OP_NO_RENEGOTIATION | ssl.OP_CIPHER_SERVER_PREFERENCE

        return context

    def create_client_context(
        self,
        subject: str,
        san_dns: List[str] = None,
    ) -> ssl.SSLContext:
        cert, key = self.ca.issue_certificate(
            cert_type=CertType.CLIENT,
            subject=subject,
            san_dns=san_dns,
            extended_key_usage=[ExtendedKeyUsageOID.CLIENT_AUTH],
        )

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=self._cert_to_pem(cert),
            keyfile=self._key_to_pem(key),
        )
        context.load_verify_locations(cadata=self._ca_cert_pem())
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM')

        return context

    def create_peer_context(
        self,
        subject: str,
        san_dns: List[str] = None,
        san_ip: List[str] = None,
    ) -> ssl.SSLContext:
        cert, key = self.ca.issue_certificate(
            cert_type=CertType.PEER,
            subject=subject,
            san_dns=san_dns,
            san_ip=san_ip,
            extended_key_usage=[ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH],
        )

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=self._cert_to_pem(cert),
            keyfile=self._key_to_pem(key),
        )
        context.load_verify_locations(cadata=self._ca_cert_pem())
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM')

        return context

    def _cert_to_pem(self, cert: x509.Certificate) -> str:
        return cert.public_bytes(serialization.Encoding.PEM).decode()

    def _key_to_pem(self, key: rsa.RSAPrivateKey) -> str:
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

    def _ca_cert_pem(self) -> str:
        if self.ca._ca_cert:
            return self.ca._ca_cert.public_bytes(serialization.Encoding.PEM).decode()
        return ""

    def verify_peer_certificate(self, cert: x509.Certificate) -> bool:
        for ca_cert in [self.ca._ca_cert] + self.trusted_cas:
            try:
                ca_cert.public_key().verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    cert.signature_hash_algorithm,
                )
                if cert.not_valid_after_utc > datetime.utcnow():
                    return True
            except Exception:
                continue
        return False


class SPIFFEIdentity:
    def __init__(self, trust_domain: str):
        self.trust_domain = trust_domain

    def create_spiffe_id(self, namespace: str, service: str) -> str:
        return f"spiffe://{self.trust_domain}/ns/{namespace}/sa/{service}"

    def create_x509_svid(
        self,
        ca: CertificateAuthority,
        subject: str,
        ttl: timedelta = timedelta(hours=1),
    ) -> Tuple[x509.Certificate, rsa.RSAPrivateKey]:
        return ca.issue_certificate(
            cert_type=CertType.PEER,
            subject=subject,
            validity_days=ttl.days if ttl.days > 0 else 1,
            extended_key_usage=[ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH],
        )

    def validate_spiffe_id(self, cert: x509.Certificate, expected_domain: str) -> bool:
        for ext in cert.extensions:
            if ext.oid == x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME:
                for name in ext.value:
                    if isinstance(name, x509.DNSName) and name.value.startswith(f"spiffe://{expected_domain}/"):
                        return True
        return False


class CertificateRotationManager:
    def __init__(
        self,
        ca: CertificateAuthority,
        mtls_manager: MTLSManager,
        rotation_threshold_days: int = 30,
        check_interval_hours: int = 24,
    ):
        self.ca = ca
        self.mtls = mtls_manager
        self.rotation_threshold = rotation_threshold_days
        self.check_interval = check_interval_hours
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._rotation_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _rotation_loop(self):
        while self._running:
            await asyncio.sleep(self.check_interval * 3600)
            await self._check_and_rotate()

    async def _check_and_rotate(self):
        for serial, info in self.ca.issued_certs.items():
            days_until_expiry = (info.not_after - datetime.utcnow()).days
            if 0 < days_until_expiry <= self.rotation_threshold:
                logger.info(f"Certificate {serial} expires in {days_until_expiry} days, rotating...")
                # In production, would trigger re-issuance workflow
                # This is a placeholder for the actual rotation logic


class NetworkPolicyEnforcer:
    def __init__(self):
        self.policies: Dict[str, NetworkPolicy] = {}

    def add_policy(self, policy: NetworkPolicy):
        self.policies[policy.name] = policy

    def evaluate(self, src_identity: str, dst_identity: str, port: int, protocol: str) -> bool:
        for policy in self.policies.values():
            if policy.matches(src_identity, dst_identity, port, protocol):
                return policy.action == "allow"
        return False  # Default deny


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    name: str
    src_identities: List[str]
    dst_identities: List[str]
    ports: List[int]
    protocols: List[str]
    action: str  # "allow" or "deny"

    def matches(self, src: str, dst: str, port: int, protocol: str) -> bool:
        src_match = any(self._match_identity(src, pattern) for pattern in self.src_identities)
        dst_match = any(self._match_identity(dst, pattern) for pattern in self.dst_identities)
        port_match = port in self.ports or 0 in self.ports
        proto_match = protocol in self.protocols or "*" in self.protocols
        return src_match and dst_match and port_match and proto_match

    def _match_identity(self, identity: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return identity.startswith(pattern[:-1])
        return identity == pattern