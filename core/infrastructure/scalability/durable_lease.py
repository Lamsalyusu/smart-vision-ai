"""Durable, database-backed leases for cross-worker coordination.

``DistributedLock`` provides an in-process/Redis fast path.  This store adds a
persistent fencing-token lease that survives process restarts and works across
workers that only share the relational database.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

_LEASE_METADATA = MetaData()

LEASE_TABLE = Table(
    "service_leases",
    _LEASE_METADATA,
    Column("resource", String(255), primary_key=True),
    Column("owner", String(64), nullable=False),
    Column("fencing_token", BigInteger, nullable=False),
    Column("expires_at_seconds", Float, nullable=False),
)


@dataclass(frozen=True, slots=True)
class Lease:
    resource: str
    owner: str
    fencing_token: int
    expires_at_seconds: float


class DurableLeaseStore:
    """Acquire, renew, and release leases with monotonically increasing fencing tokens."""

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if engine is None:
            from core.infrastructure.database.engine import engine as application_engine

            engine = application_engine
        self._engine = engine
        self._clock = clock
        self._ensure_schema()

    @property
    def engine(self) -> Engine:
        return self._engine

    def _ensure_schema(self) -> None:
        LEASE_TABLE.create(self._engine, checkfirst=True)

    def acquire(
        self,
        resource: str,
        *,
        owner: str | None = None,
        ttl_seconds: float = 30.0,
        now_seconds: float | None = None,
    ) -> Lease | None:
        resource = self._validate_resource(resource)
        owner = owner or uuid.uuid4().hex
        if not owner or len(owner) > 64:
            raise ValueError("lease owner must be 1-64 characters")
        ttl = self._validate_ttl(ttl_seconds)
        now = self._resolve_now(now_seconds)

        with self._engine.begin() as connection:
            row = (
                connection.execute(
                    select(LEASE_TABLE)
                    .where(LEASE_TABLE.c.resource == resource)
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if row is None:
                lease = Lease(resource, owner, 1, now + ttl)
                connection.execute(insert(LEASE_TABLE).values(
                    resource=lease.resource,
                    owner=lease.owner,
                    fencing_token=lease.fencing_token,
                    expires_at_seconds=lease.expires_at_seconds,
                ))
                return lease

            if (
                float(row["expires_at_seconds"]) > now
                and str(row["owner"]) != owner
            ):
                return None

            fencing_token = int(row["fencing_token"]) + 1
            lease = Lease(resource, owner, fencing_token, now + ttl)
            result = connection.execute(
                update(LEASE_TABLE)
                .where(
                    LEASE_TABLE.c.resource == resource,
                    LEASE_TABLE.c.fencing_token == int(row["fencing_token"]),
                )
                .values(
                    owner=owner,
                    fencing_token=fencing_token,
                    expires_at_seconds=lease.expires_at_seconds,
                )
            )
            return lease if result.rowcount == 1 else None

    def renew(
        self,
        lease: Lease,
        *,
        ttl_seconds: float = 30.0,
        now_seconds: float | None = None,
    ) -> bool:
        ttl = self._validate_ttl(ttl_seconds)
        now = self._resolve_now(now_seconds)
        with self._engine.begin() as connection:
            result = connection.execute(
                update(LEASE_TABLE)
                .where(
                    LEASE_TABLE.c.resource == lease.resource,
                    LEASE_TABLE.c.owner == lease.owner,
                    LEASE_TABLE.c.fencing_token == lease.fencing_token,
                    LEASE_TABLE.c.expires_at_seconds > now,
                )
                .values(expires_at_seconds=now + ttl)
            )
            return result.rowcount == 1

    def release(self, lease: Lease) -> bool:
        with self._engine.begin() as connection:
            result = connection.execute(
                update(LEASE_TABLE)
                .where(
                    LEASE_TABLE.c.resource == lease.resource,
                    LEASE_TABLE.c.owner == lease.owner,
                    LEASE_TABLE.c.fencing_token == lease.fencing_token,
                )
                .values(expires_at_seconds=0.0)
            )
            return result.rowcount == 1

    def current(self, resource: str) -> Lease | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    select(LEASE_TABLE).where(LEASE_TABLE.c.resource == resource)
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return Lease(
            str(row["resource"]),
            str(row["owner"]),
            int(row["fencing_token"]),
            float(row["expires_at_seconds"]),
        )

    def _resolve_now(self, explicit: float | None) -> float:
        value = self._clock() if explicit is None else explicit
        if not math.isfinite(value) or value < 0:
            raise ValueError("lease clock must be finite and non-negative")
        return value

    @staticmethod
    def _validate_resource(resource: str) -> str:
        if not resource or len(resource) > 255:
            raise ValueError("lease resource must be 1-255 characters")
        return resource

    @staticmethod
    def _validate_ttl(ttl_seconds: float) -> float:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("lease TTL must be finite and positive")
        return ttl_seconds


def create_memory_lease_store(
    *, clock: Callable[[], float] = time.time
) -> DurableLeaseStore:
    """Create an isolated in-memory store, primarily for tests and tooling."""
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    return DurableLeaseStore(engine, clock=clock)
