"""SQLAlchemy ORM models used by the application.

Only the core entities that previously lived in the in‑memory `DataManager`
are defined here.  They map closely to the TypeScript `UserAccount`,
`SessionInfo`, `DetectionCropRecord` and telemetry structures.

The models are deliberately minimal – fields required for the current
business logic are included.  Additional columns can be added later
without breaking existing code.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .clock import utcnow_naive
from .engine import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    auth_provider: Mapped[str] = mapped_column(String, default="local")
    role: Mapped[str] = mapped_column(String, default="operator")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)

    # One‑to‑many relationship to sessions
    sessions: Mapped[list[Session]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.id} ({self.username})>"


class Session(Base):
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Simple foreign‑key relationship (no explicit FK to keep it lightweight)
    user: Mapped[User] = relationship(back_populates="sessions")

    def is_expired(self) -> bool:
        return self.expires_at is None or utcnow_naive() > self.expires_at

    def slide(self, delta: timedelta = timedelta(days=7)) -> None:
        now = utcnow_naive()
        self.last_active_at = now
        self.expires_at = now + delta

    def __repr__(self) -> str:
        return f"<Session {self.token[:8]}... for {self.user_id}>"


class DetectionCropRecord(Base):
    __tablename__ = "crops"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    object_class: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # optional fields
    distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_crop: Mapped[bool] = mapped_column(Boolean, default=False)
    # store any extra payload as JSON for flexibility
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<Crop {self.id} {self.object_class} {self.confidence:.2f}>"


class FailedLoginAttempt(Base):
    __tablename__ = "failed_login_attempts"

    identifier: Mapped[str] = mapped_column(String, primary_key=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)


class TelemetryStat(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)

    def __repr__(self) -> str:
        return f"<Telemetry {self.key}={self.value}>"


class ReplayFrameRecord(Base):
    __tablename__ = "replay_frames"
    __table_args__ = (Index("ix_replay_session_captured", "session_id", "captured_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    captured_at: Mapped[float] = mapped_column(Float, nullable=False)
    detections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)


class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)

    def __repr__(self) -> str:
        return f"<AuditLog {self.event_type} user={self.user_id}>"


class PasswordResetToken(Base):
    """Single-use password reset token. Only the SHA-256 digest is stored."""

    __tablename__ = "password_reset_tokens"

    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EmailVerification(Base):
    """Email verification state per user. Only the SHA-256 digest of a pending token is stored."""

    __tablename__ = "email_verifications"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    token_hash: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
