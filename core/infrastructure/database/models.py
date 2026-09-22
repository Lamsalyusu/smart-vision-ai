"""SQLAlchemy ORM models used by the application.

Only the core entities that previously lived in the in‑memory `DataManager`
are defined here.  They map closely to the TypeScript `UserAccount`,
`SessionInfo`, `DetectionCropRecord` and telemetry structures.

The models are deliberately minimal – fields required for the current
business logic are included.  Additional columns can be added later
without breaking existing code.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, DateTime, Boolean, Float, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship

from .engine import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    auth_provider = Column(String, default="local")
    role = Column(String, default="operator")
    created_at = Column(DateTime, default=datetime.utcnow)

    # One‑to‑many relationship to sessions
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.id} ({self.username})>"


class Session(Base):
    __tablename__ = "sessions"

    token = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

    # Simple foreign‑key relationship (no explicit FK to keep it lightweight)
    user = relationship("User", back_populates="sessions")

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def slide(self, delta: timedelta = timedelta(days=7)) -> None:
        now = datetime.utcnow()
        self.last_active_at = now
        self.expires_at = now + delta

    def __repr__(self) -> str:
        return f"<Session {self.token[:8]}... for {self.user_id}>"


class DetectionCropRecord(Base):
    __tablename__ = "crops"

    id = Column(String, primary_key=True)
    object_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # optional fields
    distance_m = Column(Float, nullable=True)
    has_crop = Column(Boolean, default=False)
    # store any extra payload as JSON for flexibility
    payload = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<Crop {self.id} {self.object_class} {self.confidence:.2f}>"


class TelemetryStat(Base):
    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False)
    value = Column(JSON, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Telemetry {self.key}={self.value}>"


class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String, nullable=False)
    user_id = Column(String, nullable=True)
    details = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<AuditLog {self.event_type} user={self.user_id}>"
