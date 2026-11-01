"""Database‑backed implementation of the former in‑memory DataManager.

The original `lib/store.ts` provided a TypeScript‑only in‑memory store.  For
persistence and scalability we now expose a Python class with a very similar
API that uses the SQLAlchemy ORM models defined in
`core.infrastructure.database.models`.

Only the core business‑logic methods required by the existing API routes are
implemented.  Additional helpers can be added later without breaking the
public contract.
"""

import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .clock import utcnow_naive
from .engine import Base, SessionLocal, engine
from .models import DetectionCropRecord, FailedLoginAttempt, TelemetryStat, User
from .models import Session as DBSession

logger = logging.getLogger(__name__)

# Ensure tables exist (useful for development). In production migrations should
# be handled separately (e.g., Alembic).
try:
    Base.metadata.create_all(bind=engine)
except SQLAlchemyError:
    logger.exception("Database schema initialization failed")


class DataManager:
    """Thin wrapper around SQLAlchemy sessions providing the same methods that
    the original TypeScript `DataManager` offered.
    """

    # User management
    def create_user(
        self,
        full_name: str,
        email: str,
        username: str,
        password_hash: str,
        role: str = "operator",
    ) -> User:
        """Create a new user record.

        Args:
            full_name: Human readable name.
            email: Unique e‑mail address (lower‑cased).
            username: Unique login name.
            password_hash: Already‑hashed password (e.g., PBKDF2).
            role: Application role – defaults to ``operator``.
        """
        with SessionLocal() as db:
            user = User(
                id=f"usr_{int(utcnow_naive().timestamp())}_{hash(email) % 1000}",
                full_name=full_name,
                email=email.lower().strip(),
                username=username.strip(),
                password_hash=password_hash,
                auth_provider="local",
                role=role,
                created_at=utcnow_naive(),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user

    def find_user_by_identifier(self, identifier: str) -> User | None:
        """Lookup a user by e‑mail or username (case‑insensitive)."""
        identifier = identifier.lower().strip()
        with SessionLocal() as db:
            return (
                db.query(User)
                .filter((User.email == identifier) | (User.username == identifier))
                .first()
            )

    def get_user_by_id(self, user_id: str) -> User | None:
        with SessionLocal() as db:
            return db.query(User).filter(User.id == user_id).first()

    def is_account_locked(self, identifier: str) -> bool:
        key = identifier.lower().strip()
        now = utcnow_naive()
        with SessionLocal() as db:
            record = (
                db.query(FailedLoginAttempt)
                .filter_by(identifier=key)
                .with_for_update()
                .first()
            )
            if record is None:
                return False
            if record.locked_until is None:
                return False
            if now < record.locked_until:
                return True
            db.delete(record)
            db.commit()
            return False

    def record_failed_login(self, identifier: str) -> dict[str, Any]:
        key = identifier.lower().strip()
        now = utcnow_naive()
        with SessionLocal() as db:
            record = (
                db.query(FailedLoginAttempt)
                .filter_by(identifier=key)
                .with_for_update()
                .first()
            )
            if record is None:
                record = FailedLoginAttempt(identifier=key, attempt_count=0)
                db.add(record)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    return self.record_failed_login(identifier)
            elif record.locked_until is not None and record.locked_until <= now:
                record.attempt_count = 0
                record.locked_until = None

            record.attempt_count += 1
            locked = record.attempt_count >= 5
            if locked:
                record.locked_until = now + timedelta(minutes=15)
            remaining_attempts = max(0, 5 - record.attempt_count)
            db.commit()
            return {"locked": locked, "remaining_attempts": remaining_attempts}

    def clear_failed_logins(self, identifier: str) -> None:
        key = identifier.lower().strip()
        with SessionLocal() as db:
            db.query(FailedLoginAttempt).filter_by(identifier=key).delete()
            db.commit()

    # Session management – persisted in the ``sessions`` table.
    def _make_token(self) -> str:
        now = utcnow_naive()
        return f"sess_{int(now.timestamp())}_{hash(now) % 10000}"

    def create_session(self, user_id: str, days: int = 7) -> str:
        token = self._make_token()
        now = utcnow_naive()
        expires = now + timedelta(days=days)
        with SessionLocal() as db:
            session = DBSession(
                token=token,
                user_id=user_id,
                created_at=now,
                last_active_at=now,
                expires_at=expires,
            )
            db.add(session)
            db.commit()
        return token

    def get_user_from_session(self, token: str) -> User | None:
        with SessionLocal() as db:
            sess = db.query(DBSession).filter(DBSession.token == token).first()
            if not sess:
                return None
            if sess.expires_at is None or utcnow_naive() > sess.expires_at:
                db.delete(sess)
                db.commit()
                return None
            # slide expiration window
            now = utcnow_naive()
            sess.last_active_at = now
            sess.expires_at = now + timedelta(days=7)
            db.commit()
            return self.get_user_by_id(sess.user_id)

    def revoke_session(self, token: str) -> None:
        with SessionLocal() as db:
            sess = db.query(DBSession).filter(DBSession.token == token).first()
            if sess:
                db.delete(sess)
                db.commit()

    def revoke_all_user_sessions(self, user_id: str) -> None:
        with SessionLocal() as db:
            db.query(DBSession).filter(DBSession.user_id == user_id).delete(synchronize_session=False)
            db.commit()

    # Crop handling – persisted in ``crops`` table.
    def save_crop(self, crop_data: dict[str, Any]) -> DetectionCropRecord:
        """Insert a new crop record.

        ``crop_data`` should contain the fields required by the ORM model:
        ``object_class``, ``confidence`` and optionally ``distance_m`` and
        ``has_crop``.
        """
        with SessionLocal() as db:
            record = DetectionCropRecord(
                id=f"crop_{int(utcnow_naive().timestamp())}",
                object_class=crop_data["object_class"],
                confidence=crop_data["confidence"],
                timestamp=utcnow_naive(),
                distance_m=crop_data.get("distance_m"),
                has_crop=crop_data.get("has_crop", False),
                payload=crop_data.get("payload"),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            # also record detection (mirrors original logic)
            self.record_detection(
                class_name=crop_data["object_class"],
                confidence=crop_data["confidence"],
                distance_m=crop_data.get("distance_m"),
                has_crop=crop_data.get("has_crop", False),
            )
            return record

    def get_latest_crop(self, object_class: str | None = None) -> DetectionCropRecord | None:
        with SessionLocal() as db:
            query = db.query(DetectionCropRecord).order_by(DetectionCropRecord.timestamp.desc())
            if object_class:
                query = query.filter(DetectionCropRecord.object_class == object_class)
            return query.first()

    def record_detection(
        self,
        class_name: str,
        confidence: float,
        distance_m: float | None = None,
        has_crop: bool = False,
        max_history: int = 200,
    ) -> None:
        self.record_detections([{
            "class_name": class_name,
            "confidence": confidence,
            "distance_m": distance_m,
            "has_crop": has_crop,
        }])

    def record_detections(self, detections: Sequence[dict[str, Any]]) -> None:
        recorded_at = utcnow_naive()
        records = [
            TelemetryStat(
                key="detection",
                value={
                    "class": detection["class_name"],
                    "confidence": round(detection["confidence"], 2),
                    "distance_m": (
                        round(detection["distance_m"], 2)
                        if detection.get("distance_m") is not None
                        else None
                    ),
                    "timestamp": recorded_at.isoformat(),
                    "has_crop": detection.get("has_crop", False),
                },
                recorded_at=recorded_at,
            )
            for detection in detections
        ]
        if not records:
            return
        with SessionLocal() as db:
            db.add_all(records)
            db.commit()

    def clear_history(self) -> None:
        with SessionLocal() as db:
            db.query(TelemetryStat).filter(TelemetryStat.key == "detection").delete(synchronize_session=False)
            db.commit()

    def get_stats(self) -> dict[str, Any]:
        with SessionLocal() as db:
            detection_query = db.query(TelemetryStat).filter_by(key="detection")
            total_detections = detection_query.count()
            session_count = (
                db.query(DBSession)
                .filter(DBSession.expires_at > utcnow_naive())
                .count()
            )
            class_rows = (
                db.query(
                    TelemetryStat.value["class"].as_string(),
                    func.count(TelemetryStat.id),
                )
                .filter(TelemetryStat.key == "detection")
                .group_by(TelemetryStat.value["class"].as_string())
                .all()
            )
            recent_records = (
                detection_query.order_by(TelemetryStat.recorded_at.desc())
                .limit(10)
                .all()
            )

        distribution = {
            class_name: count
            for class_name, count in class_rows
            if class_name is not None
        }
        recent_activity = [
            {
                "id": str(record.id),
                "class": record.value.get("class"),
                "confidence": record.value.get("confidence"),
                "distance_m": record.value.get("distance_m"),
                "timestamp": record.value.get("timestamp") or record.recorded_at.isoformat(),
                "has_crop": record.value.get("has_crop", False),
            }
            for record in recent_records
        ]
        return {
            "total_detections": total_detections,
            "active_sessions": session_count,
            "class_distribution": distribution,
            "recent_activity": recent_activity,
        }


# Export a singleton for easy import, mirroring the original TS file.
data_manager = DataManager()
