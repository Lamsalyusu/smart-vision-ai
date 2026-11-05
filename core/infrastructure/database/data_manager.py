"""Database‑backed implementation of the former in‑memory DataManager.

The original `lib/store.ts` provided a TypeScript‑only in‑memory store.  For
persistence and scalability we now expose a Python class with a very similar
API that uses the SQLAlchemy ORM models defined in
`core.infrastructure.database.models`.

Only the core business‑logic methods required by the existing API routes are
implemented.  Additional helpers can be added later without breaking the
public contract.
"""

import hashlib
import logging
import secrets
from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import uuid4

import numpy as np
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .clock import utcnow_naive
from .engine import SessionLocal, engine
from .migrate import upgrade as upgrade_schema
from .models import (
    AuditLogRecord,
    DetectionCropRecord,
    EmailVerification,
    FailedLoginAttempt,
    PasswordResetToken,
    ReplayFrameRecord,
    TelemetryStat,
    User,
)
from .models import Session as DBSession

logger = logging.getLogger(__name__)

# Bring the schema up to date with the Alembic migrations in ./migrations.
# Databases created before migrations existed are upgraded in place.
try:
    upgrade_schema(engine)
except SQLAlchemyError:
    logger.exception("Database schema migration failed")


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
                id=f"usr_{uuid4().hex}",
                full_name=full_name,
                email=email.lower().strip(),
                username=username.strip(),
                password_hash=password_hash,
                auth_provider="local",
                role=role,
                created_at=utcnow_naive(),
            )
            db.add(user)
            try:
                db.commit()
            except IntegrityError as exc:
                db.rollback()
                raise ValueError("A user with this email or username already exists") from exc
            db.refresh(user)
            return user

    def find_user_by_identifier(self, identifier: str) -> User | None:
        """Lookup a user by e‑mail or username (case‑insensitive)."""
        identifier = identifier.lower().strip()
        with SessionLocal() as db:
            return (
                db.query(User)
                .filter(
                    (func.lower(User.email) == identifier)
                    | (func.lower(User.username) == identifier)
                )
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
    #
    # The client receives a 256-bit random token; only its SHA-256 digest is
    # stored, so a leaked database does not expose usable session cookies.
    @staticmethod
    def _make_token() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_session_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, user_id: str, days: int = 7) -> str:
        token = self._make_token()
        now = utcnow_naive()
        expires = now + timedelta(days=days)
        with SessionLocal() as db:
            session = DBSession(
                token=self.hash_session_token(token),
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
            sess = (
                db.query(DBSession)
                .filter(DBSession.token == self.hash_session_token(token))
                .first()
            )
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
            sess = (
                db.query(DBSession)
                .filter(DBSession.token == self.hash_session_token(token))
                .first()
            )
            if sess:
                db.delete(sess)
                db.commit()

    def revoke_all_user_sessions(self, user_id: str) -> None:
        with SessionLocal() as db:
            db.query(DBSession).filter(DBSession.user_id == user_id).delete(synchronize_session=False)
            db.commit()

    # Password reset
    def create_password_reset(
        self,
        user_id: str,
        ttl: timedelta = timedelta(minutes=30),
        min_interval: timedelta = timedelta(minutes=1),
    ) -> str | None:
        """Issue a reset token, or return None if one was issued within ``min_interval``.

        Issuing a token invalidates any earlier unused tokens for the user.
        """
        now = utcnow_naive()
        with SessionLocal() as db:
            recent = (
                db.query(PasswordResetToken)
                .filter(
                    PasswordResetToken.user_id == user_id,
                    PasswordResetToken.created_at > now - min_interval,
                )
                .first()
            )
            if recent is not None:
                return None
            db.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user_id
            ).delete(synchronize_session=False)
            token = self._make_token()
            db.add(PasswordResetToken(
                token_hash=self.hash_session_token(token),
                user_id=user_id,
                created_at=now,
                expires_at=now + ttl,
            ))
            db.commit()
        return token

    def consume_password_reset(self, token: str, new_password_hash: str) -> str | None:
        """Set a new password if the token is valid; return the user id or None.

        The token is single-use and every existing session of the user is revoked.
        """
        now = utcnow_naive()
        with SessionLocal() as db:
            record = (
                db.query(PasswordResetToken)
                .filter(PasswordResetToken.token_hash == self.hash_session_token(token))
                .with_for_update()
                .first()
            )
            if record is None or record.used_at is not None or record.expires_at < now:
                return None
            user = db.query(User).filter(User.id == record.user_id).first()
            if user is None:
                return None
            record.used_at = now
            user.password_hash = new_password_hash
            db.query(DBSession).filter(DBSession.user_id == user.id).delete(synchronize_session=False)
            db.query(FailedLoginAttempt).filter(
                FailedLoginAttempt.identifier.in_([user.email.lower(), user.username.lower()])
            ).delete(synchronize_session=False)
            db.commit()
            return user.id

    # Email verification
    def issue_email_verification(
        self,
        user_id: str,
        ttl: timedelta = timedelta(hours=24),
        min_interval: timedelta = timedelta(minutes=1),
    ) -> str | None:
        """Issue a verification token; None if already verified or issued within ``min_interval``."""
        now = utcnow_naive()
        token = self._make_token()
        with SessionLocal() as db:
            record = db.get(EmailVerification, user_id)
            if record is not None:
                if record.verified_at is not None:
                    return None
                if record.token_hash is not None and record.created_at > now - min_interval:
                    return None
            else:
                record = EmailVerification(user_id=user_id)
                db.add(record)
            record.token_hash = self.hash_session_token(token)
            record.created_at = now
            record.expires_at = now + ttl
            db.commit()
        return token

    def confirm_email_verification(self, token: str) -> str | None:
        """Mark the email verified if the token is valid; return the user id or None."""
        now = utcnow_naive()
        with SessionLocal() as db:
            record = (
                db.query(EmailVerification)
                .filter(EmailVerification.token_hash == self.hash_session_token(token))
                .first()
            )
            if record is None or record.expires_at is None or record.expires_at < now:
                return None
            record.verified_at = now
            record.token_hash = None
            record.expires_at = None
            db.commit()
            return record.user_id

    def mark_email_verified(self, user_id: str) -> None:
        now = utcnow_naive()
        with SessionLocal() as db:
            record = db.get(EmailVerification, user_id) or EmailVerification(user_id=user_id)
            record.verified_at = now
            record.token_hash = None
            record.expires_at = None
            db.merge(record)
            db.commit()

    def is_email_verified(self, user_id: str) -> bool:
        with SessionLocal() as db:
            record = db.get(EmailVerification, user_id)
            return record is not None and record.verified_at is not None

    # Administration
    def list_users(self) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = (
                db.query(User, EmailVerification.verified_at)
                .outerjoin(EmailVerification, EmailVerification.user_id == User.id)
                .order_by(User.created_at)
                .all()
            )
        return [
            {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "username": user.username,
                "role": user.role,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "email_verified": verified_at is not None,
            }
            for user, verified_at in rows
        ]

    def set_user_role(self, user_id: str, role: str) -> tuple[str, str]:
        """Change a user's role; return (previous, new).

        Raises LookupError for an unknown user and ValueError when the change
        would leave no administrator.
        """
        with SessionLocal() as db:
            user = db.query(User).filter(User.id == user_id).with_for_update().first()
            if user is None:
                raise LookupError(user_id)
            previous = user.role
            if previous == "admin" and role != "admin":
                admins = db.query(func.count(User.id)).filter(User.role == "admin").scalar() or 0
                if admins <= 1:
                    raise ValueError("Cannot remove the last administrator")
            user.role = role
            db.commit()
        return previous, role

    # Audit log
    def record_audit(self, event_type: str, user_id: str | None = None, **details: Any) -> None:
        try:
            with SessionLocal() as db:
                db.add(AuditLogRecord(event_type=event_type, user_id=user_id, details=details))
                db.commit()
        except SQLAlchemyError:
            # Auditing must not break the request it describes.
            logger.exception("Could not write audit event %s", event_type)

    def get_audit_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            records = (
                db.query(AuditLogRecord)
                .order_by(AuditLogRecord.timestamp.desc(), AuditLogRecord.id.desc())
                .limit(limit)
                .all()
            )
        return [
            {
                "event_type": r.event_type,
                "user_id": r.user_id,
                "details": r.details,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in records
        ]

    # Crop handling – persisted in ``crops`` table.
    def save_crop(self, crop_data: dict[str, Any]) -> DetectionCropRecord:
        """Insert a new crop record.

        ``crop_data`` should contain the fields required by the ORM model:
        ``object_class``, ``confidence`` and optionally ``distance_m`` and
        ``has_crop``.
        """
        with SessionLocal() as db:
            record = DetectionCropRecord(
                id=f"crop_{uuid4().hex}",
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

    def record_detections(
        self,
        detections: Sequence[dict[str, Any]],
        frame: dict[str, Any] | None = None,
    ) -> None:
        """Store detections and, when given, one timing record for the frame.

        ``frame`` holds measured durations in milliseconds (``inference_ms``,
        ``total_ms``, ``roundtrip_ms``); frames without detections are recorded
        too, so throughput and latency reflect every processed frame.
        """
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
        if frame is not None:
            records.append(TelemetryStat(
                key="frame",
                value={**frame, "detections": len(detections), "timestamp": recorded_at.isoformat()},
                recorded_at=recorded_at,
            ))
        if not records:
            return
        with SessionLocal() as db:
            db.add_all(records)
            db.commit()

    def clear_history(self) -> None:
        with SessionLocal() as db:
            db.query(TelemetryStat).filter(
                TelemetryStat.key.in_(["detection", "frame"])
            ).delete(synchronize_session=False)
            db.commit()

    def get_performance(self, sample_frames: int = 500, window_minutes: int = 5) -> dict[str, Any]:
        """Latency percentiles and throughput computed from recorded frame timings."""
        since = utcnow_naive() - timedelta(minutes=window_minutes)
        with SessionLocal() as db:
            frames = (
                db.query(TelemetryStat)
                .filter(TelemetryStat.key == "frame")
                .order_by(TelemetryStat.recorded_at.desc(), TelemetryStat.id.desc())
                .limit(sample_frames)
                .all()
            )
            frames_in_window = (
                db.query(func.count(TelemetryStat.id))
                .filter(TelemetryStat.key == "frame", TelemetryStat.recorded_at >= since)
                .scalar()
            ) or 0

        def percentiles(field: str) -> tuple[float | None, float | None]:
            values = [
                float(f.value[field]) for f in frames
                if isinstance(f.value.get(field), (int, float))
            ]
            if not values:
                return None, None
            p50, p95 = np.percentile(values, [50, 95])
            return round(float(p50), 1), round(float(p95), 1)

        inference_p50, inference_p95 = percentiles("inference_ms")
        roundtrip_p50, roundtrip_p95 = percentiles("roundtrip_ms")
        detections_per_frame = (
            round(sum(int(f.value.get("detections", 0)) for f in frames) / len(frames), 2)
            if frames else None
        )
        return {
            "sample_frames": len(frames),
            "window_minutes": window_minutes,
            "frames_in_window": frames_in_window,
            "frames_per_minute": round(frames_in_window / window_minutes, 1),
            "inference_ms_p50": inference_p50,
            "inference_ms_p95": inference_p95,
            "roundtrip_ms_p50": roundtrip_p50,
            "roundtrip_ms_p95": roundtrip_p95,
            "detections_per_frame": detections_per_frame,
        }

    def save_replay_frame(
        self,
        session_id: str,
        captured_at: float,
        detections: list[dict[str, Any]],
        max_frames: int = 240,
        retention_days: int = 30,
    ) -> dict[str, Any]:
        now = utcnow_naive()
        record = ReplayFrameRecord(
            id=str(uuid4()),
            session_id=session_id,
            captured_at=captured_at,
            detections=detections,
            recorded_at=now,
        )
        frame_data = {"capturedAt": captured_at, "detections": detections}
        with SessionLocal() as db:
            db.add(record)
            db.flush()
            expired_before = now - timedelta(days=retention_days)
            db.query(ReplayFrameRecord).filter(
                ReplayFrameRecord.recorded_at < expired_before
            ).delete(synchronize_session=False)
            stale_ids = (
                db.query(ReplayFrameRecord.id)
                .filter(ReplayFrameRecord.session_id == session_id)
                .order_by(ReplayFrameRecord.captured_at.desc(), ReplayFrameRecord.id.desc())
                .offset(max_frames)
                .all()
            )
            if stale_ids:
                db.query(ReplayFrameRecord).filter(
                    ReplayFrameRecord.id.in_([frame_id for (frame_id,) in stale_ids])
                ).delete(synchronize_session=False)
            db.commit()
        return frame_data

    def get_replay_frames(self, session_id: str, limit: int = 240) -> list[dict[str, Any]]:
        retention_cutoff = utcnow_naive() - timedelta(days=30)
        with SessionLocal() as db:
            records = (
                db.query(ReplayFrameRecord)
                .filter(
                    ReplayFrameRecord.session_id == session_id,
                    ReplayFrameRecord.recorded_at >= retention_cutoff,
                )
                .order_by(ReplayFrameRecord.captured_at.desc(), ReplayFrameRecord.id.desc())
                .limit(limit)
                .all()
            )
        return [
            {"capturedAt": record.captured_at, "detections": record.detections}
            for record in reversed(records)
        ]

    def clear_replay_frames(self, session_id: str) -> None:
        with SessionLocal() as db:
            db.query(ReplayFrameRecord).filter_by(session_id=session_id).delete(
                synchronize_session=False
            )
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
            "performance": self.get_performance(),
        }


# Export a singleton for easy import, mirroring the original TS file.
data_manager = DataManager()
