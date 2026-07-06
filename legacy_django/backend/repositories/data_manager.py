from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta
from typing import Any, Optional

from django.conf import settings
from django.db.models import Count
from django.utils import timezone

from backend.models import (
    DetectionCrop,
    DetectionHistory,
    User,
    UserSession,
)
from backend.supabase_client import supabase_admin

logger = logging.getLogger(__name__)


class DataRepositoryManager:
    """
    Unified Data Access Layer supporting both local SQLite ORM and Supabase.
    Automatically routes to local ORM if Supabase is offline or unconfigured.
    """

    def __init__(self) -> None:
        self._supabase_enabled = bool(
            not getattr(settings, "OFFLINE_MODE", False)
            and supabase_admin is not None
        )

    def is_supabase_active(self) -> bool:
        return bool(
            not getattr(settings, "OFFLINE_MODE", False)
            and supabase_admin is not None
        )

    # -------------------------------------------------------------------------
    # Authentication & Session Management
    # -------------------------------------------------------------------------

    def create_user(
        self,
        full_name: str,
        email: str,
        username: str,
        password_hash: str,
        auth_provider: str = "email",
    ) -> dict[str, Any]:
        """Create new user account in local ORM (and sync to Supabase if active)."""
        user = User.objects.create(
            full_name=full_name,
            email=email.lower().strip(),
            username=username.strip(),
            password_hash=password_hash,
            auth_provider=auth_provider,
            is_active=True,
            is_verified=False,
        )

        user_data = {
            "id": str(user.id),
            "full_name": user.full_name,
            "email": user.email,
            "username": user.username,
            "password_hash": user.password_hash,
            "auth_provider": user.auth_provider,
            "created_at": user.created_at.isoformat(),
        }

        if self.is_supabase_active() and supabase_admin:
            try:
                supabase_admin.table("users").insert(user_data).execute()
            except Exception as exc:
                logger.warning("Supabase user sync skipped: %s", exc)

        return user_data

    def find_user_by_identifier(self, identifier: str) -> Optional[dict[str, Any]]:
        """Locate user by username or email."""
        clean_id = identifier.strip().lower()
        user = User.objects.filter(email__iexact=clean_id).first()
        if not user:
            user = User.objects.filter(username__iexact=identifier.strip()).first()

        if user:
            return {
                "id": str(user.id),
                "full_name": user.full_name,
                "email": user.email,
                "username": user.username,
                "password_hash": user.password_hash,
                "auth_provider": user.auth_provider,
                "is_active": user.is_active,
            }

        # Check Supabase fallback if local user missing and Supabase active
        if self.is_supabase_active() and supabase_admin:
            for col in ("email", "username"):
                try:
                    res = (
                        supabase_admin.table("users")
                        .select("*")
                        .eq(col, identifier.strip())
                        .execute()
                    )
                    if res.data and isinstance(res.data, list) and len(res.data) > 0:
                        first_item = res.data[0]
                        if isinstance(first_item, dict):
                            return first_item
                except Exception:
                    pass

        return None

    def create_session(
        self,
        user_id: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        """Generate high-entropy session token and persist session."""
        session_token = secrets.token_urlsafe(32)
        expires_at = timezone.now() + timedelta(days=7)

        user = User.objects.filter(id=user_id).first()
        if user:
            UserSession.objects.create(
                user=user,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=expires_at,
                is_active=True,
            )

        if self.is_supabase_active() and supabase_admin:
            try:
                supabase_admin.table("user_sessions").insert({
                    "user_id": str(user_id),
                    "session_token": session_token,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "expires_at": expires_at.isoformat(),
                    "is_active": True,
                }).execute()
            except Exception as exc:
                logger.debug("Supabase session insert skipped: %s", exc)

        return session_token

    def get_user_from_session(self, session_token: str) -> Optional[dict[str, Any]]:
        """Validate active session token and return user metadata."""
        if not session_token:
            return None

        # Check local ORM first
        session = UserSession.objects.select_related("user").filter(
            session_token=session_token,
            is_active=True,
        ).first()

        if session:
            if session.is_expired():
                session.is_active = False
                session.save(update_fields=["is_active"])
                return None

            return {
                "id": str(session.user.id),
                "user_id": str(session.user.id),
                "full_name": session.user.full_name,
                "email": session.user.email,
                "username": session.user.username,
                "session_token": session_token,
                "users": {
                    "id": str(session.user.id),
                    "full_name": session.user.full_name,
                    "email": session.user.email,
                    "username": session.user.username,
                },
            }

        # Fallback to Supabase if active
        if self.is_supabase_active() and supabase_admin:
            try:
                res = (
                    supabase_admin.table("user_sessions")
                    .select("*, users(*)")
                    .eq("session_token", session_token)
                    .eq("is_active", True)
                    .single()
                    .execute()
                )
                if res.data and isinstance(res.data, dict):
                    data_dict: dict[str, Any] = res.data
                    exp_str = data_dict.get("expires_at", "")
                    if exp_str and isinstance(exp_str, str):
                        exp_dt = datetime.fromisoformat(
                            exp_str.replace("Z", "+00:00")
                        )
                        if datetime.now(exp_dt.tzinfo) > exp_dt:
                            return None
                    u_data = data_dict.get("users")
                    if not isinstance(u_data, dict):
                        u_data = {}
                    return {
                        "id": str(u_data.get("id", data_dict.get("user_id", ""))),
                        "full_name": str(u_data.get("full_name", "")),
                        "email": str(u_data.get("email", "")),
                        "username": str(u_data.get("username", "")),
                        "session_token": session_token,
                    }
            except Exception:
                pass

        return None

    def invalidate_session(self, session_token: str) -> None:
        """Terminate active session."""
        UserSession.objects.filter(session_token=session_token).update(is_active=False)
        if self.is_supabase_active() and supabase_admin:
            try:
                (
                    supabase_admin.table("user_sessions")
                    .update({"is_active": False})
                    .eq("session_token", session_token)
                    .execute()
                )
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Telemetry, Detections & Crops
    # -------------------------------------------------------------------------

    def save_detection_record(
        self,
        user_id: Optional[str],
        object_class: str,
        confidence: float,
        bbox: Optional[list[float]] = None,
        crop_image: Optional[str] = None,
    ) -> None:
        """Persist detection record to local SQLite (and forward to Supabase)."""
        user = User.objects.filter(id=user_id).first() if user_id else None
        DetectionHistory.objects.create(
            user=user,
            object_class=object_class,
            confidence=float(confidence),
            bbox=bbox,
            crop_image=crop_image,
        )

        if self.is_supabase_active() and supabase_admin and user_id:
            try:
                supabase_admin.table("detection_history").insert({
                    "user_id": str(user_id),
                    "object_class": object_class,
                    "confidence": round(float(confidence), 4),
                    "bbox": bbox,
                    "crop_image": crop_image,
                }).execute()
            except Exception:
                pass

    def get_user_stats(self, user_id: str) -> dict[str, Any]:
        """Compute aggregated detection analytics for user dashboard."""
        qs = DetectionHistory.objects.filter(user_id=user_id)
        total_detections = qs.count()

        class_counts = (
            qs.values("object_class")
            .annotate(count=Count("id"))
            .order_by("-count")[:10]
        )

        recent = list(
            qs.order_by("-created_at")[:15].values(
                "object_class", "confidence", "created_at"
            )
        )

        dist = {item["object_class"]: item["count"] for item in class_counts}
        formatted_recent = [
            {
                "class": r["object_class"],
                "confidence": round(r["confidence"], 2),
                "timestamp": (
                    r["created_at"].isoformat() if r["created_at"] else ""
                ),
            }
            for r in recent
        ]

        return {
            "total_detections": total_detections,
            "class_distribution": dist,
            "recent_detections": formatted_recent,
        }

    def save_crop(
        self,
        user_id: Optional[str],
        object_class: str,
        confidence: float,
        crop_image: str,
    ) -> dict[str, Any]:
        """Save cropped detection frame."""
        user = User.objects.filter(id=user_id).first() if user_id else None
        crop = DetectionCrop.objects.create(
            user=user,
            object_class=object_class,
            confidence=float(confidence) if confidence is not None else 0.0,
            crop_image=crop_image,
        )
        return {
            "id": str(crop.id),
            "class": crop.object_class,
            "created_at": crop.created_at.isoformat(),
        }

    def get_latest_crop(self, user_id: Optional[str]) -> Optional[dict[str, Any]]:
        """Retrieve most recently captured crop."""
        qs = DetectionCrop.objects.all()
        if user_id:
            qs = qs.filter(user_id=user_id)
        crop = qs.order_by("-created_at").first()
        if not crop:
            return None
        return {
            "id": str(crop.id),
            "object_class": crop.object_class,
            "confidence": crop.confidence,
            "crop_image": crop.crop_image,
            "created_at": crop.created_at.isoformat(),
        }

    def clear_history(self, user_id: str) -> bool:
        """Purge historical detections for user."""
        DetectionHistory.objects.filter(user_id=user_id).delete()
        DetectionCrop.objects.filter(user_id=user_id).delete()
        if self.is_supabase_active() and supabase_admin:
            try:
                (
                    supabase_admin.table("detection_history")
                    .delete()
                    .eq("user_id", str(user_id))
                    .execute()
                )
                (
                    supabase_admin.table("detection_crops")
                    .delete()
                    .eq("user_id", str(user_id))
                    .execute()
                )
            except Exception as exc:
                logger.debug("Supabase clear history skipped: %s", exc)
        return True


data_manager = DataRepositoryManager()
