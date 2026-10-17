from __future__ import annotations

import uuid
from django.db import models
from django.utils import timezone


class User(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True, db_index=True)
    username = models.CharField(max_length=100, unique=True, db_index=True)
    password_hash = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    profile_image_url = models.TextField(blank=True, null=True)
    auth_provider = models.CharField(max_length=50, default="email")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = "users"
        indexes = [
            models.Index(fields=["email"]),
            models.Index(fields=["username"]),
        ]

    def __str__(self) -> str:
        return f"{self.username} ({self.email})"


class UserSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="sessions"
    )
    session_token = models.CharField(max_length=255, unique=True, db_index=True)
    ip_address = models.CharField(max_length=50, blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    expires_at = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "user_sessions"
        indexes = [
            models.Index(fields=["session_token"]),
            models.Index(fields=["user", "is_active"]),
        ]

    def is_expired(self) -> bool:
        return timezone.now() > self.expires_at


class DetectionSession(models.Model):
    SESSION_TYPES = (
        ("image", "Static Image"),
        ("realtime", "Real-time Stream"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="detection_sessions"
    )
    session_name = models.CharField(max_length=255, blank=True, null=True)
    session_type = models.CharField(
        max_length=50, choices=SESSION_TYPES, default="image"
    )
    started_at = models.DateTimeField(default=timezone.now)
    ended_at = models.DateTimeField(blank=True, null=True)
    total_detections = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "detection_sessions"


class DetectionHistory(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="detection_history",
        null=True,
        blank=True,
    )
    object_class = models.CharField(max_length=100, db_index=True)
    confidence = models.FloatField()
    bbox = models.JSONField(blank=True, null=True)
    crop_image = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        db_table = "detection_history"
        ordering = ["-created_at"]


class DetectionCrop(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="detection_crops",
        null=True,
        blank=True,
    )
    object_class = models.CharField(max_length=100)
    confidence = models.FloatField(blank=True, null=True)
    crop_image = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "detection_crops"
        ordering = ["-created_at"]


class DetectionResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        DetectionSession,
        on_delete=models.CASCADE,
        related_name="results",
        null=True,
        blank=True,
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="detection_results"
    )
    image_url = models.TextField(blank=True, null=True)
    detected_objects = models.JSONField(default=list)
    total_objects_detected = models.IntegerField(default=0)
    processing_time_ms = models.IntegerField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "detection_results"


class LLMQuery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="llm_queries",
        null=True,
        blank=True,
    )
    detection_result = models.ForeignKey(
        DetectionResult, on_delete=models.SET_NULL, null=True, blank=True
    )
    query_text = models.TextField()
    llm_response = models.TextField()
    context_objects = models.JSONField(blank=True, null=True)
    processing_time_ms = models.IntegerField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "llm_queries"


class Model3D(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    object_class = models.CharField(max_length=100, unique=True, db_index=True)
    model_name = models.CharField(max_length=255)
    model_url = models.TextField()
    thumbnail_url = models.TextField(blank=True, null=True)
    file_size_mb = models.DecimalField(
        max_digits=10, decimal_places=2, blank=True, null=True
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "models_3d"


class PasswordReset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="password_resets"
    )
    token = models.CharField(max_length=255, unique=True)
    expires_at = models.DateTimeField()
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "password_resets"
