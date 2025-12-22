from django.db import models
from django.contrib.auth.hashers import make_password, check_password
import uuid
from datetime import datetime, timedelta


class User(models.Model):
    """Custom User model stored in Supabase"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=100, unique=True)
    password_hash = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    profile_image_url = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'users'
        managed = False  # Let Supabase manage the table

    def set_password(self, raw_password):
        """Hash and set password"""
        self.password_hash = make_password(raw_password)

    def check_password(self, raw_password):
        """Verify password"""
        return check_password(raw_password, self.password_hash)

    def __str__(self):
        return self.email


class UserSession(models.Model):
    """User session management"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField()
    session_token = models.CharField(max_length=255, unique=True)
    ip_address = models.CharField(max_length=50, null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'user_sessions'
        managed = False

    def is_expired(self):
        """Check if session is expired"""
        return datetime.now() > self.expires_at

    def __str__(self):
        return f"Session for user {self.user_id}"


class DetectionSession(models.Model):
    """Tracking detection sessions (image or realtime)"""
    SESSION_TYPES = [
        ('image', 'Image Upload'),
        ('realtime', 'Real-time Detection'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField()
    session_name = models.CharField(max_length=255, null=True, blank=True)
    session_type = models.CharField(max_length=50, choices=SESSION_TYPES, default='image')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    total_detections = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'detection_sessions'
        managed = False

    def __str__(self):
        return f"{self.session_type} session - {self.session_name or self.id}"


class DetectionResult(models.Model):
    """Individual detection results"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_id = models.UUIDField()
    user_id = models.UUIDField()
    image_url = models.TextField(null=True, blank=True)
    detected_objects = models.JSONField()  # Array of {class, confidence, bbox}
    total_objects_detected = models.IntegerField(default=0)
    processing_time_ms = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'detection_results'
        managed = False
        ordering = ['-created_at']

    def __str__(self):
        return f"Detection {self.id} - {self.total_objects_detected} objects"


class LLMQuery(models.Model):
    """LLM queries and responses"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField()
    detection_result_id = models.UUIDField(null=True, blank=True)
    query_text = models.TextField()
    llm_response = models.TextField()
    context_objects = models.JSONField(null=True, blank=True)
    processing_time_ms = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'llm_queries'
        managed = False
        ordering = ['-created_at']

    def __str__(self):
        return f"Query: {self.query_text[:50]}..."


class Model3D(models.Model):
    """3D models for detected objects"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    object_class = models.CharField(max_length=100)
    model_name = models.CharField(max_length=255)
    model_url = models.TextField()  # URL to .gltf or .obj file
    thumbnail_url = models.TextField(null=True, blank=True)
    file_size_mb = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'models_3d'
        managed = False

    def __str__(self):
        return f"{self.object_class} - {self.model_name}"