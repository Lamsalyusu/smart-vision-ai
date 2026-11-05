"""FastAPI service for authentication, persistence and mesh reconstruction.

This service is internal: the Next.js server is its only intended client.
Every endpoint except the health probes requires the shared
``INTERNAL_API_TOKEN`` in the ``X-Internal-Token`` header, so exposing the
port by mistake does not expose user data or write access.

Database work is synchronous SQLAlchemy, so the endpoints that touch the
database are plain ``def`` functions; FastAPI runs them in its threadpool
instead of blocking the event loop.
"""

import hmac
import ipaddress
import logging
import os
import socket
import time
from typing import Annotated, Any
from urllib.parse import urlsplit

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from passlib.context import CryptContext
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import SQLAlchemyError

from ...domain.spatial.mesh_reconstruction import (
    CanonicalShapeGenerator,
    MarchingCubesReconstructor,
    MeshExporter,
    MeshPostProcessor,
    PointCloudNormalEstimator,
    VolumetricTSDFBuilder,
)
from ..database.data_manager import data_manager
from ..email import EmailNotConfiguredError, email_configured, send_email
from ..security.headers import SecurityHeadersMiddleware
from ..security.jwt_auth import jwt_service
from ..security.rate_limit import rate_limit
from ..security.rbac import AppPermission, AppRole, require_permission
from .webhooks import webhook_dispatcher
from .websocket_stream import ws_manager

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# A fixed hash used to spend the same verification time when a login names an
# unknown user, so response timing does not reveal which accounts exist.
_DUMMY_PASSWORD_HASH = pwd_context.hash("timing-equalisation-only")

SESSION_COOKIE = "session_token"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
MAX_MESH_POINTS = 50_000


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


async def require_internal_api_token(
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    expected_token = os.getenv("INTERNAL_API_TOKEN", "")
    if not expected_token:
        raise HTTPException(status_code=503, detail="INTERNAL_API_TOKEN is not configured")
    if x_internal_token is None or not hmac.compare_digest(x_internal_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid internal service token")


internal = [Depends(require_internal_api_token)]

# Per-client limits on the endpoints attackers would hammer. Login also has a
# per-account lockout after 5 failures; these stop spraying many accounts.
login_limit = Depends(rate_limit("login", limit=30, window_seconds=60))
signup_limit = Depends(rate_limit("signup", limit=10, window_seconds=600))
email_limit = Depends(rate_limit("email", limit=10, window_seconds=600))

app = FastAPI(title="Smart Vision AI Auth and Persistence Service")
app.add_middleware(SecurityHeadersMiddleware)
admin_users_dependency = Depends(require_permission(AppPermission.ADMIN_USERS))


# Request models
class LoginPayload(BaseModel):
    email_or_username: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class SignupPayload(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    username: str = Field(min_length=3, max_length=30, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=1024)


class EmailVerificationConfirm(BaseModel):
    token: str = Field(min_length=20, max_length=200)


class EmailVerificationResend(BaseModel):
    email: str = Field(max_length=254)


class RoleChange(BaseModel):
    role: AppRole
    actor_id: str | None = Field(default=None, max_length=64)


class PasswordResetRequest(BaseModel):
    email: str = Field(max_length=254)


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    new_password: str = Field(min_length=8, max_length=1024)


class CropPayload(BaseModel):
    object_class: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    distance_m: float | None = None
    has_crop: bool = False
    payload: dict[str, Any] | None = None


class DetectionPayload(BaseModel):
    class_name: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    distance_m: float | None = None
    has_crop: bool = False


class FrameTiming(BaseModel):
    """Measured durations for one processed camera frame, in milliseconds."""

    inference_ms: float | None = Field(default=None, ge=0, le=600_000)
    total_ms: float | None = Field(default=None, ge=0, le=600_000)
    roundtrip_ms: float | None = Field(default=None, ge=0, le=600_000)


class DetectionBatchPayload(BaseModel):
    detections: list[DetectionPayload] = Field(max_length=500)
    frame: FrameTiming | None = None


class ReplayDetectionPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4)
    track_id: int | None = None
    spatial_3d: dict[str, Any] | None = None


class ReplayFramePayload(BaseModel):
    session_id: str
    captured_at: float
    detections: list[ReplayDetectionPayload] = Field(max_length=100)


class WebhookRequest(BaseModel):
    target_url: str = Field(max_length=2048)
    event_type: str = Field(min_length=1, max_length=64)
    data: dict[str, Any]


class MeshReconstructionRequest(BaseModel):
    object_class: str | None = "chair"
    points: list[list[float]] | None = Field(default=None, max_length=MAX_MESH_POINTS)
    voxel_size: float = 0.08
    smoothing_iterations: int = 2
    output_format: str = "threejs"  # "threejs", "obj", "ply"


# Helpers
def _user_json(user: Any) -> dict[str, Any]:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "username": user.username,
        "role": user.role,
        "email_verified": data_manager.is_email_verified(user.id),
    }


def _verification_required() -> bool:
    """Enforce verification only when it is switched on and email can be sent."""
    return os.getenv("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true" and email_configured()


def _send_verification_email(user: Any) -> bool:
    """Email a verification link if email is configured; return whether one was sent."""
    if not email_configured():
        return False
    token = data_manager.issue_email_verification(user.id)
    if token is None:
        return False
    base_url = os.getenv("APP_BASE_URL", "http://localhost:3000").rstrip("/")
    try:
        send_email(
            user.email,
            "Verify your SmartVision AI email address",
            f"Hello {user.full_name},\n\n"
            "Please confirm this email address for your SmartVision AI account by "
            f"opening this link within 24 hours:\n{base_url}/verify-email?token={token}\n\n"
            "If you did not create an account, ignore this email.\n",
        )
    except (EmailNotConfiguredError, OSError):
        logger.exception("Could not send verification email")
        return False
    data_manager.record_audit("EMAIL_VERIFICATION_SENT", user.id)
    return True


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=os.getenv("NODE_ENV") == "production",
        samesite="lax",
        path="/",
    )


def _session_token(request: Request, x_session_token: str | None) -> str | None:
    """Read the session token from the X-Session-Token header or the cookie."""
    return x_session_token or request.cookies.get(SESSION_COOKIE)


def _login_response(response: Response, user: Any) -> dict[str, Any]:
    token = data_manager.create_session(user.id)
    _set_session_cookie(response, token)
    return {
        "success": True,
        "token": token,
        "access_token": jwt_service.create_token(user.id, user.username, user.role),
        "token_type": "Bearer",
        "user": _user_json(user),
    }


def _validate_webhook_target(url: str) -> None:
    """Reject webhook targets that would let callers reach internal hosts."""
    parts = urlsplit(url)
    allow_http = os.getenv("NODE_ENV") != "production"
    if parts.scheme not in ({"https", "http"} if allow_http else {"https"}):
        raise HTTPException(status_code=400, detail="Webhook URL must use https")
    if not parts.hostname:
        raise HTTPException(status_code=400, detail="Webhook URL has no host")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="Webhook host does not resolve") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise HTTPException(status_code=400, detail="Webhook host must be a public address")


# Authentication
@app.post("/login", dependencies=[*internal, login_limit])
def login(payload: LoginPayload, response: Response):
    identifier = payload.email_or_username.strip()
    if data_manager.is_account_locked(identifier):
        data_manager.record_audit("LOGIN_BLOCKED_LOCKED", identifier=identifier)
        raise HTTPException(
            status_code=429,
            detail="Account temporarily locked after repeated failed attempts. Try again in 15 minutes.",
        )
    user = data_manager.find_user_by_identifier(identifier)
    password_ok = verify_password(
        payload.password, user.password_hash if user else _DUMMY_PASSWORD_HASH
    )
    if user is None or not password_ok:
        result = data_manager.record_failed_login(identifier)
        data_manager.record_audit(
            "ACCOUNT_LOCKED" if result["locked"] else "LOGIN_FAILURE",
            user.id if user else None,
            identifier=identifier,
        )
        raise HTTPException(status_code=401, detail="Invalid email/username or password")
    data_manager.clear_failed_logins(identifier)
    # Checked only after the password, so it reveals nothing about unknown accounts.
    if _verification_required() and not data_manager.is_email_verified(user.id):
        data_manager.record_audit("LOGIN_BLOCKED_UNVERIFIED", user.id)
        raise HTTPException(
            status_code=403,
            detail="Please verify your email address before signing in. Check your inbox for the link.",
        )
    data_manager.record_audit("LOGIN_SUCCESS", user.id)
    return _login_response(response, user)


@app.post("/signup", dependencies=[*internal, signup_limit])
def signup(payload: SignupPayload, response: Response):
    email = payload.email.lower().strip()
    username = payload.username.strip()
    if data_manager.find_user_by_identifier(email):
        raise HTTPException(status_code=400, detail="Email is already registered")
    if data_manager.find_user_by_identifier(username):
        raise HTTPException(status_code=400, detail="Username is already taken")
    try:
        user = data_manager.create_user(
            full_name=payload.full_name.strip(),
            email=email,
            username=username,
            password_hash=hash_password(payload.password),
            role="operator",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data_manager.record_audit("USER_REGISTERED", user.id)
    verification_sent = _send_verification_email(user)
    if _verification_required():
        # No session until the address is confirmed.
        return {
            "success": True,
            "user_id": user.id,
            "verification_required": True,
            "verification_sent": verification_sent,
        }
    return {
        **_login_response(response, user),
        "user_id": user.id,
        "verification_required": False,
        "verification_sent": verification_sent,
    }


@app.post("/email-verification/confirm", dependencies=[*internal, email_limit])
def confirm_email_verification(payload: EmailVerificationConfirm):
    user_id = data_manager.confirm_email_verification(payload.token)
    if user_id is None:
        raise HTTPException(status_code=400, detail="This verification link is invalid or has expired")
    data_manager.record_audit("EMAIL_VERIFIED", user_id)
    return {"success": True}


@app.post("/email-verification/resend", dependencies=[*internal, email_limit])
def resend_email_verification(payload: EmailVerificationResend):
    """Send a new verification link. The response is the same whether or not the account exists."""
    if not email_configured():
        raise HTTPException(status_code=503, detail="Email is not configured")
    user = data_manager.find_user_by_identifier(payload.email)
    if user is not None and user.email == payload.email.lower().strip():
        _send_verification_email(user)
    return {"success": True}


@app.get("/session", dependencies=internal)
def get_session(
    request: Request,
    x_session_token: Annotated[str | None, Header()] = None,
):
    token = _session_token(request, x_session_token)
    if not token:
        raise HTTPException(status_code=401, detail="No session token")
    user = data_manager.get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return {"user": _user_json(user)}


@app.post("/logout", dependencies=internal)
def logout(
    request: Request,
    response: Response,
    x_session_token: Annotated[str | None, Header()] = None,
):
    token = _session_token(request, x_session_token)
    if token:
        user = data_manager.get_user_from_session(token)
        data_manager.revoke_session(token)
        if user:
            data_manager.record_audit("LOGOUT", user.id)
    response.delete_cookie(key=SESSION_COOKIE, path="/")
    return {"success": True}


@app.post("/password-reset/request", dependencies=[*internal, email_limit])
def request_password_reset(payload: PasswordResetRequest):
    """Email a reset link. The response is the same whether or not the account exists."""
    if not email_configured():
        raise HTTPException(status_code=503, detail="Password reset email is not configured")
    user = data_manager.find_user_by_identifier(payload.email)
    # Only match on email, not username, so the link goes to the account's own address.
    if user is not None and user.email == payload.email.lower().strip():
        token = data_manager.create_password_reset(user.id)
        if token is not None:
            base_url = os.getenv("APP_BASE_URL", "http://localhost:3000").rstrip("/")
            link = f"{base_url}/reset-password?token={token}"
            try:
                send_email(
                    user.email,
                    "Reset your SmartVision AI password",
                    "Someone asked to reset the password for your SmartVision AI account.\n\n"
                    f"Open this link within 30 minutes to choose a new password:\n{link}\n\n"
                    "If this wasn't you, ignore this email; your password will not change.\n",
                )
                data_manager.record_audit("PASSWORD_RESET_REQUESTED", user.id)
            except (EmailNotConfiguredError, OSError):
                logger.exception("Could not send password reset email")
    return {"success": True}


@app.post("/password-reset/confirm", dependencies=internal)
def confirm_password_reset(payload: PasswordResetConfirm):
    user_id = data_manager.consume_password_reset(payload.token, hash_password(payload.new_password))
    if user_id is None:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired")
    data_manager.record_audit("PASSWORD_RESET_COMPLETED", user_id)
    return {"success": True}


# Health probes (unauthenticated)
@app.get("/health")
async def health():
    return {"status": "ok", "service": "smart-vision-ai-persistence"}


@app.get("/health/live")
async def liveness():
    """Container liveness probe."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready")
def readiness():
    """Container readiness probe; fails when the database is unreachable."""
    try:
        stats = data_manager.get_stats()
    except SQLAlchemyError as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ready", "database": "connected", "active_sessions": stats.get("active_sessions", 0)}


# Crops and telemetry
@app.post("/crops", dependencies=internal)
def save_crop(payload: CropPayload):
    rec = data_manager.save_crop(payload.model_dump())
    return {
        "success": True,
        "crop": {
            "id": rec.id,
            "object_class": rec.object_class,
            "confidence": rec.confidence,
            "timestamp": rec.timestamp.isoformat(),
            "distance_m": rec.distance_m,
            "has_crop": rec.has_crop,
        },
    }


@app.get("/crops/latest", dependencies=internal)
def get_latest_crop(object_class: str | None = None):
    rec = data_manager.get_latest_crop(object_class=object_class)
    if not rec:
        return {"crop": None}
    return {
        "crop": {
            "id": rec.id,
            "object_class": rec.object_class,
            "confidence": rec.confidence,
            "timestamp": rec.timestamp.isoformat(),
            "distance_m": rec.distance_m,
            "has_crop": rec.has_crop,
            "payload": rec.payload,
        }
    }


@app.post("/detections", dependencies=internal)
def record_detection(payload: DetectionPayload):
    data_manager.record_detection(
        class_name=payload.class_name,
        confidence=payload.confidence,
        distance_m=payload.distance_m,
        has_crop=payload.has_crop,
    )
    return {"success": True}


# Detection alerts: POST a signed webhook when a watched class is detected.
# ALERT_WEBHOOK_URL enables them; ALERT_CLASSES (default "knife,fire") and
# ALERT_MIN_CONFIDENCE (default 0.5) select what counts. Each class alerts at
# most once per ALERT_COOLDOWN_SECONDS (default 60) per service process.
_alert_last_sent: dict[str, float] = {}


def _alerts_due(detections: list[DetectionPayload], now: float) -> list[DetectionPayload]:
    if not os.getenv("ALERT_WEBHOOK_URL"):
        return []
    watched = {
        name.strip() for name in os.getenv("ALERT_CLASSES", "knife,fire").split(",") if name.strip()
    }
    min_confidence = float(os.getenv("ALERT_MIN_CONFIDENCE", "0.5"))
    cooldown = float(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))
    due: dict[str, DetectionPayload] = {}
    for detection in detections:
        if detection.class_name not in watched or detection.confidence < min_confidence:
            continue
        if now - _alert_last_sent.get(detection.class_name, float("-inf")) < cooldown:
            continue
        best = due.get(detection.class_name)
        if best is None or detection.confidence > best.confidence:
            due[detection.class_name] = detection
    for class_name in due:
        _alert_last_sent[class_name] = now
    return list(due.values())


async def _send_alerts(alerts: list[DetectionPayload]) -> None:
    target = os.getenv("ALERT_WEBHOOK_URL", "")
    try:
        _validate_webhook_target(target)
    except HTTPException as exc:
        logger.warning("ALERT_WEBHOOK_URL rejected: %s", exc.detail)
        return
    for alert in alerts:
        delivered = await webhook_dispatcher.dispatch(
            target_url=target,
            event_type="detection.alert",
            data=alert.model_dump(),
        )
        data_manager.record_audit(
            "DETECTION_ALERT", None, object_class=alert.class_name,
            confidence=alert.confidence, delivered=delivered,
        )


async def _broadcast_detections(payload: DetectionBatchPayload) -> None:
    await ws_manager.broadcast_json({
        "type": "detections",
        "timestamp": time.time(),
        "detections": [d.model_dump() for d in payload.detections],
        "frame": payload.frame.model_dump() if payload.frame else None,
    })


@app.post("/detections/batch", dependencies=internal)
def record_detection_batch(payload: DetectionBatchPayload, background: BackgroundTasks):
    data_manager.record_detections(
        [detection.model_dump() for detection in payload.detections],
        frame=payload.frame.model_dump() if payload.frame else None,
    )
    if ws_manager.total_connections:
        background.add_task(_broadcast_detections, payload)
    alerts = _alerts_due(payload.detections, time.monotonic())
    if alerts:
        background.add_task(_send_alerts, alerts)
    return {"success": True, "count": len(payload.detections), "alerts": len(alerts)}


@app.get("/stats", dependencies=internal)
def get_stats():
    return data_manager.get_stats()


@app.post("/history/clear", dependencies=internal)
def clear_history():
    data_manager.clear_history()
    return {"success": True}


# Replay
@app.post("/replay/frames", dependencies=internal)
def save_replay_frame(payload: ReplayFramePayload):
    frame = data_manager.save_replay_frame(
        payload.session_id,
        payload.captured_at,
        [detection.model_dump(by_alias=True) for detection in payload.detections],
    )
    return {"success": True, "frame": frame}


@app.get("/replay/frames", dependencies=internal)
def get_replay_frames(session_id: str):
    return {"success": True, "frames": data_manager.get_replay_frames(session_id)}


@app.delete("/replay/frames", dependencies=internal)
def clear_replay_frames(session_id: str):
    data_manager.clear_replay_frames(session_id)
    return {"success": True}


# Observability
@app.get("/metrics", dependencies=internal)
def prometheus_metrics():
    """Prometheus text exposition of values the service actually measures."""
    stats = data_manager.get_stats()
    lines = [
        "# HELP smart_vision_total_detections Total detections recorded",
        "# TYPE smart_vision_total_detections counter",
        f"smart_vision_total_detections {stats.get('total_detections', 0)}",
        "# HELP smart_vision_active_sessions Unexpired user sessions",
        "# TYPE smart_vision_active_sessions gauge",
        f"smart_vision_active_sessions {stats.get('active_sessions', 0)}",
        "# HELP smart_vision_ws_connections Active WebSocket subscribers",
        "# TYPE smart_vision_ws_connections gauge",
        f"smart_vision_ws_connections {ws_manager.total_connections}",
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """Telemetry channel; requires the internal token as a query parameter."""
    expected_token = os.getenv("INTERNAL_API_TOKEN", "")
    supplied = websocket.query_params.get("token", "")
    if not expected_token or not hmac.compare_digest(supplied, expected_token):
        await websocket.close(code=1008)
        return
    await ws_manager.connect(websocket, channel="telemetry")
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, RuntimeError):
        ws_manager.disconnect(websocket)


@app.post("/webhooks/dispatch", dependencies=internal)
async def trigger_webhook(payload: WebhookRequest):
    _validate_webhook_target(payload.target_url)
    delivered = await webhook_dispatcher.dispatch(
        target_url=payload.target_url,
        event_type=payload.event_type,
        data=payload.data,
    )
    return {"delivered": delivered, "dead_letter_queue_size": len(webhook_dispatcher.dead_letter_queue)}


# Administration (called by the Next.js server, which checks the admin role)
@app.get("/admin/users", dependencies=internal)
def admin_list_users():
    return {"users": data_manager.list_users()}


@app.post("/admin/users/{user_id}/role", dependencies=internal)
def admin_set_role(user_id: str, payload: RoleChange):
    try:
        previous, new = data_manager.set_user_role(user_id, payload.role.value)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data_manager.record_audit("ROLE_CHANGED", user_id, previous=previous, new=new, actor=payload.actor_id, via="admin-ui")
    return {"success": True, "previous": previous, "role": new}


@app.post("/admin/users/{user_id}/verify-email", dependencies=internal)
def admin_verify_email(user_id: str, actor_id: str | None = None):
    if data_manager.get_user_by_id(user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    data_manager.mark_email_verified(user_id)
    data_manager.record_audit("EMAIL_VERIFIED", user_id, actor=actor_id, via="admin-ui")
    return {"success": True}


@app.get("/admin/audit", dependencies=internal)
def admin_audit(limit: int = 100):
    return {"events": data_manager.get_audit_logs(max(1, min(limit, 500)))}


@app.get("/admin/audit-logs")
def get_audit_logs(limit: int = 100, auth: Any = admin_users_dependency):
    """Most recent audit events. Restricted to users with AppPermission.ADMIN_USERS."""
    return {
        "success": True,
        "operator": auth.username,
        "role": auth.role,
        "events": data_manager.get_audit_logs(max(1, min(limit, 500))),
    }


# 3D mesh reconstruction
@app.post("/mesh/reconstruct", dependencies=internal)
def reconstruct_3d_mesh(payload: MeshReconstructionRequest):
    """Reconstruct a mesh from a supplied point cloud or a canonical object shape."""
    if payload.points and any(len(point) != 3 for point in payload.points):
        raise HTTPException(status_code=422, detail="Each point must have exactly 3 coordinates")
    try:
        if payload.points and len(payload.points) >= 4:
            pts = np.array(payload.points, dtype=np.float64)
        else:
            pts = CanonicalShapeGenerator.generate_object(payload.object_class or "chair")

        normals = PointCloudNormalEstimator(k_neighbors=8).estimate_normals(pts)

        voxel_sz = max(0.04, min(0.2, payload.voxel_size))
        builder = VolumetricTSDFBuilder(voxel_size=voxel_sz, truncation_distance=voxel_sz * 2.5)
        tsdf_grid, metadata = builder.build_tsdf(pts, normals)

        mesh = MarchingCubesReconstructor(isovalue=0.0).reconstruct(tsdf_grid, metadata)

        if payload.smoothing_iterations > 0:
            iterations = min(5, payload.smoothing_iterations)
            mesh = MeshPostProcessor.laplacian_smooth(mesh, iterations=iterations, lambda_factor=0.35)

        threejs_geometry = MeshExporter.to_threejs_json(mesh)
        response_data: dict[str, Any] = {
            "success": True,
            "object_class": payload.object_class,
            "format": payload.output_format,
            "threejs_geometry": threejs_geometry,
            "statistics": threejs_geometry["statistics"],
        }
        if payload.output_format == "obj":
            response_data["obj_data"] = MeshExporter.to_obj(
                mesh, object_name=payload.object_class or "ReconstructedMesh"
            )
        elif payload.output_format == "ply":
            response_data["ply_data"] = MeshExporter.to_ply(mesh)
        return response_data
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("Mesh reconstruction rejected input: %s", exc)
        raise HTTPException(status_code=422, detail="Point cloud could not be reconstructed") from exc
