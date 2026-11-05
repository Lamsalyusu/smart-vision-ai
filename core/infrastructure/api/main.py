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
from ..security.rbac import AppPermission, require_permission
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


class DetectionBatchPayload(BaseModel):
    detections: list[DetectionPayload] = Field(max_length=500)


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
    }


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
@app.post("/login", dependencies=internal)
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
    data_manager.record_audit("LOGIN_SUCCESS", user.id)
    return _login_response(response, user)


@app.post("/signup", dependencies=internal)
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
    return {**_login_response(response, user), "user_id": user.id}


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


@app.post("/password-reset/request", dependencies=internal)
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


@app.post("/detections/batch", dependencies=internal)
def record_detection_batch(payload: DetectionBatchPayload):
    data_manager.record_detections(
        [detection.model_dump() for detection in payload.detections]
    )
    return {"success": True, "count": len(payload.detections)}


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
