from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BoundingBoxModel(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionRecord(BaseModel):
    id: str
    class_name: str
    confidence: float
    bbox: BoundingBoxModel
    depth_m: float | None = None
    spatial_coords: list[float] | None = None


class SpatialTrackRecord(BaseModel):
    track_id: int
    class_name: str
    state: str
    confidence: float
    position_3d: list[float] = Field(default_factory=list)
    velocity_3d: list[float] = Field(default_factory=list)
    age_frames: int = 1


class MeshReconstructionResult(BaseModel):
    success: bool
    object_class: str
    vertex_count: int
    face_count: int
    surface_area_sq_m: float
    obj_data: str | None = None
    threejs_geometry: dict[str, Any] | None = None


class TelemetryPacket(BaseModel):
    timestamp_ns: int
    fps: float
    active_tracks: int
    inference_latency_ms: float
    drift_score: float
    geofence_breaches: list[dict[str, Any]] = Field(default_factory=list)
