from __future__ import annotations

from typing import Any, Dict, List, Optional
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
    depth_m: Optional[float] = None
    spatial_coords: Optional[List[float]] = None


class SpatialTrackRecord(BaseModel):
    track_id: int
    class_name: str
    state: str
    confidence: float
    position_3d: List[float] = Field(default_factory=list)
    velocity_3d: List[float] = Field(default_factory=list)
    age_frames: int = 1


class MeshReconstructionResult(BaseModel):
    success: bool
    object_class: str
    vertex_count: int
    face_count: int
    surface_area_sq_m: float
    obj_data: Optional[str] = None
    threejs_geometry: Optional[Dict[str, Any]] = None


class TelemetryPacket(BaseModel):
    timestamp_ns: int
    fps: float
    active_tracks: int
    inference_latency_ms: float
    drift_score: float
    geofence_breaches: List[Dict[str, Any]] = Field(default_factory=list)
