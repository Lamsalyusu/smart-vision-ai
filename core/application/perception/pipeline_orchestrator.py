import time
from collections.abc import Sequence
from dataclasses import dataclass

from core.application.perception.build_detection_batch import BuildDetectionBatch
from core.application.spatial.temporal_scene_graph import (
    SpatialEntity,
    TemporalSceneGraph,
    TemporalSceneSnapshot,
)
from core.domain.perception import DetectionBatch
from core.domain.spatial.frustum import CameraIntrinsics, backproject_frustum_box
from core.domain.tracking.reid import AppearanceEmbedding, FeatureGallery
from core.domain.tracking.state import TrackState3D
from core.domain.tracking.ukf import UnscentedKalmanFilter3D
from core.infrastructure.resilience.fallback_engine import (
    BackendTier,
    ResilientInferencePipeline,
)


@dataclass(frozen=True, slots=True)
class PipelineExecutionSnapshot:
    camera_id: str
    frame_sequence: int
    backend_tier: BackendTier
    fallback_occurred: bool
    latency_ms: float
    detection_batch: DetectionBatch
    active_tracks_count: int
    scene_snapshot: TemporalSceneSnapshot


class PerceptionPipelineOrchestrator:
    """Full-lifecycle spatial perception pipeline connecting inference, UKF tracking,

    3D frustum backprojection, and temporal scene graph construction.
    """

    def __init__(
        self,
        camera_id: str,
        inference_pipeline: ResilientInferencePipeline,
        intrinsics: CameraIntrinsics | None = None,
        *,
        history_capacity: int = 120,
    ) -> None:
        self.camera_id = camera_id
        self._inference_pipeline = inference_pipeline
        self._intrinsics = intrinsics or CameraIntrinsics(
            fx=1000.0, fy=1000.0, cx=960.0, cy=540.0
        )
        self._batch_builder = BuildDetectionBatch()
        self._scene_graph = TemporalSceneGraph(history_capacity=history_capacity)

        # Track management state
        self._next_track_id = 1
        self._track_filters: dict[int, UnscentedKalmanFilter3D] = {}
        self._track_galleries: dict[int, FeatureGallery] = {}
        self._track_classes: dict[int, str] = {}
        self._track_last_seen: dict[int, int] = {}

    def process_frame(
        self,
        frame_sequence: int,
        frame_data: object,
        timestamp_seconds: float,
        simulated_embeddings: Sequence[Sequence[float]] | None = None,
        estimated_depth_m: float = 3.0,
    ) -> PipelineExecutionSnapshot:
        start_time = time.monotonic()

        # 1. Resilient Inference Execution
        inf_result = self._inference_pipeline.execute(frame_data)
        raw_detections = inf_result.output
        if not isinstance(raw_detections, list):
            raw_detections = []

        # 2. Translate to validated Domain Batch
        detection_batch = self._batch_builder.execute(
            camera_id=self.camera_id,
            frame_sequence=frame_sequence,
            detections=raw_detections,
        )

        # 3. 3D Spatial Back-projection & Tracking association
        spatial_entities: list[SpatialEntity] = []

        for idx, det in enumerate(detection_batch.detections):
            # Back-project 3D frustum box
            obb = backproject_frustum_box(
                x1=det.bbox.x1,
                y1=det.bbox.y1,
                x2=det.bbox.x2,
                y2=det.bbox.y2,
                depth_m=estimated_depth_m,
                depth_span_m=0.5,
                intrinsics=self._intrinsics,
            )

            # Assign or update UKF tracker
            track_id = det.track_id if det.track_id is not None else self._next_track_id
            if track_id not in self._track_filters:
                # Initialize new 3D UKF
                init_state = TrackState3D(
                    x=obb.center.x,
                    y=obb.center.y,
                    z=obb.center.z,
                    velocity_x=0.0,
                    velocity_y=0.0,
                    velocity_z=0.0,
                    heading_rad=0.0,
                    heading_rate_radps=0.0,
                )
                self._track_filters[track_id] = UnscentedKalmanFilter3D(init_state)
                self._track_galleries[track_id] = FeatureGallery()
                self._track_classes[track_id] = det.class_name
                if det.track_id is None:
                    self._next_track_id += 1
            else:
                # Update existing UKF filter
                ukf = self._track_filters[track_id]
                ukf.predict(0.033)  # ~30 FPS dt
                ukf.update((obb.center.x, obb.center.y, obb.center.z, 0.0))

            self._track_last_seen[track_id] = frame_sequence

            # Update Re-ID embedding gallery if provided
            if simulated_embeddings and idx < len(simulated_embeddings):
                emb = AppearanceEmbedding.from_raw(simulated_embeddings[idx])
                self._track_galleries[track_id].update(emb)

            current_pos = self._track_filters[track_id].state
            spatial_entities.append(
                SpatialEntity(
                    track_id=track_id,
                    class_name=det.class_name,
                    position_xyz_m=(current_pos.x, current_pos.y, current_pos.z),
                    bbox_xyxy=(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                    confidence=det.confidence.value,
                )
            )

        # 4. Prune stale tracks (>30 frames unseen)
        stale_ids = [
            t_id
            for t_id, last_seq in self._track_last_seen.items()
            if frame_sequence - last_seq > 30
        ]
        for t_id in stale_ids:
            del self._track_filters[t_id]
            del self._track_galleries[t_id]
            del self._track_classes[t_id]
            del self._track_last_seen[t_id]

        # 5. Update Temporal Scene Graph
        scene_snapshot = self._scene_graph.update(
            timestamp_seconds=timestamp_seconds,
            entities=spatial_entities,
        )

        latency_ms = (time.monotonic() - start_time) * 1000.0

        return PipelineExecutionSnapshot(
            camera_id=self.camera_id,
            frame_sequence=frame_sequence,
            backend_tier=inf_result.tier_used,
            fallback_occurred=inf_result.fallback_occurred,
            latency_ms=latency_ms,
            detection_batch=detection_batch,
            active_tracks_count=len(self._track_filters),
            scene_snapshot=scene_snapshot,
        )
