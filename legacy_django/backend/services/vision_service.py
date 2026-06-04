import time
import logging
from backend.ml_model.inference import detector
from backend.ml_model.tracker import SortTracker
from backend.ml_model.spatial_fusion import spatial_fuser
from backend.repositories.detection_buffer import detection_buffer

logger = logging.getLogger(__name__)


class VisionPipelineService:
    """Coordinates object detection, temporal tracking, and spatial estimation."""

    def __init__(self, max_idle_sessions=100):
        self._trackers = {}
        self._tracker_last_seen = {}
        self._max_idle_sessions = max_idle_sessions

    def _get_tracker(self, session_key):
        """Retrieve or initialize a SORT tracker for the active client session."""
        now = time.time()
        # Clean up stale sessions idle for > 15 minutes
        if len(self._trackers) > self._max_idle_sessions:
            stale_keys = [
                k for k, last_t in self._tracker_last_seen.items()
                if (now - last_t) > 900
            ]
            for k in stale_keys:
                self._trackers.pop(k, None)
                self._tracker_last_seen.pop(k, None)

        if session_key not in self._trackers:
            self._trackers[session_key] = SortTracker(max_age=15, min_hits=2, iou_threshold=0.3)

        self._tracker_last_seen[session_key] = now
        return self._trackers[session_key]

    def process_frame(self, image_bytes, session_key, user_id=None, include_depth=False):
        """
        Run detection, track across frames, and optionally compute 3D spatial properties.
        """
        raw_detections = detector.detect_from_bytes(image_bytes)

        # Apply multi-object tracking
        tracker = self._get_tracker(session_key)
        tracked_detections = tracker.update(raw_detections)

        # Fallback to raw detections if tracking hits haven't accumulated yet
        detections = tracked_detections if tracked_detections else raw_detections

        # Optionally estimate metric 3D attributes
        if include_depth and detections:
            try:
                from backend.ml_model.depth_estimation import depth_estimator
                depth_map, img_rgb = depth_estimator.estimate_depth(image_bytes)
                detections = spatial_fuser.estimate_spatial_metrics(
                    detections, depth_map, img_rgb.shape
                )
            except Exception:
                logger.exception("Optional depth fusion failed during frame processing")

        # Asynchronously queue detections for database persistence
        if user_id and detections:
            for det in detections:
                detection_buffer.add(user_id, det)

        return detections


vision_service = VisionPipelineService()
