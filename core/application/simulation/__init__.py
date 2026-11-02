"""Synthetic scenarios, camera generation, and load/benchmark utilities."""

from .camera_farm import SyntheticCameraFarm, SyntheticCameraFrame, SyntheticDetection
from .load_harness import (
    CameraStreamPlan,
    ChaosPolicy,
    LoadReport,
    StreamingLoadHarness,
)
from .tracking_metrics import TrackingMetrics, evaluate_tracking_sequence
from .tracking_scenario import (
    GroundTruthObject,
    SyntheticFrame,
    SyntheticTrackingScenario,
)

__all__ = [
    "CameraStreamPlan",
    "ChaosPolicy",
    "GroundTruthObject",
    "LoadReport",
    "StreamingLoadHarness",
    "SyntheticCameraFarm",
    "SyntheticCameraFrame",
    "SyntheticDetection",
    "SyntheticFrame",
    "SyntheticTrackingScenario",
    "TrackingMetrics",
    "evaluate_tracking_sequence",
]
