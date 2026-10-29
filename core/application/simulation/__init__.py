"""Synthetic scenarios and tracking benchmark utilities."""

from .tracking_metrics import TrackingMetrics, evaluate_tracking_sequence
from .tracking_scenario import GroundTruthObject, SyntheticFrame, SyntheticTrackingScenario
from .ingestion_stress import IngestionStressReport, run_ingestion_stress

__all__ = [
    "GroundTruthObject",
    "SyntheticFrame",
    "SyntheticTrackingScenario",
    "TrackingMetrics",
    "evaluate_tracking_sequence",
    "IngestionStressReport",
    "run_ingestion_stress",
]