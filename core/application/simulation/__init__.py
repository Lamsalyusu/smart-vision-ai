"""Synthetic scenarios, 3D multi-camera simulation, and tracking benchmark utilities."""

from .ingestion_stress import IngestionStressReport, run_ingestion_stress
from .synthetic_scene import (
    GroundTruthTarget3D,
    SimulatedDetection,
    Synthetic3DSceneSimulator,
    SyntheticSceneFrame,
)
from .tracking_metrics import TrackingMetrics, evaluate_tracking_sequence
from .tracking_scenario import (
    GroundTruthObject,
    SyntheticFrame,
    SyntheticTrackingScenario,
)

__all__ = [
    "GroundTruthObject",
    "GroundTruthTarget3D",
    "IngestionStressReport",
    "SimulatedDetection",
    "Synthetic3DSceneSimulator",
    "SyntheticFrame",
    "SyntheticSceneFrame",
    "SyntheticTrackingScenario",
    "TrackingMetrics",
    "evaluate_tracking_sequence",
    "run_ingestion_stress",
]