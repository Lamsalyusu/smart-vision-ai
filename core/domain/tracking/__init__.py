"""Tracking state, measurement-association primitives, and visual Re-ID."""

from .association import mahalanobis_squared
from .reid import (
    AppearanceEmbedding,
    FeatureGallery,
    composite_association_cost,
    compute_reid_cost_matrix,
)
from .state import TrackKinematicLimits, TrackLifecycle, TrackState3D, TrackStatus
from .ukf import UnscentedKalmanFilter3D

__all__ = [
    "AppearanceEmbedding",
    "FeatureGallery",
    "TrackKinematicLimits",
    "TrackLifecycle",
    "TrackState3D",
    "TrackStatus",
    "UnscentedKalmanFilter3D",
    "composite_association_cost",
    "compute_reid_cost_matrix",
    "mahalanobis_squared",
]