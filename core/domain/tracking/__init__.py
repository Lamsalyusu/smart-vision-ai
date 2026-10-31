"""Tracking state, measurement-association primitives, and visual Re-ID."""

from .association import mahalanobis_squared
from .reid import (
    AppearanceEmbedding,
    FeatureGallery,
    composite_association_cost,
    compute_reid_cost_matrix,
)
from .state import TrackState3D
from .ukf import UnscentedKalmanFilter3D

__all__ = [
    "AppearanceEmbedding",
    "FeatureGallery",
    "TrackState3D",
    "UnscentedKalmanFilter3D",
    "composite_association_cost",
    "compute_reid_cost_matrix",
    "mahalanobis_squared",
]