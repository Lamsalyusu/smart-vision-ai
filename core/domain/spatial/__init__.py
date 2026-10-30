"""Spatial coordinate, camera calibration, frustum geometry, multi-camera, bundle adjustment, and point-cloud value objects."""

from .bundle_adjustment import (
    BundleAdjustmentOptimizer,
    BundleAdjustmentResult,
    ReprojectionObservation,
)
from .calibration import (
    estimate_fundamental_matrix,
    estimate_homography,
    solve_zhang_intrinsics,
    symmetric_epipolar_distance,
)
from .frustum import (
    CameraIntrinsics,
    OrientedBox3D,
    backproject_frustum_box,
    estimate_surface_normal,
)
from .ground_plane import PlaneEstimate, fit_ground_plane_ransac
from .multicam import (
    CameraExtrinsics,
    CrossCameraAssociator,
    GlobalTrackMatch,
    MultiCameraObservation,
    triangulate_dlt,
)
from .point_cloud import BoundingBox3D, PointCloud3D
from .transforms import CoordinateFrame, Quaternion, Transform3D, Vector3

__all__ = [
    "BoundingBox3D",
    "BundleAdjustmentOptimizer",
    "BundleAdjustmentResult",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "CoordinateFrame",
    "CrossCameraAssociator",
    "GlobalTrackMatch",
    "MultiCameraObservation",
    "OrientedBox3D",
    "PlaneEstimate",
    "PointCloud3D",
    "Quaternion",
    "ReprojectionObservation",
    "Transform3D",
    "Vector3",
    "backproject_frustum_box",
    "estimate_fundamental_matrix",
    "estimate_homography",
    "estimate_surface_normal",
    "fit_ground_plane_ransac",
    "solve_zhang_intrinsics",
    "symmetric_epipolar_distance",
    "triangulate_dlt",
]