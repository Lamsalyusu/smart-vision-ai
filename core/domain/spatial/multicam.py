import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.domain.spatial.frustum import CameraIntrinsics
from core.domain.spatial.transforms import Vector3
from core.domain.tracking.reid import AppearanceEmbedding


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Rigid 3D transformation [R | t] from world coordinates to camera coordinates."""

    rotation: tuple[tuple[float, float, float], ...]  # 3x3 rotation matrix
    translation: tuple[float, float, float]  # 3x1 translation vector

    def __post_init__(self) -> None:
        if len(self.rotation) != 3 or any(len(r) != 3 for r in self.rotation):
            raise ValueError("rotation must be a 3x3 matrix")
        if len(self.translation) != 3:
            raise ValueError("translation must be a 3-element vector")
        r_mat = np.asarray(self.rotation, dtype=np.float64)
        if not np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-5):
            raise ValueError("rotation matrix must be orthogonal")

    def projection_matrix(self, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Compute the 3x4 camera projection matrix P = K * [R | t]."""
        k_mat = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rt_mat = np.column_stack(
            (
                np.asarray(self.rotation, dtype=np.float64),
                np.asarray(self.translation, dtype=np.float64),
            )
        )
        return k_mat @ rt_mat


def triangulate_dlt(
    projection_matrices: Sequence[np.ndarray],
    point_observations: Sequence[tuple[float, float]],
) -> Vector3:
    """Triangulate 3D point in world coordinates from multiple calibrated camera views

    using the Direct Linear Transform (DLT) algorithm with SVD.
    """
    if len(projection_matrices) < 2 or len(projection_matrices) != len(
        point_observations
    ):
        raise ValueError(
            "at least two synchronized camera views required for triangulation"
        )

    rows = []
    for proj, (u, v) in zip(
        projection_matrices, point_observations, strict=True
    ):
        p1 = proj[0, :]
        p2 = proj[1, :]
        p3 = proj[2, :]
        rows.append(u * p3 - p1)
        rows.append(v * p3 - p2)

    design = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(design)
    x_homog = vt[-1]  # Solution is eigenvector corresponding to smallest singular value

    if abs(x_homog[3]) < 1e-9:
        raise ValueError("triangulated point is at infinity or degenerate")

    x_world = x_homog[:3] / x_homog[3]
    return Vector3(float(x_world[0]), float(x_world[1]), float(x_world[2]))


@dataclass(frozen=True, slots=True)
class MultiCameraObservation:
    camera_id: str
    track_id: int
    pixel_center: tuple[float, float]
    appearance: AppearanceEmbedding | None = None


@dataclass(frozen=True, slots=True)
class GlobalTrackMatch:
    global_track_id: str
    camera_observations: tuple[MultiCameraObservation, ...]
    triangulated_position: Vector3 | None


class CrossCameraAssociator:
    """Associates local camera tracks into globally consistent identities across overlapping views."""

    def __init__(
        self,
        epipolar_threshold_px: float = 15.0,
        appearance_threshold: float = 0.45,
        weight_appearance: float = 0.6,
    ) -> None:
        self.epipolar_threshold_px = epipolar_threshold_px
        self.appearance_threshold = appearance_threshold
        self.weight_appearance = weight_appearance
        self._next_global_id = 1

    def compute_epipolar_distance(
        self,
        fundamental_matrix: np.ndarray,
        p_cam1: tuple[float, float],
        p_cam2: tuple[float, float],
    ) -> float:
        """Compute point-to-epipolar-line distance: d(p2, F * p1)."""
        p1_homog = np.array([p_cam1[0], p_cam1[1], 1.0], dtype=np.float64)
        p2_homog = np.array([p_cam2[0], p_cam2[1], 1.0], dtype=np.float64)

        line2 = fundamental_matrix @ p1_homog
        line_norm = math.sqrt(line2[0] ** 2 + line2[1] ** 2)
        if line_norm < 1e-12:
            return math.inf

        # Perpendicular distance from p2 to line2
        dist = abs(float(np.dot(line2, p2_homog))) / line_norm
        return dist

    def associate_pair(
        self,
        fundamental_matrix: np.ndarray,
        cam1_obs: Sequence[MultiCameraObservation],
        cam2_obs: Sequence[MultiCameraObservation],
    ) -> list[tuple[MultiCameraObservation, MultiCameraObservation, float]]:
        """Find matching pairs between two camera views satisfying epipolar and appearance constraints."""
        matches: list[
            tuple[MultiCameraObservation, MultiCameraObservation, float]
        ] = []

        for obs1 in cam1_obs:
            best_match: MultiCameraObservation | None = None
            min_cost = math.inf

            for obs2 in cam2_obs:
                epi_dist = self.compute_epipolar_distance(
                    fundamental_matrix, obs1.pixel_center, obs2.pixel_center
                )
                if epi_dist > self.epipolar_threshold_px:
                    continue

                app_dist = 0.0
                if obs1.appearance and obs2.appearance:
                    app_dist = obs1.appearance.cosine_distance(obs2.appearance)
                    if app_dist > self.appearance_threshold:
                        continue

                norm_epi = min(1.0, epi_dist / self.epipolar_threshold_px)
                cost = (
                    self.weight_appearance * app_dist
                    + (1.0 - self.weight_appearance) * norm_epi
                )

                if cost < min_cost:
                    min_cost = cost
                    best_match = obs2

            if best_match is not None:
                matches.append((obs1, best_match, min_cost))

        return matches
