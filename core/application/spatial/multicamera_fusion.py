from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from core.domain.spatial.multicam import (
    CrossCameraAssociator,
    GlobalTrackMatch,
    MultiCameraObservation,
    triangulate_dlt,
)
from core.domain.spatial.transforms import Vector3


class MultiCameraFusionCoordinator:
    """Associate synchronized camera tracks and triangulate globally matched observations."""

    def __init__(self, associator: CrossCameraAssociator | None = None) -> None:
        self._associator = associator or CrossCameraAssociator()
        self._fundamental_matrices: dict[tuple[str, str], np.ndarray] = {}
        self._projection_matrices: dict[str, np.ndarray] = {}
        self._global_ids: dict[tuple[str, int], str] = {}
        self._next_global_id = 1
        self._last_frame_sequence = -1

    def register_projection(self, camera_id: str, projection_matrix: np.ndarray) -> None:
        projection = np.asarray(projection_matrix, dtype=np.float64)
        if not camera_id or projection.shape != (3, 4) or not np.isfinite(projection).all():
            raise ValueError("camera id and a finite 3x4 projection matrix are required")
        self._projection_matrices[camera_id] = projection.copy()

    def register_pair_calibration(
        self,
        camera_a: str,
        camera_b: str,
        fundamental_matrix: np.ndarray,
    ) -> None:
        matrix = np.asarray(fundamental_matrix, dtype=np.float64)
        if not camera_a or not camera_b or camera_a == camera_b:
            raise ValueError("pair calibration requires two distinct camera ids")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("fundamental matrix must be finite and 3x3")
        self._fundamental_matrices[(camera_a, camera_b)] = matrix.copy()

    def forget_track(self, camera_id: str, track_id: int) -> None:
        self._global_ids.pop((camera_id, track_id), None)

    def fuse_synchronized_frame(
        self,
        frame_sequence: int,
        observations_by_camera: Mapping[str, Sequence[MultiCameraObservation]],
    ) -> tuple[GlobalTrackMatch, ...]:
        if frame_sequence <= self._last_frame_sequence:
            raise ValueError("frame sequence must increase monotonically")
        self._last_frame_sequence = frame_sequence

        observations: dict[tuple[str, int], MultiCameraObservation] = {}
        by_camera: dict[str, tuple[MultiCameraObservation, ...]] = {}
        for camera_id, camera_observations in observations_by_camera.items():
            normalized = tuple(camera_observations)
            if any(observation.camera_id != camera_id for observation in normalized):
                raise ValueError("observation camera id must match its camera group")
            track_ids = [observation.track_id for observation in normalized]
            if len(track_ids) != len(set(track_ids)):
                raise ValueError("camera frame contains duplicate local track ids")
            by_camera[camera_id] = normalized
            observations.update(
                {(camera_id, observation.track_id): observation for observation in normalized}
            )

        candidate_edges: list[tuple[float, tuple[str, int], tuple[str, int]]] = []
        for (camera_a, camera_b), fundamental in self._fundamental_matrices.items():
            if camera_a not in by_camera or camera_b not in by_camera:
                continue
            for observation_a, observation_b, cost in self._associator.associate_pair(
                fundamental,
                by_camera[camera_a],
                by_camera[camera_b],
            ):
                candidate_edges.append(
                    (
                        cost,
                        (camera_a, observation_a.track_id),
                        (camera_b, observation_b.track_id),
                    )
                )

        parent = {key: key for key in observations}
        cameras = {key: {key[0]} for key in observations}

        def find(key: tuple[str, int]) -> tuple[str, int]:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for _, left, right in sorted(candidate_edges, key=lambda edge: (edge[0], edge[1], edge[2])):
            root_left = find(left)
            root_right = find(right)
            if root_left == root_right or cameras[root_left] & cameras[root_right]:
                continue
            parent[root_right] = root_left
            cameras[root_left].update(cameras.pop(root_right))

        components: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for key in observations:
            components.setdefault(find(key), []).append(key)

        results = []
        for members in sorted(components.values(), key=lambda group: min(group)):
            known_ids = sorted({self._global_ids[key] for key in members if key in self._global_ids})
            global_id = known_ids[0] if known_ids else f"global-{self._next_global_id}"
            if not known_ids:
                self._next_global_id += 1
            for old_id in known_ids[1:]:
                self._global_ids = {
                    key: global_id if value == old_id else value
                    for key, value in self._global_ids.items()
                }
            for key in members:
                self._global_ids[key] = global_id

            group_observations = tuple(observations[key] for key in sorted(members))
            position = self._triangulate(group_observations)
            results.append(GlobalTrackMatch(global_id, group_observations, position))

        return tuple(results)

    def _triangulate(
        self, observations: Sequence[MultiCameraObservation]
    ) -> Vector3 | None:
        calibrated = [
            (self._projection_matrices[observation.camera_id], observation.pixel_center)
            for observation in observations
            if observation.camera_id in self._projection_matrices
        ]
        if len(calibrated) < 2:
            return None
        try:
            return triangulate_dlt(
                [projection for projection, _ in calibrated],
                [pixel for _, pixel in calibrated],
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
