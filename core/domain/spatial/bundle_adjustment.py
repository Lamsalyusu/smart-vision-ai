import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.domain.spatial.transforms import Vector3


@dataclass(frozen=True, slots=True)
class ReprojectionObservation:
    camera_index: int
    landmark_index: int
    uv_observed: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BundleAdjustmentResult:
    optimized_landmarks: list[Vector3]
    initial_mean_error_px: float
    final_mean_error_px: float
    iterations: int
    converged: bool


class BundleAdjustmentOptimizer:
    """Non-linear least squares multi-view bundle adjustment optimizer with Huber robust loss

    for refining 3D spatial landmark estimates and camera reprojection errors.
    """

    def __init__(
        self,
        max_iterations: int = 15,
        huber_delta_px: float = 2.0,
        tolerance: float = 1e-4,
    ) -> None:
        self.max_iterations = max_iterations
        self.huber_delta_px = huber_delta_px
        self.tolerance = tolerance

    def _huber_weight(self, error: float) -> float:
        """Compute Huber loss derivative weight."""
        abs_err = abs(error)
        if abs_err <= self.huber_delta_px:
            return 1.0
        return self.huber_delta_px / abs_err

    def optimize_landmarks(
        self,
        projection_matrices: Sequence[np.ndarray],
        initial_landmarks: Sequence[Vector3],
        observations: Sequence[ReprojectionObservation],
    ) -> BundleAdjustmentResult:
        """Refine 3D landmark positions given fixed camera projection matrices and 2D pixel observations."""
        num_landmarks = len(initial_landmarks)
        if num_landmarks == 0 or len(observations) == 0:
            return BundleAdjustmentResult([], 0.0, 0.0, 0, True)

        # Landmarks array of shape (N, 3)
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in initial_landmarks], dtype=np.float64
        )

        def compute_errors(
            lm_array: np.ndarray,
        ) -> tuple[np.ndarray, list[float]]:
            residuals = []
            pixel_errors = []
            for obs in observations:
                proj = projection_matrices[obs.camera_index]
                x_3d = lm_array[obs.landmark_index]
                homog = np.array([x_3d[0], x_3d[1], x_3d[2], 1.0])

                p_img = proj @ homog
                if p_img[2] <= 1e-6:
                    u_proj, v_proj = 0.0, 0.0
                else:
                    u_proj = p_img[0] / p_img[2]
                    v_proj = p_img[1] / p_img[2]

                du = u_proj - obs.uv_observed[0]
                dv = v_proj - obs.uv_observed[1]
                err = math.sqrt(du**2 + dv**2)
                pixel_errors.append(err)

                w = self._huber_weight(err)
                residuals.extend([math.sqrt(w) * du, math.sqrt(w) * dv])

            return np.array(residuals, dtype=np.float64), pixel_errors

        initial_residuals, initial_errs = compute_errors(landmarks)
        initial_mean = float(np.mean(initial_errs)) if initial_errs else 0.0
        prev_cost = float(np.sum(initial_residuals**2))

        converged = False
        iteration = 0

        # Gauss-Newton / Levenberg-Marquardt iteration
        for it in range(self.max_iterations):
            iteration = it + 1
            residuals, _pixel_errs = compute_errors(landmarks)
            current_cost = float(np.sum(residuals**2))

            if it > 0 and abs(prev_cost - current_cost) < self.tolerance:
                converged = True
                break

            # Numeric Jacobian approximation per landmark: (2 * num_obs, 3 * num_landmarks)
            # Since each observation only depends on its own landmark, we update per landmark
            step = 1e-5
            for lm_idx in range(num_landmarks):
                # Gradient update for landmark lm_idx
                grad = np.zeros(3, dtype=np.float64)
                diag_h = np.zeros(3, dtype=np.float64)

                obs_lm = [
                    obs
                    for obs in observations
                    if obs.landmark_index == lm_idx
                ]
                if not obs_lm:
                    continue

                for dim in range(3):
                    lm_plus = landmarks.copy()
                    lm_plus[lm_idx, dim] += step
                    res_plus, _ = compute_errors(lm_plus)
                    j_col = (res_plus - residuals) / step
                    grad[dim] = float(np.dot(j_col, residuals))
                    diag_h[dim] = float(np.dot(j_col, j_col)) + 1e-4

                delta = -grad / diag_h
                # Limit step length
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > 0.5:
                    delta = (delta / delta_norm) * 0.5

                landmarks[lm_idx] += delta

            prev_cost = current_cost

        _, final_errs = compute_errors(landmarks)
        final_mean = float(np.mean(final_errs)) if final_errs else 0.0

        refined = [
            Vector3(float(row[0]), float(row[1]), float(row[2]))
            for row in landmarks
        ]

        return BundleAdjustmentResult(
            optimized_landmarks=refined,
            initial_mean_error_px=initial_mean,
            final_mean_error_px=final_mean,
            iterations=iteration,
            converged=converged,
        )

    @staticmethod
    def compute_depth_covariance(
        depth_m: float,
        baseline_m: float,
        focal_length_px: float,
        disparity_variance_px2: float = 0.5,
    ) -> float:
        """Calculate geometric depth estimation variance: sigma_Z^2 = (Z^2 / (f * B))^2 * sigma_d^2."""
        if baseline_m <= 0 or focal_length_px <= 0 or depth_m <= 0:
            raise ValueError(
                "baseline, focal length, and depth must be strictly positive"
            )
        factor = (depth_m**2) / (focal_length_px * baseline_m)
        return float((factor**2) * disparity_variance_px2)
