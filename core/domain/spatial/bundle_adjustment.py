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
        max_iterations: int = 20,
        huber_delta_px: float = 2.0,
        tolerance: float = 1e-4,
        initial_lambda: float = 0.0,
        robust_loss: str = "huber",
    ) -> None:
        self.max_iterations = max_iterations
        self.huber_delta_px = huber_delta_px
        self.tolerance = tolerance
        self.initial_lambda = initial_lambda
        self.robust_loss = robust_loss

    def _robust_weight(self, error: float) -> float:
        """Compute robust M-estimator weight (Huber or Tukey biweight)."""
        abs_err = abs(error)
        if self.robust_loss == "tukey":
            c = self.huber_delta_px * 2.5
            if abs_err >= c:
                return 0.0
            return float((1.0 - (abs_err / c) ** 2) ** 2)
        # Huber loss
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

                w = self._robust_weight(err)
                residuals.extend([math.sqrt(w) * du, math.sqrt(w) * dv])

            return np.array(residuals, dtype=np.float64), pixel_errors

        initial_residuals, initial_errs = compute_errors(landmarks)
        initial_mean = float(np.mean(initial_errs)) if initial_errs else 0.0
        current_cost = float(np.sum(initial_residuals**2))

        converged = False
        iteration = 0
        damping = self.initial_lambda

        # Levenberg-Marquardt iteration with adaptive damping lambda
        step = 1e-5
        for it in range(self.max_iterations):
            iteration = it + 1
            residuals, _ = compute_errors(landmarks)
            candidate_landmarks = landmarks.copy()

            for lm_idx in range(num_landmarks):
                obs_lm = [
                    obs for obs in observations if obs.landmark_index == lm_idx
                ]
                if not obs_lm:
                    continue

                grad = np.zeros(3, dtype=np.float64)
                hessian = np.zeros((3, 3), dtype=np.float64)

                j_cols = []
                for dim in range(3):
                    lm_plus = landmarks.copy()
                    lm_plus[lm_idx, dim] += step
                    res_plus, _ = compute_errors(lm_plus)
                    j_col = (res_plus - residuals) / step
                    j_cols.append(j_col)
                    grad[dim] = float(np.dot(j_col, residuals))

                for r in range(3):
                    for c in range(3):
                        hessian[r, c] = float(np.dot(j_cols[r], j_cols[c]))

                # Levenberg-Marquardt diagonally-scaled gradient step
                diag_h = np.array([float(np.dot(j_cols[d], j_cols[d])) + 1e-4 for d in range(3)])
                delta = -grad / (diag_h + damping)

                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > 0.5:
                    delta = (delta / delta_norm) * 0.5

                candidate_landmarks[lm_idx] += delta

            # Evaluate step quality
            cand_residuals, _ = compute_errors(candidate_landmarks)
            candidate_cost = float(np.sum(cand_residuals**2))

            if candidate_cost < current_cost:
                rel_improvement = (current_cost - candidate_cost) / max(current_cost, 1.0)
                landmarks = candidate_landmarks
                current_cost = candidate_cost
                damping = max(damping * 0.5, 1e-7)
                if it > 2 and rel_improvement < self.tolerance:
                    converged = True
                    break
            else:
                damping = min(damping * 2.0, 1e4)

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
