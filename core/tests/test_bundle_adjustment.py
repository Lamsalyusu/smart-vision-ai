import unittest

from core.domain.spatial.bundle_adjustment import (
    BundleAdjustmentOptimizer,
    ReprojectionObservation,
)
from core.domain.spatial.frustum import CameraIntrinsics
from core.domain.spatial.multicam import CameraExtrinsics
from core.domain.spatial.transforms import Vector3


class BundleAdjustmentTests(unittest.TestCase):
    def test_bundle_adjustment_reduces_reprojection_error(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

        # 2 camera poses
        ext1 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        )
        ext2 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (-1.0, 0.0, 0.0),
        )

        p1 = ext1.projection_matrix(intrinsics)
        p2 = ext2.projection_matrix(intrinsics)

        # Ground truth 3D landmark at (0.0, 0.0, 5.0)
        # Cam1 sees it at (640, 360)
        # Cam2 sees it at (-200 + 640, 360) = (440, 360)
        obs = [
            ReprojectionObservation(camera_index=0, landmark_index=0, uv_observed=(640.0, 360.0)),
            ReprojectionObservation(camera_index=1, landmark_index=0, uv_observed=(440.0, 360.0)),
        ]

        # Initial noisy landmark displaced to (0.3, -0.2, 5.5)
        initial_landmarks = [Vector3(0.3, -0.2, 5.5)]

        optimizer = BundleAdjustmentOptimizer(max_iterations=20)
        result = optimizer.optimize_landmarks([p1, p2], initial_landmarks, obs)

        # Reprojection error should be significantly reduced
        self.assertLess(result.final_mean_error_px, result.initial_mean_error_px)

        # Refined landmark should be very close to ground truth (0.0, 0.0, 5.0)
        refined = result.optimized_landmarks[0]
        self.assertAlmostEqual(refined.x, 0.0, places=1)
        self.assertAlmostEqual(refined.y, 0.0, places=1)
        self.assertAlmostEqual(refined.z, 5.0, places=1)

    def test_depth_covariance_computation(self) -> None:
        cov = BundleAdjustmentOptimizer.compute_depth_covariance(
            depth_m=5.0,
            baseline_m=1.0,
            focal_length_px=1000.0,
            disparity_variance_px2=0.5,
        )
        # (5.0^2 / 1000)^2 * 0.5 = (0.025)^2 * 0.5 = 0.000625 * 0.5 = 0.0003125
        self.assertAlmostEqual(cov, 0.0003125, places=6)


if __name__ == "__main__":
    unittest.main()
