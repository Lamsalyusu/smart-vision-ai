import unittest

import numpy as np

from core.domain.spatial.point_cloud import PointCloud3D


class PointCloud3DTests(unittest.TestCase):
    def test_point_cloud_bounding_box(self) -> None:
        pts = np.array(
            [
                [-1.0, 0.0, 2.0],
                [1.0, 2.0, 4.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=np.float32,
        )
        pc = PointCloud3D(pts)
        self.assertEqual(len(pc), 3)

        bbox = pc.get_bounding_box()
        self.assertAlmostEqual(bbox.min_bound.x, -1.0)
        self.assertAlmostEqual(bbox.max_bound.x, 1.0)
        self.assertAlmostEqual(bbox.extents.x, 2.0)
        self.assertAlmostEqual(bbox.extents.y, 3.0)

    def test_voxel_downsampling(self) -> None:
        # Create a dense cluster of points inside a single 0.1m voxel
        pts = np.array(
            [
                [0.01, 0.01, 1.01],
                [0.02, 0.02, 1.02],
                [0.03, 0.03, 1.03],
                [2.0, 2.0, 5.0],  # Far point in separate voxel
            ],
            dtype=np.float32,
        )
        pc = PointCloud3D(pts)
        downsampled = pc.voxel_downsample(voxel_size_m=0.1)

        # 4 points should reduce to 2 voxels
        self.assertEqual(len(downsampled), 2)

        # First voxel centroid should be ~ (0.02, 0.02, 1.02)
        p_filtered = downsampled.points
        cluster_centroid = p_filtered[p_filtered[:, 2] < 2.0][0]
        self.assertAlmostEqual(cluster_centroid[0], 0.02, places=3)
        self.assertAlmostEqual(cluster_centroid[1], 0.02, places=3)

    def test_statistical_outlier_removal(self) -> None:
        # Create a dense cluster of 20 points and 1 extreme outlier
        np.random.seed(42)
        cluster = np.random.normal(loc=0.0, scale=0.05, size=(30, 3)).astype(
            np.float32
        )
        outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        all_pts = np.vstack([cluster, outlier])

        pc = PointCloud3D(all_pts)
        self.assertEqual(len(pc), 31)

        cleaned = pc.remove_statistical_outliers(k_neighbors=5, std_ratio=1.5)
        # Outlier should be filtered out
        self.assertLess(len(cleaned), 31)
        self.assertFalse(np.any(cleaned.points[:, 0] > 50.0))


if __name__ == "__main__":
    unittest.main()
