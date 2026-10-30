import unittest

import numpy as np

from core.domain.spatial.frustum import CameraIntrinsics
from core.domain.spatial.multicam import (
    CameraExtrinsics,
    CrossCameraAssociator,
    MultiCameraObservation,
    triangulate_dlt,
)
from core.domain.tracking.reid import AppearanceEmbedding


class MultiCamSpatialTests(unittest.TestCase):
    def test_camera_extrinsics_projection_matrix(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        # Identity rotation, camera at (0, 0, 0)
        rot = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        trans = (0.0, 0.0, 0.0)
        extrinsics = CameraExtrinsics(rot, trans)

        p_mat = extrinsics.projection_matrix(intrinsics)
        self.assertEqual(p_mat.shape, (3, 4))
        self.assertAlmostEqual(p_mat[0, 0], 1000.0)
        self.assertAlmostEqual(p_mat[0, 2], 640.0)

    def test_triangulate_dlt_stereo(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=0.0, cy=0.0)

        # Cam 1 at origin (0, 0, 0)
        ext1 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        )
        # Cam 2 translated by 1 meter to the right (x = 1.0)
        ext2 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (-1.0, 0.0, 0.0),
        )

        p1 = ext1.projection_matrix(intrinsics)
        p2 = ext2.projection_matrix(intrinsics)

        # Real 3D point at (0.0, 0.0, 5.0)
        # Cam1 sees it at (0, 0)
        # Cam2 sees it at (-1/5 * 1000, 0) = (-200, 0)
        obs1 = (0.0, 0.0)
        obs2 = (-200.0, 0.0)

        triangulated = triangulate_dlt([p1, p2], [obs1, obs2])
        self.assertAlmostEqual(triangulated.x, 0.0, places=2)
        self.assertAlmostEqual(triangulated.y, 0.0, places=2)
        self.assertAlmostEqual(triangulated.z, 5.0, places=2)

    def test_cross_camera_associator(self) -> None:
        associator = CrossCameraAssociator(
            epipolar_threshold_px=20.0, appearance_threshold=0.5
        )

        # Fundamental matrix for baseline translation along X
        # F = [e2]x * H = skew([1, 0, 0]) = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        f_mat = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

        emb_target = AppearanceEmbedding.from_raw([1.0, 0.5, 0.0])
        emb_other = AppearanceEmbedding.from_raw([0.0, 0.0, 1.0])

        cam1_obs = [
            MultiCameraObservation(
                camera_id="cam-1",
                track_id=1,
                pixel_center=(100.0, 200.0),
                appearance=emb_target,
            )
        ]
        cam2_obs = [
            MultiCameraObservation(
                camera_id="cam-2",
                track_id=10,
                pixel_center=(120.0, 202.0),
                appearance=emb_target,
            ),
            MultiCameraObservation(
                camera_id="cam-2",
                track_id=11,
                pixel_center=(500.0, 500.0),
                appearance=emb_other,
            ),
        ]

        matches = associator.associate_pair(f_mat, cam1_obs, cam2_obs)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].track_id, 1)
        self.assertEqual(matches[0][1].track_id, 10)
        self.assertLess(matches[0][2], 0.2)


if __name__ == "__main__":
    unittest.main()
