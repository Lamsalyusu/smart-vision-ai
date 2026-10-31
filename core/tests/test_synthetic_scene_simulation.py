import unittest

from core.application.simulation.synthetic_scene import (
    GroundTruthTarget3D,
    Synthetic3DSceneSimulator,
)
from core.domain.spatial.frustum import CameraIntrinsics
from core.domain.spatial.multicam import CameraExtrinsics
from core.domain.spatial.transforms import Vector3


class SyntheticSceneSimulationTests(unittest.TestCase):
    def test_multi_camera_synthetic_scene_generation(self) -> None:
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
        ext1 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, 0.0, 0.0),
        )
        ext2 = CameraExtrinsics(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (-2.0, 0.0, 0.0),
        )

        cameras = {
            "cam-left": (intrinsics, ext1),
            "cam-right": (intrinsics, ext2),
        }
        sim = Synthetic3DSceneSimulator(cameras, seed=123)

        # Target walking along +Z at 1.0 m/s
        sim.add_target(
            GroundTruthTarget3D(
                track_id=1,
                class_name="person",
                start_pos=Vector3(0.0, 0.0, 5.0),
                velocity=Vector3(0.0, 0.0, 1.0),
            )
        )

        frames = sim.generate_scenario(
            duration_seconds=1.0, fps=10.0, miss_probability=0.0
        )
        self.assertEqual(len(frames), 10)

        # Verify frame 0
        f0 = frames[0]
        self.assertEqual(len(f0.ground_truth_targets), 1)
        self.assertEqual(f0.ground_truth_targets[0][0], 1)
        self.assertAlmostEqual(f0.ground_truth_targets[0][1].z, 5.0)

        # Both cameras should observe the target
        self.assertGreaterEqual(len(f0.camera_detections["cam-left"]), 1)
        self.assertGreaterEqual(len(f0.camera_detections["cam-right"]), 1)

        det_left = f0.camera_detections["cam-left"][0]
        self.assertEqual(det_left.class_name, "person")
        self.assertAlmostEqual(det_left.noisy_depth_m, 5.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
