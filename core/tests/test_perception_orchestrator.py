import unittest

from core.application.perception.pipeline_orchestrator import (
    PerceptionPipelineOrchestrator,
)
from core.domain.spatial.frustum import CameraIntrinsics
from core.infrastructure.resilience.fallback_engine import (
    BackendTier,
    ResilientInferencePipeline,
)


class PerceptionOrchestratorTests(unittest.TestCase):
    def test_pipeline_orchestrates_frame_through_tracks_and_scene_graph(self) -> None:
        def mock_detector(_: object) -> list[dict[str, object]]:
            return [
                {
                    "class": "person",
                    "confidence": 0.92,
                    "bbox": (100.0, 100.0, 300.0, 500.0),
                    "track_id": 1,
                },
                {
                    "class": "can",
                    "confidence": 0.85,
                    "bbox": (350.0, 300.0, 420.0, 450.0),
                    "track_id": 2,
                },
            ]

        inference_pipeline = ResilientInferencePipeline(
            primary_fn=mock_detector,
            secondary_fn=mock_detector,
            fallback_fn=lambda _: [],
        )

        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0)
        orchestrator = PerceptionPipelineOrchestrator(
            camera_id="cam-front-door",
            inference_pipeline=inference_pipeline,
            intrinsics=intrinsics,
        )

        simulated_embeddings = [
            [1.0, 0.0, 0.5, 0.2],
            [0.0, 1.0, -0.2, 0.4],
        ]

        # Frame 1
        snapshot1 = orchestrator.process_frame(
            frame_sequence=1,
            frame_data=b"mock_frame_1",
            timestamp_seconds=1.0,
            simulated_embeddings=simulated_embeddings,
            estimated_depth_m=3.0,
        )

        self.assertEqual(snapshot1.camera_id, "cam-front-door")
        self.assertEqual(snapshot1.frame_sequence, 1)
        self.assertEqual(snapshot1.backend_tier, BackendTier.PRIMARY_GPU)
        self.assertFalse(snapshot1.fallback_occurred)
        self.assertEqual(snapshot1.active_tracks_count, 2)
        self.assertEqual(len(snapshot1.detection_batch.detections), 2)
        self.assertGreater(snapshot1.latency_ms, 0.0)

        # Frame 2 (moving closer)
        snapshot2 = orchestrator.process_frame(
            frame_sequence=2,
            frame_data=b"mock_frame_2",
            timestamp_seconds=1.033,
            simulated_embeddings=simulated_embeddings,
            estimated_depth_m=2.8,
        )

        self.assertEqual(snapshot2.frame_sequence, 2)
        self.assertEqual(snapshot2.active_tracks_count, 2)
        self.assertEqual(len(snapshot2.scene_snapshot.nodes), 2)


if __name__ == "__main__":
    unittest.main()
