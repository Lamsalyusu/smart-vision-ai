import unittest

from core.application.perception import BuildDetectionBatch
from core.domain.perception import CameraId, FrameSequence


class BuildDetectionBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.use_case = BuildDetectionBatch()

    def test_maps_existing_detector_payload(self) -> None:
        batch = self.use_case.execute(
            "camera-1",
            12,
            [
                {
                    "class": "knife",
                    "confidence": 0.93,
                    "bbox": [10, 20, 110, 220],
                    "track_id": 4,
                }
            ],
        )

        self.assertEqual(batch.camera_id, CameraId("camera-1"))
        self.assertEqual(batch.frame_sequence, FrameSequence(12))
        self.assertEqual(batch.detections[0].class_name, "knife")
        self.assertEqual(batch.detections[0].bbox.x2, 110)
        self.assertEqual(batch.detections[0].track_id, 4)

    def test_generated_ids_are_stable_for_same_frame_input(self) -> None:
        payload = [{"class": "chair", "confidence": 0.8, "bbox": [1, 2, 3, 4]}]

        first = self.use_case.execute("camera-1", 12, payload)
        second = self.use_case.execute("camera-1", 12, payload)

        self.assertEqual(first.detections[0].detection_id, second.detections[0].detection_id)

    def test_rejects_malformed_detector_bbox(self) -> None:
        with self.assertRaisesRegex(ValueError, "four coordinates"):
            self.use_case.execute(
                "camera-1",
                12,
                [{"class": "chair", "confidence": 0.8, "bbox": [1, 2, 3]}],
            )

    def test_rejects_non_numeric_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence must be numeric"):
            self.use_case.execute(
                "camera-1",
                12,
                [{"class": "chair", "confidence": "high", "bbox": [1, 2, 3, 4]}],
            )

    def test_rejects_invalid_domain_values(self) -> None:
        with self.assertRaises(ValueError):
            self.use_case.execute(
                "camera-1",
                12,
                [{"class": "chair", "confidence": 1.5, "bbox": [1, 2, 3, 4]}],
            )


if __name__ == "__main__":
    unittest.main()