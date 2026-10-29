import unittest

from core.domain.perception import (
    BoundingBox,
    CameraId,
    Confidence,
    Detection,
    DetectionBatch,
    FrameSequence,
)


def build_detection(detection_id: str = "detection-1") -> Detection:
    return Detection(
        detection_id=detection_id,
        class_name="person",
        confidence=Confidence(0.95),
        bbox=BoundingBox(10, 20, 40, 80),
    )


class PerceptionValueObjectTests(unittest.TestCase):
    def test_bounding_box_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(10, 20, 10, 80)

    def test_bounding_box_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(0, 0, float("inf"), 1)

    def test_confidence_accepts_zero_and_one(self) -> None:
        self.assertEqual(Confidence(0).value, 0)
        self.assertEqual(Confidence(1).value, 1)

    def test_confidence_rejects_values_outside_unit_interval(self) -> None:
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Confidence(value)

    def test_camera_id_rejects_empty_or_padded_values(self) -> None:
        for value in ("", " camera-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CameraId(value)

    def test_frame_sequence_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            FrameSequence(-1)


class DetectionBatchTests(unittest.TestCase):
    def test_batch_accepts_unique_detections(self) -> None:
        batch = DetectionBatch(CameraId("camera-1"), FrameSequence(3), (build_detection(),))

        self.assertEqual(batch.camera_id.value, "camera-1")
        self.assertEqual(batch.frame_sequence.value, 3)

    def test_batch_rejects_duplicate_detection_ids(self) -> None:
        with self.assertRaises(ValueError):
            DetectionBatch(
                CameraId("camera-1"),
                FrameSequence(3),
                (build_detection(), build_detection()),
            )


if __name__ == "__main__":
    unittest.main()