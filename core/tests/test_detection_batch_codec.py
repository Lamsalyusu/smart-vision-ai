import unittest

from core.domain.perception import (
    BoundingBox,
    CameraId,
    Confidence,
    Detection,
    DetectionBatch,
    FrameSequence,
)
from core.infrastructure.protobuf import DetectionBatchCodec
from proto import detection_pb2


class DetectionBatchCodecTests(unittest.TestCase):
    def test_wire_round_trip_preserves_domain_values(self) -> None:
        batch = DetectionBatch(
            camera_id=CameraId("camera-1"),
            frame_sequence=FrameSequence(23),
            detections=(
                Detection(
                    detection_id="detection-1",
                    class_name="person",
                    confidence=Confidence(0.9),
                    bbox=BoundingBox(10, 20, 50, 80),
                    track_id=6,
                ),
            ),
        )

        encoded = DetectionBatchCodec.encode(batch)
        decoded = DetectionBatchCodec.decode(
            detection_pb2.DetectionBatch.FromString(encoded.SerializeToString())
        )

        self.assertEqual(decoded.camera_id, batch.camera_id)
        self.assertEqual(decoded.frame_sequence, batch.frame_sequence)
        self.assertEqual(decoded.detections[0].class_name, "person")
        self.assertAlmostEqual(decoded.detections[0].confidence.value, 0.9, places=6)
        self.assertEqual(decoded.detections[0].bbox, batch.detections[0].bbox)
        self.assertEqual(decoded.detections[0].track_id, 6)

    def test_decode_rejects_missing_bounding_box(self) -> None:
        message = detection_pb2.DetectionBatch(camera_id="camera-1")
        message.detections.add(class_name="person", confidence=0.9)

        with self.assertRaisesRegex(ValueError, "missing its bounding box"):
            DetectionBatchCodec.decode(message)

    def test_decode_rejects_invalid_confidence(self) -> None:
        message = detection_pb2.DetectionBatch(camera_id="camera-1")
        detection = message.detections.add(class_name="person", confidence=1.5)
        detection.bbox.x1 = 0
        detection.bbox.y1 = 0
        detection.bbox.x2 = 10
        detection.bbox.y2 = 10

        with self.assertRaisesRegex(ValueError, "confidence"):
            DetectionBatchCodec.decode(message)


if __name__ == "__main__":
    unittest.main()