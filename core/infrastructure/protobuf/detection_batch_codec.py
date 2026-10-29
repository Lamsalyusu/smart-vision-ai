from proto import detection_pb2

from core.domain.perception import (
    BoundingBox,
    CameraId,
    Confidence,
    Detection,
    DetectionBatch,
    FrameSequence,
)


class DetectionBatchCodec:
    """Translate between perception-domain batches and protobuf messages."""

    @staticmethod
    def encode(batch: DetectionBatch) -> detection_pb2.DetectionBatch:
        message = detection_pb2.DetectionBatch(
            camera_id=batch.camera_id.value,
            frame_sequence_number=batch.frame_sequence.value,
        )
        for detection in batch.detections:
            wire_detection = message.detections.add(
                class_name=detection.class_name,
                confidence=detection.confidence.value,
                track_id=detection.track_id,
            )
            wire_detection.bbox.x1 = detection.bbox.x1
            wire_detection.bbox.y1 = detection.bbox.y1
            wire_detection.bbox.x2 = detection.bbox.x2
            wire_detection.bbox.y2 = detection.bbox.y2
        return message

    @staticmethod
    def decode(message: detection_pb2.DetectionBatch) -> DetectionBatch:
        detections = []
        for index, wire_detection in enumerate(message.detections):
            if not wire_detection.HasField("bbox"):
                raise ValueError(f"detection at index {index} is missing its bounding box")
            track_id = wire_detection.track_id if wire_detection.HasField("track_id") else None
            detections.append(
                Detection(
                    detection_id=(
                        f"{message.camera_id}:{message.frame_sequence_number}:{index}"
                    ),
                    class_name=wire_detection.class_name,
                    confidence=Confidence(wire_detection.confidence),
                    bbox=BoundingBox(
                        wire_detection.bbox.x1,
                        wire_detection.bbox.y1,
                        wire_detection.bbox.x2,
                        wire_detection.bbox.y2,
                    ),
                    track_id=track_id,
                )
            )
        return DetectionBatch(
            camera_id=CameraId(message.camera_id),
            frame_sequence=FrameSequence(message.frame_sequence_number),
            detections=tuple(detections),
        )