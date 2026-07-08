from dataclasses import dataclass

from .value_objects import BoundingBox, CameraId, Confidence, FrameSequence


@dataclass(frozen=True, slots=True)
class Detection:
    detection_id: str
    class_name: str
    confidence: Confidence
    bbox: BoundingBox
    track_id: int | None = None

    def __post_init__(self) -> None:
        if not self.detection_id or self.detection_id.strip() != self.detection_id:
            raise ValueError("detection id must be non-empty and have no surrounding whitespace")
        if not self.class_name or self.class_name.strip() != self.class_name:
            raise ValueError("class name must be non-empty and have no surrounding whitespace")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("track id must be non-negative")


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    camera_id: CameraId
    frame_sequence: FrameSequence
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        detection_ids = [detection.detection_id for detection in self.detections]
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("detection ids must be unique within a frame")