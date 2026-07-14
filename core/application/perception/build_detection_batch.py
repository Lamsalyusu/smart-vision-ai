from collections.abc import Mapping, Sequence
from numbers import Real

from core.domain.perception import (
    BoundingBox,
    CameraId,
    Confidence,
    Detection,
    DetectionBatch,
    FrameSequence,
)


class BuildDetectionBatch:
    """Translate detector dictionaries into the validated perception domain."""

    def execute(
        self,
        camera_id: str,
        frame_sequence: int,
        detections: Sequence[Mapping[str, object]],
    ) -> DetectionBatch:
        camera = CameraId(camera_id)
        frame = FrameSequence(frame_sequence)
        domain_detections = tuple(
            self._build_detection(camera, frame, index, item)
            for index, item in enumerate(detections)
        )
        return DetectionBatch(camera, frame, domain_detections)

    @staticmethod
    def _build_detection(
        camera_id: CameraId,
        frame_sequence: FrameSequence,
        index: int,
        payload: Mapping[str, object],
    ) -> Detection:
        class_name = payload.get("class")
        confidence = payload.get("confidence")
        bbox = payload.get("bbox")
        track_id = payload.get("track_id")

        if not isinstance(class_name, str):
            raise ValueError("detection class must be a string")
        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("detection confidence must be numeric")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise ValueError("detection bbox must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in bbox):
            raise ValueError("detection bbox coordinates must be numeric")
        if track_id is not None and (
            isinstance(track_id, bool) or not isinstance(track_id, int)
        ):
            raise ValueError("detection track id must be an integer")

        detection_id = f"{camera_id.value}:{frame_sequence.value}:{index}"
        return Detection(
            detection_id=detection_id,
            class_name=class_name,
            confidence=Confidence(float(confidence)),
            bbox=BoundingBox(*(float(value) for value in bbox)),
            track_id=track_id,
        )