from dataclasses import dataclass

from core.application.ingestion.frame_admission import FrameAdmissionScheduler
from core.application.ingestion.stream_registry import CameraStreamRegistry
from core.infrastructure.shared_memory.frame_ring_buffer import (
    FramePixelFormat,
    SharedFrameRingBuffer,
)


@dataclass(frozen=True, slots=True)
class IncomingFrame:
    camera_id: str
    sequence_number: int
    width: int
    height: int
    pixel_format: FramePixelFormat
    captured_at_ns: int
    data: bytes | bytearray | memoryview
    is_keyframe: bool = False


@dataclass(frozen=True, slots=True)
class FrameIngressResult:
    accepted: bool
    advance_tracking: bool
    reason: str
    tokens_remaining: float


class FrameIngressCoordinator:
    """Apply stream registration, flow control, then shared-ring handoff."""

    def __init__(
        self,
        registry: CameraStreamRegistry,
        scheduler: FrameAdmissionScheduler,
        ring: SharedFrameRingBuffer,
    ) -> None:
        self._registry = registry
        self._scheduler = scheduler
        self._ring = ring

    def submit(self, frame: IncomingFrame) -> FrameIngressResult:
        if self._registry.get(frame.camera_id) is None:
            raise KeyError(f"camera stream is not registered: {frame.camera_id}")
        decision = self._scheduler.admit(
            frame.camera_id,
            frame.is_keyframe,
            frame.captured_at_ns / 1_000_000_000,
        )
        if not decision.admitted:
            return FrameIngressResult(
                False, decision.advance_tracking, decision.reason, decision.tokens_remaining
            )

        written = self._ring.write_frame(
            camera_id=frame.camera_id,
            sequence_number=frame.sequence_number,
            width=frame.width,
            height=frame.height,
            pixel_format=frame.pixel_format,
            captured_at_ns=frame.captured_at_ns,
            data=frame.data,
        )
        if not written:
            return FrameIngressResult(
                False, True, "ring_full", decision.tokens_remaining
            )
        return FrameIngressResult(
            True, True, "enqueued", decision.tokens_remaining
        )