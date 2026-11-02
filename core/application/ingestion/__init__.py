"""Camera stream registration, admission control, and synchronization."""

from .frame_admission import AdmissionDecision, FrameAdmissionScheduler
from .stream_registry import (
    AdaptiveBitrateController,
    BitrateDecision,
    CameraStream,
    CameraStreamRegistry,
    VideoCodec,
)
from .synchronized_frames import (
    CameraFrameBatch,
    SynchronizationMetrics,
    SynchronizedFrameBridge,
)

__all__ = [
    "AdaptiveBitrateController",
    "AdmissionDecision",
    "BitrateDecision",
    "CameraFrameBatch",
    "CameraStream",
    "CameraStreamRegistry",
    "FrameAdmissionScheduler",
    "SynchronizationMetrics",
    "SynchronizedFrameBridge",
    "VideoCodec",
]
