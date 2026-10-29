"""Application policies for camera frame ingestion."""

from .frame_admission import AdmissionDecision, FrameAdmissionScheduler
from .stream_registry import (
	AdaptiveBitrateController,
	BitrateDecision,
	CameraStream,
	CameraStreamRegistry,
	VideoCodec,
)
from .frame_ingress import FrameIngressCoordinator, FrameIngressResult, IncomingFrame

__all__ = [
	"AdmissionDecision",
	"FrameAdmissionScheduler",
	"AdaptiveBitrateController",
	"BitrateDecision",
	"CameraStream",
	"CameraStreamRegistry",
	"VideoCodec",
	"FrameIngressCoordinator",
	"FrameIngressResult",
	"IncomingFrame",
]