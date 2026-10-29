"""Application policies for camera frame ingestion."""

from .frame_admission import AdmissionDecision, FrameAdmissionScheduler

__all__ = ["AdmissionDecision", "FrameAdmissionScheduler"]