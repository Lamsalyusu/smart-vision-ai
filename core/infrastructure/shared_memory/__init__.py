"""Shared-memory adapters for high-throughput frame transfer."""

from .frame_ring_buffer import FrameDescriptor, FrameLease, FramePixelFormat, SharedFrameRingBuffer

__all__ = ["FrameDescriptor", "FrameLease", "FramePixelFormat", "SharedFrameRingBuffer"]