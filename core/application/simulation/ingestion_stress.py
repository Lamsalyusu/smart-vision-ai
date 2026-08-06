from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Callable

from core.application.ingestion import (
    CameraStream,
    CameraStreamRegistry,
    FrameAdmissionScheduler,
    FrameIngressCoordinator,
    IncomingFrame,
    VideoCodec,
)
from core.infrastructure.shared_memory import FramePixelFormat, SharedFrameRingBuffer


@dataclass(frozen=True, slots=True)
class IngestionStressReport:
    stream_count: int
    frames_per_stream: int
    submitted_frames: int
    accepted_frames: int
    dropped_frames: int
    elapsed_seconds: float
    throughput_fps: float
    p50_latency_ms: float
    p99_latency_ms: float


def run_ingestion_stress(
    *,
    stream_count: int = 100,
    frames_per_stream: int = 10,
    frame_width: int = 16,
    frame_height: int = 16,
    worker_count: int = 32,
    ring_capacity: int | None = None,
    admission_capacity_per_stream: int | None = None,
    before_submit: Callable[[int, int], None] | None = None,
) -> IngestionStressReport:
    """Measure concurrent in-process ingress submission, not network media."""
    if stream_count < 1 or frames_per_stream < 1 or worker_count < 1:
        raise ValueError("stream, frame, and worker counts must be positive")
    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    capacity = ring_capacity if ring_capacity is not None else stream_count * frames_per_stream
    if capacity < 1:
        raise ValueError("ring capacity must be positive")
    admission_capacity = (
        frames_per_stream
        if admission_capacity_per_stream is None
        else admission_capacity_per_stream
    )
    if admission_capacity < 1:
        raise ValueError("admission capacity per stream must be positive")

    registry = CameraStreamRegistry(maximum_streams=stream_count)
    for stream_index in range(stream_count):
        registry.register(
            CameraStream(
                f"camera-{stream_index}",
                VideoCodec.H264,
                frame_width,
                frame_height,
                1_000_000,
            )
        )

    frame_size = frame_width * frame_height * 3
    ring = SharedFrameRingBuffer(capacity, frame_size)
    coordinator = FrameIngressCoordinator(
        registry,
        FrameAdmissionScheduler(capacity=admission_capacity, refill_rate_per_second=0),
        ring,
    )
    payload = bytes(frame_size)
    latency_samples: list[float] = []
    accepted = 0
    dropped = 0
    result_lock = threading.Lock()

    def submit(camera_index: int, frame_index: int) -> tuple[bool, float]:
        if before_submit is not None:
            before_submit(camera_index, frame_index)
        frame = IncomingFrame(
            f"camera-{camera_index}",
            frame_index,
            frame_width,
            frame_height,
            FramePixelFormat.RGB24,
            (frame_index + 1) * 1_000_000,
            payload,
            frame_index == 0,
        )
        started = time.perf_counter()
        result = coordinator.submit(frame)
        return result.accepted, (time.perf_counter() - started) * 1000.0

    started_at = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(submit, camera_index, frame_index)
                for camera_index in range(stream_count)
                for frame_index in range(frames_per_stream)
            ]
            for future in as_completed(futures):
                was_accepted, latency_ms = future.result()
                with result_lock:
                    accepted += int(was_accepted)
                    dropped += int(not was_accepted)
                    latency_samples.append(latency_ms)
        elapsed = time.perf_counter() - started_at
        ordered_latencies = sorted(latency_samples)
        p99_index = max(0, math.ceil(0.99 * len(ordered_latencies)) - 1)
        submitted = stream_count * frames_per_stream
        return IngestionStressReport(
            stream_count=stream_count,
            frames_per_stream=frames_per_stream,
            submitted_frames=submitted,
            accepted_frames=accepted,
            dropped_frames=dropped,
            elapsed_seconds=elapsed,
            throughput_fps=submitted / elapsed if elapsed > 0 else 0.0,
            p50_latency_ms=statistics.median(ordered_latencies),
            p99_latency_ms=ordered_latencies[p99_index],
        )
    finally:
        ring.close()
        ring.unlink()