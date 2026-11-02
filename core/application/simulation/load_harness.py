"""Streaming load and chaos harness for camera-to-inference pipelines.

The harness drives many concurrent synthetic camera streams through a frame
processor, injects deterministic faults, and reports throughput plus latency
percentiles.  It is used by unit/regression tests and by the operational
``scripts/load_test_synthetic.py`` command.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera_farm import SyntheticCameraFarm, SyntheticCameraFrame

FrameProcessor = Callable[[SyntheticCameraFrame], Any]


@dataclass(frozen=True, slots=True)
class CameraStreamPlan:
    camera_id: str
    frames: int
    target_fps: float = 30.0

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera id is required")
        if self.frames < 0:
            raise ValueError("frame count must be non-negative")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target fps must be finite and positive")


@dataclass(frozen=True, slots=True)
class ChaosPolicy:
    drop_every: int = 0
    disconnect_every: int = 0
    error_every: int = 0
    latency_spike_every: int = 0
    latency_spike_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name, period in (
            ("drop_every", self.drop_every),
            ("disconnect_every", self.disconnect_every),
            ("error_every", self.error_every),
            ("latency_spike_every", self.latency_spike_every),
        ):
            if period < 0:
                raise ValueError(f"{name} must be non-negative")
        if not math.isfinite(self.latency_spike_seconds) or self.latency_spike_seconds < 0:
            raise ValueError("latency spike must be finite and non-negative")

    def action_for(self, frame_index: int) -> str:
        for period, action in (
            (self.error_every, "error"),
            (self.disconnect_every, "disconnect"),
            (self.drop_every, "drop"),
            (self.latency_spike_every, "spike"),
        ):
            if period > 0 and (frame_index + 1) % period == 0:
                return action
        return "process"


@dataclass(frozen=True, slots=True)
class LoadReport:
    cameras: int
    frames_attempted: int
    frames_processed: int
    frames_dropped: int
    disconnects: int
    processing_errors: int
    duration_seconds: float
    throughput_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cameras": self.cameras,
            "frames_attempted": self.frames_attempted,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "disconnects": self.disconnects,
            "processing_errors": self.processing_errors,
            "duration_seconds": round(self.duration_seconds, 6),
            "throughput_fps": round(self.throughput_fps, 3),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "latency_p95_ms": round(self.latency_p95_ms, 3),
            "latency_p99_ms": round(self.latency_p99_ms, 3),
            "latency_max_ms": round(self.latency_max_ms, 3),
        }


@dataclass(slots=True)
class _StreamResult:
    attempted: int = 0
    processed: int = 0
    dropped: int = 0
    disconnects: int = 0
    errors: int = 0


class StreamingLoadHarness:
    """Run concurrent synthetic camera streams with fault injection."""

    def __init__(
        self,
        process_frame: FrameProcessor,
        *,
        farm: SyntheticCameraFarm | None = None,
        realtime: bool = False,
        max_workers: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._process_frame = process_frame
        self._farm = farm
        self._realtime = realtime
        self._max_workers = max_workers

    def run(
        self,
        plans: Sequence[CameraStreamPlan],
        *,
        chaos: ChaosPolicy | None = None,
    ) -> LoadReport:
        if not plans:
            raise ValueError("at least one stream plan is required")
        camera_ids = [plan.camera_id for plan in plans]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("stream camera ids must be unique")
        policy = chaos or ChaosPolicy()
        farm = self._farm or SyntheticCameraFarm(camera_ids)
        missing = set(camera_ids) - set(farm.camera_ids)
        if missing:
            raise ValueError(f"synthetic farm is missing cameras: {sorted(missing)}")

        latencies: list[float] = []
        results: list[_StreamResult] = []
        lock = threading.Lock()
        workers = self._max_workers or max(1, min(len(plans), 64))
        started = time.perf_counter()

        def run_stream(plan: CameraStreamPlan) -> _StreamResult:
            result = _StreamResult()
            local_latencies: list[float] = []
            interval = 1.0 / plan.target_fps
            next_deadline = time.perf_counter()
            for sequence in range(plan.frames):
                result.attempted += 1
                action = policy.action_for(sequence)
                if action == "drop":
                    result.dropped += 1
                    continue
                if action == "disconnect":
                    result.disconnects += 1
                    continue
                if action == "error":
                    result.errors += 1
                    continue
                frame = farm.frame(plan.camera_id, sequence, sequence * interval)
                if action == "spike" and policy.latency_spike_seconds:
                    time.sleep(policy.latency_spike_seconds)
                frame_started = time.perf_counter()
                try:
                    self._process_frame(frame)
                    result.processed += 1
                except Exception:  # noqa: BLE001 - chaos harness counts pipeline failures
                    result.errors += 1
                else:
                    local_latencies.append((time.perf_counter() - frame_started) * 1000.0)
                if self._realtime:
                    next_deadline += interval
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        next_deadline = time.perf_counter()
            with lock:
                latencies.extend(local_latencies)
            return result

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="camera-load") as pool:
            results = list(pool.map(run_stream, plans))

        duration = max(time.perf_counter() - started, 1e-9)
        attempted = sum(result.attempted for result in results)
        processed = sum(result.processed for result in results)
        ordered = sorted(latencies)

        return LoadReport(
            cameras=len(plans),
            frames_attempted=attempted,
            frames_processed=processed,
            frames_dropped=sum(result.dropped for result in results),
            disconnects=sum(result.disconnects for result in results),
            processing_errors=sum(result.errors for result in results),
            duration_seconds=duration,
            throughput_fps=processed / duration,
            latency_p50_ms=_percentile(ordered, 50),
            latency_p95_ms=_percentile(ordered, 95),
            latency_p99_ms=_percentile(ordered, 99),
            latency_max_ms=float(ordered[-1]) if ordered else 0.0,
        )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    return float(np.percentile(np.asarray(sorted_values), percentile))
