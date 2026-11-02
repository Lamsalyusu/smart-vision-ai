"""Synchronize per-camera observations before multi-camera fusion.

Live cameras never deliver frames on exactly the same clock tick.  This bridge
buffers the newest observation batch from each camera, emits a batch once the
cameras agree within a configurable tolerance, and flushes partial batches when
a camera stalls.  It is deliberately transport agnostic: RTSP, WebRTC, or a
replay source all feed the same ``CameraFrameBatch`` structure.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from core.domain.spatial.multicam import GlobalTrackMatch, MultiCameraObservation


class FrameFusion(Protocol):
    def fuse_synchronized_frame(
        self,
        frame_sequence: int,
        observations_by_camera: Mapping[str, Sequence[MultiCameraObservation]],
    ) -> tuple[GlobalTrackMatch, ...]: ...


@dataclass(frozen=True, slots=True)
class CameraFrameBatch:
    camera_id: str
    captured_at_seconds: float
    observations: tuple[MultiCameraObservation, ...]


@dataclass(frozen=True, slots=True)
class SynchronizationMetrics:
    frames_accepted: int
    frames_dropped_late: int
    batches_emitted: int
    partial_batches: int
    cameras_waiting: int


class SynchronizedFrameBridge:
    """Combine time-aligned camera observations into fusion calls."""

    def __init__(
        self,
        coordinator: FrameFusion,
        expected_cameras: Sequence[str],
        *,
        tolerance_seconds: float = 0.05,
        stall_timeout_seconds: float = 0.5,
        max_buffered_per_camera: int = 4,
    ) -> None:
        expected = tuple(expected_cameras)
        if not expected:
            raise ValueError("at least one expected camera is required")
        if len(set(expected)) != len(expected):
            raise ValueError("expected camera ids must be unique")
        if any(not camera_id for camera_id in expected):
            raise ValueError("camera ids must be non-empty")
        if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if not math.isfinite(stall_timeout_seconds) or stall_timeout_seconds <= 0:
            raise ValueError("stall timeout must be finite and positive")
        if max_buffered_per_camera < 1:
            raise ValueError("camera buffer must hold at least one frame")

        self._coordinator = coordinator
        self._expected = expected
        self._tolerance = tolerance_seconds
        self._stall_timeout = stall_timeout_seconds
        self._pending: dict[str, deque[CameraFrameBatch]] = {
            camera_id: deque(maxlen=max_buffered_per_camera) for camera_id in expected
        }
        self._last_seen: dict[str, float] = {}
        self._sequence = 0
        self._last_emitted_at = -math.inf
        self._lock = threading.Lock()
        self._frames_accepted = 0
        self._frames_dropped_late = 0
        self._batches_emitted = 0
        self._partial_batches = 0

    def submit(
        self, frame: CameraFrameBatch, *, now_seconds: float | None = None
    ) -> tuple[GlobalTrackMatch, ...]:
        self._validate_frame(frame)
        with self._lock:
            if frame.captured_at_seconds <= self._last_emitted_at:
                self._frames_dropped_late += 1
                return ()
            pending = self._pending[frame.camera_id]
            if len(pending) == pending.maxlen:
                self._frames_dropped_late += 1
            pending.append(frame)
            self._frames_accepted += 1
            self._last_seen[frame.camera_id] = frame.captured_at_seconds
            now = self._resolve_now(frame.captured_at_seconds, now_seconds)
            return self._emit(now, force=False)

    def flush_stalled(
        self, *, now_seconds: float | None = None
    ) -> tuple[GlobalTrackMatch, ...]:
        with self._lock:
            if not any(self._pending.values()):
                return ()
            now = self._resolve_now(None, now_seconds)
            return self._emit(now, force=True)

    def metrics(self) -> SynchronizationMetrics:
        with self._lock:
            waiting = sum(1 for frames in self._pending.values() if frames)
            return SynchronizationMetrics(
                frames_accepted=self._frames_accepted,
                frames_dropped_late=self._frames_dropped_late,
                batches_emitted=self._batches_emitted,
                partial_batches=self._partial_batches,
                cameras_waiting=waiting,
            )

    def _validate_frame(self, frame: CameraFrameBatch) -> None:
        if frame.camera_id not in self._pending:
            raise ValueError(f"camera {frame.camera_id!r} is not expected by this bridge")
        if not math.isfinite(frame.captured_at_seconds) or frame.captured_at_seconds < 0:
            raise ValueError("capture timestamp must be finite and non-negative")
        for observation in frame.observations:
            if observation.camera_id != frame.camera_id:
                raise ValueError("observation camera id must match the frame camera")

    def _resolve_now(self, fallback: float | None, explicit: float | None) -> float:
        if explicit is not None:
            if not math.isfinite(explicit) or explicit < 0:
                raise ValueError("now must be finite and non-negative")
            return explicit
        if fallback is not None:
            return fallback
        if self._last_seen:
            return max(self._last_seen.values())
        return 0.0

    def _emit(self, now: float, *, force: bool) -> tuple[GlobalTrackMatch, ...]:
        available = {
            camera_id: frames[-1]
            for camera_id, frames in self._pending.items()
            if frames
        }
        if not available:
            return ()

        newest = max(frame.captured_at_seconds for frame in available.values())
        oldest = min(frame.captured_at_seconds for frame in available.values())
        complete = len(available) == len(self._expected)
        synchronized = complete and newest - oldest <= self._tolerance

        if (
            not synchronized
            and not force
            and not self._has_stalled_camera(available, now)
            and now - oldest < self._stall_timeout
        ):
            return ()

        selected: dict[str, CameraFrameBatch] = {}
        for camera_id, frames in self._pending.items():
            if not frames:
                continue
            if synchronized:
                chosen = min(
                    frames,
                    key=lambda batch: (
                        abs(batch.captured_at_seconds - oldest),
                        batch.captured_at_seconds,
                    ),
                )
            else:
                chosen = frames[-1]
            while frames and frames[0] is not chosen:
                frames.popleft()
                self._frames_dropped_late += 1
            frames.popleft()
            selected[camera_id] = chosen

        if not selected:
            return ()

        self._sequence += 1
        self._last_emitted_at = max(
            batch.captured_at_seconds for batch in selected.values()
        )
        self._batches_emitted += 1
        if len(selected) != len(self._expected):
            self._partial_batches += 1

        return self._coordinator.fuse_synchronized_frame(
            self._sequence,
            {
                camera_id: batch.observations
                for camera_id, batch in selected.items()
            },
        )

    def _has_stalled_camera(
        self, available: Mapping[str, CameraFrameBatch], now: float
    ) -> bool:
        missing = [
            camera_id for camera_id in self._expected if camera_id not in available
        ]
        if not missing:
            return False
        oldest_available = min(
            frame.captured_at_seconds for frame in available.values()
        )
        if now - oldest_available >= self._stall_timeout:
            return True
        for camera_id in missing:
            seen = self._last_seen.get(camera_id)
            if seen is not None and now - seen >= self._stall_timeout:
                return True
        return False
