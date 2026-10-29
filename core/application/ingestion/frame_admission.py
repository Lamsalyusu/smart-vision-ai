from dataclasses import dataclass
import math
import threading


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    advance_tracking: bool
    tokens_remaining: float
    reason: str


@dataclass(slots=True)
class _CameraBucket:
    tokens: float
    updated_at: float


class FrameAdmissionScheduler:
    """Apply independent token buckets while allowing keyframes through."""

    def __init__(self, capacity: float, refill_rate_per_second: float) -> None:
        if not math.isfinite(capacity) or capacity < 1:
            raise ValueError("capacity must be finite and at least one token")
        if not math.isfinite(refill_rate_per_second) or refill_rate_per_second < 0:
            raise ValueError("refill rate must be finite and non-negative")
        self._capacity = capacity
        self._refill_rate = refill_rate_per_second
        self._buckets: dict[str, _CameraBucket] = {}
        self._lock = threading.Lock()

    def admit(self, camera_id: str, is_keyframe: bool, now_seconds: float) -> AdmissionDecision:
        if not camera_id or camera_id.strip() != camera_id:
            raise ValueError("camera id must be non-empty and have no surrounding whitespace")
        if not math.isfinite(now_seconds) or now_seconds < 0:
            raise ValueError("clock time must be finite and non-negative")

        with self._lock:
            bucket = self._buckets.get(camera_id)
            if bucket is None:
                bucket = _CameraBucket(self._capacity, now_seconds)
                self._buckets[camera_id] = bucket
            else:
                elapsed = max(0.0, now_seconds - bucket.updated_at)
                bucket.tokens = min(
                    self._capacity,
                    bucket.tokens + elapsed * self._refill_rate,
                )
                bucket.updated_at = max(bucket.updated_at, now_seconds)

            if is_keyframe:
                return AdmissionDecision(True, True, bucket.tokens, "keyframe")
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return AdmissionDecision(True, True, bucket.tokens, "token_available")
            return AdmissionDecision(False, True, bucket.tokens, "rate_limited")

    def forget_idle_cameras(self, now_seconds: float, idle_timeout_seconds: float) -> int:
        if not math.isfinite(now_seconds) or now_seconds < 0:
            raise ValueError("clock time must be finite and non-negative")
        if not math.isfinite(idle_timeout_seconds) or idle_timeout_seconds < 0:
            raise ValueError("idle timeout must be finite and non-negative")

        cutoff = now_seconds - idle_timeout_seconds
        with self._lock:
            stale_ids = [
                camera_id
                for camera_id, bucket in self._buckets.items()
                if bucket.updated_at < cutoff
            ]
            for camera_id in stale_ids:
                del self._buckets[camera_id]
            return len(stale_ids)