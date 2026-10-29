from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class DriftReport:
    ks_statistics: tuple[float, ...]
    wasserstein_distances: tuple[float, ...]
    ks_critical_value: float
    drift_detected: bool
    sample_count: int


@dataclass(frozen=True, slots=True)
class AnnotationCandidate:
    frame_id: str
    confidence: float
    reason: str


class _EmbeddingDriftMonitor:
    def __init__(
        self,
        reference_embeddings: Sequence[Sequence[float]],
        *,
        window_size: int,
        minimum_samples: int,
        significance: float,
        wasserstein_threshold: float,
    ) -> None:
        reference = np.asarray(reference_embeddings, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] < 1:
            raise ValueError("reference embeddings must contain at least two vectors")
        if not np.isfinite(reference).all():
            raise ValueError("reference embeddings must be finite")
        if minimum_samples < 2 or window_size < minimum_samples:
            raise ValueError("window size must be at least the minimum sample count >= 2")
        if not math.isfinite(significance) or not 0 < significance < 1:
            raise ValueError("significance must be between zero and one")
        if not math.isfinite(wasserstein_threshold) or wasserstein_threshold < 0:
            raise ValueError("Wasserstein threshold must be finite and non-negative")

        self._reference = reference
        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._minimum_samples = minimum_samples
        self._significance = significance
        self._wasserstein_threshold = wasserstein_threshold

    def observe(self, embeddings: Sequence[Sequence[float]]) -> DriftReport:
        incoming = np.asarray(embeddings, dtype=np.float64)
        if incoming.ndim != 2 or incoming.shape[1] != self._reference.shape[1]:
            raise ValueError("incoming embedding dimensions must match reference embeddings")
        if not np.isfinite(incoming).all():
            raise ValueError("incoming embeddings must be finite")
        self._window.extend(row.copy() for row in incoming)
        current = np.asarray(self._window, dtype=np.float64)
        if current.shape[0] < self._minimum_samples:
            return DriftReport((), (), 0.0, False, current.shape[0])

        ks_values = []
        wasserstein_values = []
        for dimension in range(self._reference.shape[1]):
            reference = self._reference[:, dimension]
            observed = current[:, dimension]
            ks_values.append(self._ks_statistic(reference, observed))
            wasserstein_values.append(self._wasserstein_distance(reference, observed))

        sample_total = self._reference.shape[0] + current.shape[0]
        corrected_significance = self._significance / self._reference.shape[1]
        critical = math.sqrt(-0.5 * math.log(corrected_significance / 2.0)) * math.sqrt(
            sample_total / (self._reference.shape[0] * current.shape[0])
        )
        detected = any(value > critical for value in ks_values) or any(
            value > self._wasserstein_threshold for value in wasserstein_values
        )
        return DriftReport(
            tuple(ks_values),
            tuple(wasserstein_values),
            critical,
            detected,
            current.shape[0],
        )

    @staticmethod
    def _ks_statistic(reference: np.ndarray, observed: np.ndarray) -> float:
        pooled = np.sort(np.concatenate((reference, observed)))
        reference_cdf = np.searchsorted(np.sort(reference), pooled, side="right") / len(reference)
        observed_cdf = np.searchsorted(np.sort(observed), pooled, side="right") / len(observed)
        return float(np.max(np.abs(reference_cdf - observed_cdf)))

    @staticmethod
    def _wasserstein_distance(reference: np.ndarray, observed: np.ndarray) -> float:
        pooled = np.sort(np.concatenate((reference, observed)))
        if len(pooled) < 2:
            return 0.0
        reference_cdf = np.searchsorted(np.sort(reference), pooled[:-1], side="right") / len(reference)
        observed_cdf = np.searchsorted(np.sort(observed), pooled[:-1], side="right") / len(observed)
        intervals = np.diff(pooled)
        return float(np.sum(np.abs(reference_cdf - observed_cdf) * intervals))


class ActiveLearningDriftPipeline:
    """Detect feature-distribution drift and retain high-priority frame IDs."""

    def __init__(
        self,
        reference_embeddings: Sequence[Sequence[float]],
        *,
        queue_capacity: int = 100,
        confidence_threshold: float = 0.4,
        window_size: int = 256,
        minimum_samples: int = 16,
        significance: float = 0.05,
        wasserstein_threshold: float = 0.5,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue capacity must be positive")
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between zero and one")
        self._monitor = _EmbeddingDriftMonitor(
            reference_embeddings,
            window_size=window_size,
            minimum_samples=minimum_samples,
            significance=significance,
            wasserstein_threshold=wasserstein_threshold,
        )
        self._queue_capacity = queue_capacity
        self._confidence_threshold = confidence_threshold
        self._queue: dict[str, tuple[float, AnnotationCandidate]] = {}

    def observe(
        self,
        frame_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        confidences: Sequence[float],
    ) -> DriftReport:
        if len(frame_ids) != len(embeddings) or len(frame_ids) != len(confidences):
            raise ValueError("frame IDs, embeddings, and confidences must have equal lengths")
        if any(not frame_id for frame_id in frame_ids):
            raise ValueError("frame IDs must be non-empty")
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in confidences):
            raise ValueError("confidences must be finite values between zero and one")

        report = self._monitor.observe(embeddings)
        for frame_id, confidence in zip(frame_ids, confidences):
            drift_reason = report.drift_detected
            low_confidence = confidence <= self._confidence_threshold
            if drift_reason or low_confidence:
                reasons = []
                if drift_reason:
                    reasons.append("embedding_drift")
                if low_confidence:
                    reasons.append("low_confidence")
                candidate = AnnotationCandidate(frame_id, float(confidence), "+".join(reasons))
                self._enqueue(candidate, float(drift_reason) + 1.0 - float(confidence))
        return report

    def _enqueue(self, candidate: AnnotationCandidate, priority: float) -> None:
        existing = self._queue.get(candidate.frame_id)
        if existing is not None and existing[0] >= priority:
            return
        self._queue[candidate.frame_id] = (priority, candidate)
        if len(self._queue) > self._queue_capacity:
            lowest_id = min(self._queue, key=lambda frame_id: self._queue[frame_id][0])
            del self._queue[lowest_id]

    def pop_candidate(self) -> AnnotationCandidate | None:
        if not self._queue:
            return None
        highest_id = max(self._queue, key=lambda frame_id: self._queue[frame_id][0])
        return self._queue.pop(highest_id)[1]

    @property
    def queued_count(self) -> int:
        return len(self._queue)