from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import copy


class AnomalyType(Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"
    TREND = "trend"
    LEVEL_SHIFT = "level_shift"


class DetectionMethod(Enum):
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    timestamp: float
    value: float
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Anomaly:
    start_idx: int
    end_idx: int
    anomaly_type: AnomalyType
    score: float
    severity: float
    description: str
    affected_features: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    anomalies: List[Anomaly]
    scores: np.ndarray
    threshold: float
    method: DetectionMethod
    processing_time: float


class StatisticalDetector(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray):
        pass

    @abstractmethod
    def detect(self, data: np.ndarray) -> DetectionResult:
        pass


class ZScoreDetector(StatisticalDetector):
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-6

    def detect(self, data: np.ndarray) -> DetectionResult:
        scores = np.abs(data - self.mean) / self.std
        anomalies = []

        for i, score in enumerate(scores):
            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Z-score anomaly: {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class ModifiedZScoreDetector(StatisticalDetector):
    def __init__(self, window_size: int = 100, threshold: float = 3.5):
        self.window_size = window_size
        self.threshold = threshold
        self.median = 0.0
        self.mad = 1.0

    def fit(self, data: np.ndarray):
        self.median = np.median(data)
        self.mad = np.median(np.abs(data - self.median)) + 1e-6

    def detect(self, data: np.ndarray) -> DetectionResult:
        scores = 0.6745 * np.abs(data - self.median) / self.mad
        anomalies = []

        for i, score in enumerate(scores):
            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Modified Z-score anomaly: {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class IQRDetector(StatisticalDetector):
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.q1 = 0.0
        self.q3 = 0.0
        self.iqr = 1.0

    def fit(self, data: np.ndarray):
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1 + 1e-6

    def detect(self, data: np.ndarray) -> DetectionResult:
        lower = self.q1 - self.multiplier * self.iqr
        upper = self.q3 + self.multiplier * self.iqr

        scores = np.zeros_like(data)
        scores[data < lower] = (lower - data[data < lower]) / self.iqr
        scores[data > upper] = (data[data > upper] - upper) / self.iqr

        anomalies = []
        for i, score in enumerate(scores):
            if score > 0:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score, 5.0),
                    description=f"IQR anomaly: {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.multiplier,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class MovingAverageDetector(StatisticalDetector):
    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold

    def fit(self, data: np.ndarray):
        pass

    def detect(self, data: np.ndarray) -> DetectionResult:
        scores = np.zeros_like(data)
        anomalies = []

        for i in range(self.window_size, len(data)):
            window = data[i - self.window_size:i]
            mean = np.mean(window)
            std = np.std(window) + 1e-6
            score = abs(data[i] - mean) / std
            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Moving average anomaly: {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class EWMA(StatisticalDetector):
    def __init__(self, alpha: float = 0.3, threshold: float = 3.0):
        self.alpha = alpha
        self.threshold = threshold
        self.ewma = 0.0
        self.ewma_var = 1.0

    def fit(self, data: np.ndarray):
        self.ewma = data[0]
        self.ewma_var = np.var(data)

    def detect(self, data: np.ndarray) -> DetectionResult:
        scores = np.zeros_like(data)
        anomalies = []

        for i, val in enumerate(data):
            if i == 0:
                self.ewma = val
                continue

            prev_ewma = self.ewma
            self.ewma = self.alpha * val + (1 - self.alpha) * self.ewma
            self.ewma_var = self.alpha * (val - prev_ewma)**2 + (1 - self.alpha) * self.ewma_var
            std = math.sqrt(self.ewma_var) + 1e-6
            score = abs(val - self.ewma) / std
            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"EWMA anomaly: {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class SeasonalDecomposition:
    def __init__(self, period: int = 24):
        self.period = period
        self.trend = None
        self.seasonal = None
        self.residual = None

    def decompose(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(data)
        trend = np.zeros(n)
        seasonal = np.zeros(n)

        for i in range(self.period, n - self.period):
            window = data[i - self.period:i + self.period + 1]
            trend[i] = np.mean(window)

        detrended = data - trend
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])

        seasonal = seasonal - np.mean(seasonal)
        residual = detrended - seasonal

        self.trend = trend
        self.seasonal = seasonal
        self.residual = residual

        return trend, seasonal, residual


class ChangePointDetector:
    def __init__(self, min_distance: int = 10, penalty: float = 10.0):
        self.min_distance = min_distance
        self.penalty = penalty

    def detect(self, data: np.ndarray) -> List[int]:
        n = len(data)
        if n < 2 * self.min_distance:
            return []

        costs = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                segment = data[i:j+1]
                costs[i, j] = np.sum((segment - np.mean(segment))**2)

        dp = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        dp[0] = 0

        for i in range(1, n):
            for j in range(max(0, i - self.min_distance), i):
                cost = dp[j] + costs[j, i] + self.penalty
                if cost < dp[i]:
                    dp[i] = cost
                    prev[i] = j

        changes = []
        idx = n - 1
        while prev[idx] != -1:
            changes.append(idx)
            idx = prev[idx]
        changes.reverse()

        return changes


class AnomalyDetector:
    def __init__(
        self,
        detectors: List[StatisticalDetector] = None,
        ensemble_method: str = "voting"
    ):
        self.detectors = detectors or [
            ZScoreDetector(),
            ModifiedZScoreDetector(),
            IQRDetector(),
            MovingAverageDetector(),
            EWMA()
        ]
        self.ensemble_method = ensemble_method

    def fit(self, data: np.ndarray):
        for detector in self.detectors:
            detector.fit(data)

    def detect(self, data: np.ndarray) -> DetectionResult:
        results = [detector.detect(data) for detector in self.detectors]

        if self.ensemble_method == "voting":
            return self._voting_ensemble(results, data)
        elif self.ensemble_method == "max":
            return self._max_ensemble(results, data)
        elif self.ensemble_method == "average":
            return self._average_ensemble(results, data)
        return results[0]

    def _voting_ensemble(self, results: List[DetectionResult], data: np.ndarray) -> DetectionResult:
        n = len(data)
        votes = np.zeros(n)
        all_scores = np.zeros((len(results), n))

        for i, res in enumerate(results):
            all_scores[i] = res.scores
            for anomaly in res.anomalies:
                votes[anomaly.start_idx:anomaly.end_idx + 1] += 1

        threshold = len(self.detectors) / 2
        anomalies = []
        for i in range(n):
            if votes[i] >= threshold:
                score = np.mean(all_scores[:, i])
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=float(votes[i] / len(self.detectors) * 5),
                    description=f"Ensemble anomaly (votes: {votes[i]})"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=np.mean(all_scores, axis=0),
            threshold=threshold,
            method=DetectionMethod.ENSEMBLE,
            processing_time=0.0
        )

    def _max_ensemble(self, results: List[DetectionResult], data: np.ndarray) -> DetectionResult:
        all_scores = np.max([res.scores for res in results], axis=0)
        anomalies = []
        for i, score in enumerate(all_scores):
            if score > np.max([res.threshold for res in results]):
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / np.max([res.threshold for res in results]), 5.0),
                    description="Max ensemble anomaly"
                ))
        return DetectionResult(
            anomalies=anomalies,
            scores=all_scores,
            threshold=np.max([res.threshold for res in results]),
            method=DetectionMethod.ENSEMBLE,
            processing_time=0.0
        )

    def _average_ensemble(self, results: List[DetectionResult], data: np.ndarray) -> DetectionResult:
        all_scores = np.mean([res.scores for res in results], axis=0)
        avg_threshold = np.mean([res.threshold for res in results])
        anomalies = []
        for i, score in enumerate(all_scores):
            if score > avg_threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / avg_threshold, 5.0),
                    description="Average ensemble anomaly"
                ))
        return DetectionResult(
            anomalies=anomalies,
            scores=all_scores,
            threshold=avg_threshold,
            method=DetectionMethod.ENSEMBLE,
            processing_time=0.0
        )


class ContextualAnomalyDetector:
    def __init__(
        self,
        context_window: int = 50,
        threshold: float = 3.0,
        min_context_points: int = 10
    ):
        self.context_window = context_window
        self.threshold = threshold
        self.min_context_points = min_context_points

    def detect(self, data: np.ndarray, context_features: np.ndarray) -> DetectionResult:
        n = len(data)
        scores = np.zeros(n)
        anomalies = []

        for i in range(n):
            start = max(0, i - self.context_window)
            end = min(n, i + self.context_window + 1)

            context_data = data[start:end]
            context_feats = context_features[start:end]

            if len(context_data) < self.min_context_points:
                continue

            mean = np.mean(context_data)
            std = np.std(context_data) + 1e-6
            score = abs(data[i] - mean) / std
            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.CONTEXTUAL,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Contextual anomaly: {score:.2f}",
                    affected_features=[f"feat_{j}" for j in range(context_features.shape[1])]
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class CollectiveAnomalyDetector:
    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 2.0,
        min_anomaly_length: int = 3
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_anomaly_length = min_anomaly_length

    def detect(self, data: np.ndarray) -> DetectionResult:
        n = len(data)
        scores = np.zeros(n)
        anomalies = []

        for i in range(n - self.window_size + 1):
            window = data[i:i + self.window_size]
            window_mean = np.mean(window)
            window_std = np.std(window) + 1e-6

            global_mean = np.mean(data)
            global_std = np.std(data) + 1e-6

            score = abs(window_mean - global_mean) / global_std
            scores[i:i + self.window_size] = np.maximum(scores[i:i + self.window_size], score)

        in_anomaly = False
        start_idx = 0

        for i in range(n):
            if scores[i] > self.threshold and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif scores[i] <= self.threshold and in_anomaly:
                if i - start_idx >= self.min_anomaly_length:
                    anomalies.append(Anomaly(
                        start_idx=start_idx,
                        end_idx=i - 1,
                        anomaly_type=AnomalyType.COLLECTIVE,
                        score=float(np.max(scores[start_idx:i])),
                        severity=min(np.max(scores[start_idx:i]) / self.threshold, 5.0),
                        description=f"Collective anomaly: length {i - start_idx}"
                    ))
                in_anomaly = False

        if in_anomaly and n - start_idx >= self.min_anomaly_length:
            anomalies.append(Anomaly(
                start_idx=start_idx,
                end_idx=n - 1,
                anomaly_type=AnomalyType.COLLECTIVE,
                score=float(np.max(scores[start_idx:])),
                severity=min(np.max(scores[start_idx:]) / self.threshold, 5.0),
                description=f"Collective anomaly: length {n - start_idx}"
            ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class TrendAnomalyDetector:
    def __init__(self, window_size: int = 50, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold

    def detect(self, data: np.ndarray) -> DetectionResult:
        n = len(data)
        scores = np.zeros(n)
        anomalies = []

        for i in range(self.window_size, n):
            window = data[i - self.window_size:i]
            x = np.arange(self.window_size)
            coeffs = np.polyfit(x, window, 1)
            slope = coeffs[0]

            global_x = np.arange(n)
            global_coeffs = np.polyfit(global_x, data, 1)
            global_slope = global_coeffs[0]

            score = abs(slope - global_slope) / (abs(global_slope) + 1e-6)
            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.TREND,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Trend anomaly: slope change {slope:.4f} vs {global_slope:.4f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class LevelShiftDetector:
    def __init__(self, window_size: int = 50, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold

    def detect(self, data: np.ndarray) -> DetectionResult:
        n = len(data)
        scores = np.zeros(n)
        anomalies = []

        for i in range(self.window_size, n - self.window_size):
            before = data[i - self.window_size:i]
            after = data[i:i + self.window_size]

            mean_before = np.mean(before)
            mean_after = np.mean(after)
            pooled_std = math.sqrt((np.var(before) + np.var(after)) / 2) + 1e-6

            score = abs(mean_after - mean_before) / pooled_std
            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.LEVEL_SHIFT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Level shift: {mean_before:.4f} -> {mean_after:.4f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.STATISTICAL,
            processing_time=0.0
        )


class MultivariateAnomalyDetector:
    def __init__(
        self,
        method: str = "mahalanobis",
        threshold: float = None
    ):
        self.method = method
        self.threshold = threshold
        self.mean = None
        self.cov = None
        self.inv_cov = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data.T) + 1e-6 * np.eye(data.shape[1])
        self.inv_cov = np.linalg.inv(self.cov)

        if self.threshold is None:
            from scipy.stats import chi2
            self.threshold = chi2.ppf(0.999, data.shape[1])

    def detect(self, data: np.ndarray) -> DetectionResult:
        if self.mean is None:
            self.fit(data)

        n, d = data.shape
        scores = np.zeros(n)
        anomalies = []

        for i in range(n):
            diff = data[i] - self.mean
            if self.method == "mahalanobis":
                score = diff @ self.inv_cov @ diff
            elif self.method == "euclidean":
                score = np.sum(diff**2)
            else:
                score = diff @ self.inv_cov @ diff

            scores[i] = score

            if score > self.threshold:
                anomalies.append(Anomaly(
                    start_idx=i,
                    end_idx=i,
                    anomaly_type=AnomalyType.POINT,
                    score=float(score),
                    severity=min(score / self.threshold, 5.0),
                    description=f"Multivariate anomaly ({self.method}): {score:.2f}"
                ))

        return DetectionResult(
            anomalies=anomalies,
            scores=scores,
            threshold=self.threshold,
            method=DetectionMethod.ML_BASED,
            processing_time=0.0
        )


class AnomalyDetectionPipeline:
    def __init__(
        self,
        detectors: Dict[str, Any] = None
    ):
        self.detectors = detectors or {}
        self.history: List[DetectionResult] = []

    def add_detector(self, name: str, detector: Any):
        self.detectors[name] = detector

    def fit(self, data: np.ndarray, **kwargs):
        for name, detector in self.detectors.items():
            if hasattr(detector, 'fit'):
                if 'context_features' in kwargs and name == 'contextual':
                    detector.fit(data, kwargs['context_features'])
                else:
                    detector.fit(data)

    def detect(self, data: np.ndarray, **kwargs) -> Dict[str, DetectionResult]:
        results = {}
        for name, detector in self.detectors.items():
            if hasattr(detector, 'detect'):
                if 'context_features' in kwargs and name == 'contextual':
                    results[name] = detector.detect(data, kwargs['context_features'])
                else:
                    results[name] = detector.detect(data)
        self.history.append(list(results.values())[0] if results else None)
        return results


class AnomalyMetrics:
    @staticmethod
    def precision_recall_f1(
        predicted: List[Anomaly],
        ground_truth: List[Anomaly],
        tolerance: int = 5
    ) -> Tuple[float, float, float]:
        if not ground_truth:
            return 1.0 if not predicted else 0.0, 1.0 if not predicted else 0.0, 1.0 if not predicted else 0.0

        matched_gt = set()
        matched_pred = set()

        for i, pred in enumerate(predicted):
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                if abs(pred.start_idx - gt.start_idx) <= tolerance or \
                   (pred.start_idx <= gt.end_idx + tolerance and pred.end_idx >= gt.start_idx - tolerance):
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        tp = len(matched_pred)
        fp = len(predicted) - tp
        fn = len(ground_truth) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    @staticmethod
    def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(labels, scores)
        except:
            return 0.5

    @staticmethod
    def detection_delay(predicted: List[Anomaly], ground_truth: List[Anomaly]) -> float:
        delays = []
        for gt in ground_truth:
            for pred in predicted:
                if pred.start_idx >= gt.start_idx and pred.start_idx <= gt.end_idx:
                    delays.append(pred.start_idx - gt.start_idx)
                    break
        return np.mean(delays) if delays else float('inf')


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
