from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

class TimeSeriesFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    IRREGULAR = "irregular"

class DecompositionType(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

class ForecastMethod(Enum):
    NAIVE = "naive"
    MEAN = "mean"
    DRIFT = "drift"
    SES = "ses"
    HOLT = "holt"
    HOLT_WINTERS = "holt_winters"
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    LSTM = "lstm"
    PROPHET_LITE = "prophet_lite"

@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    timestamp: float
    value: float
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class TimeSeriesData:
    points: List[TimeSeriesPoint]
    frequency: TimeSeriesFrequency = TimeSeriesFrequency.IRREGULAR
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([p.timestamp for p in self.points])

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points])

    @property
    def length(self) -> int:
        return len(self.points)

    def to_regular(self, freq: TimeSeriesFrequency = TimeSeriesFrequency.DAILY) -> 'TimeSeriesData':
        if self.length < 2:
            return self

        ts = self.timestamps
        vals = self.values

        start = ts[0]
        end = ts[-1]

        if freq == TimeSeriesFrequency.HOURLY:
            step = 3600
        elif freq == TimeSeriesFrequency.DAILY:
            step = 86400
        elif freq == TimeSeriesFrequency.WEEKLY:
            step = 604800
        elif freq == TimeSeriesFrequency.MONTHLY:
            step = 2592000
        else:
            step = 86400

        new_ts = np.arange(start, end + step, step)
        new_vals = np.interp(new_ts, ts, vals)

        new_points = [TimeSeriesPoint(t, v) for t, v in zip(new_ts, new_vals)]
        return TimeSeriesData(new_points, freq, self.metadata)

@dataclass(frozen=True, slots=True)
class DecompositionResult:
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    observed: np.ndarray
    decomposition_type: DecompositionType

@dataclass(frozen=True, slots=True)
class ForecastResult:
    timestamps: np.ndarray
    values: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    method: ForecastMethod = ForecastMethod.NAIVE
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class AnomalyResult:
    indices: np.ndarray
    scores: np.ndarray
    threshold: float
    method: str
    timestamps: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None

class TimeSeriesPreprocessor:
    @staticmethod
    def remove_outliers(data: TimeSeriesData, method: str = "iqr", factor: float = 1.5) -> TimeSeriesData:
        vals = data.values
        if method == "iqr":
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            mask = (vals >= lower) & (vals <= upper)
        elif method == "zscore":
            z = np.abs((vals - np.mean(vals)) / (np.std(vals) + 1e-8))
            mask = z < factor
        elif method == "mad":
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            mask = np.abs(vals - median) < factor * mad * 1.4826
        else:
            mask = np.ones_like(vals, dtype=bool)

        filtered_points = [p for p, m in zip(data.points, mask) if m]
        return TimeSeriesData(filtered_points, data.frequency, data.metadata)

    @staticmethod
    def fill_missing(data: TimeSeriesData, method: str = "linear") -> TimeSeriesData:
        ts = data.timestamps
        vals = data.values

        if np.all(np.isfinite(vals)):
            return data

        mask = np.isfinite(vals)
        if method == "linear":
            filled = np.interp(ts, ts[mask], vals[mask])
        elif method == "forward":
            filled = np.where(mask, vals, np.nan)
            filled = np.maximum.accumulate(np.where(mask, ts, -np.inf))
            filled = np.interp(ts, filled[mask], vals[mask])
        elif method == "backward":
            filled = np.where(mask, vals, np.nan)
            filled = np.minimum.accumulate(np.where(mask[::-1], ts[::-1], np.inf))[::-1]
            filled = np.interp(ts, filled[mask], vals[mask])
        elif method == "mean":
            filled = np.where(mask, vals, np.mean(vals[mask]))
        else:
            filled = vals

        new_points = [TimeSeriesPoint(t, v, p.features, p.metadata)
                      for t, v, p in zip(ts, filled, data.points)]
        return TimeSeriesData(new_points, data.frequency, data.metadata)

    @staticmethod
    def normalize(data: TimeSeriesData, method: str = "zscore") -> Tuple[TimeSeriesData, Dict]:
        vals = data.values
        params = {}

        if method == "zscore":
            mean = np.mean(vals)
            std = np.std(vals) + 1e-8
            normed = (vals - mean) / std
            params = {"mean": mean, "std": std}
        elif method == "minmax":
            min_v = np.min(vals)
            max_v = np.max(vals)
            normed = (vals - min_v) / (max_v - min_v + 1e-8)
            params = {"min": min_v, "max": max_v}
        elif method == "robust":
            median = np.median(vals)
            mad = np.median(np.abs(vals - median)) + 1e-8
            normed = (vals - median) / mad
            params = {"median": median, "mad": mad}
        else:
            normed = vals

        new_points = [TimeSeriesPoint(p.timestamp, n, p.features, p.metadata)
                      for n, p in zip(normed, data.points)]
        return TimeSeriesData(new_points, data.frequency, data.metadata), params

    @staticmethod
    def difference(data: TimeSeriesData, order: int = 1) -> TimeSeriesData:
        vals = data.values
        diffed = vals
        for _ in range(order):
            diffed = np.diff(diffed, prepend=diffed[0])

        new_points = [TimeSeriesPoint(p.timestamp, d, p.features, p.metadata)
                      for d, p in zip(diffed, data.points)]
        return TimeSeriesData(new_points, data.frequency, data.metadata)

    @staticmethod
    def log_transform(data: TimeSeriesData) -> TimeSeriesData:
        vals = data.values
        min_val = np.min(vals)
        if min_val <= 0:
            vals = vals - min_val + 1
        log_vals = np.log(vals)
        new_points = [TimeSeriesPoint(p.timestamp, v, p.features, p.metadata)
                      for v, p in zip(log_vals, data.points)]
        return TimeSeriesData(new_points, data.frequency, data.metadata)

class Decomposer:
    def __init__(self, period: int = None):
        self.period = period

    def decompose(self, data: TimeSeriesData, decomposition_type: DecompositionType = DecompositionType.ADDITIVE) -> DecompositionResult:
        vals = data.values
        n = len(vals)

        if self.period is None:
            self.period = self._estimate_period(vals)

        if self.period >= n // 2:
            self.period = max(2, n // 2)

        trend = self._estimate_trend(vals)
        detrended = vals - trend if decomposition_type == DecompositionType.ADDITIVE else vals / (trend + 1e-8)

        seasonal = self._estimate_seasonal(detrended)
        if decomposition_type == DecompositionType.ADDITIVE:
            seasonal = seasonal - np.mean(seasonal)
        else:
            seasonal = seasonal / np.mean(seasonal)

        residual = detrended - seasonal if decomposition_type == DecompositionType.ADDITIVE else detrended / (seasonal + 1e-8)

        return DecompositionResult(trend, seasonal, residual, vals, decomposition_type)

    def _estimate_period(self, vals: np.ndarray) -> int:
        if len(vals) < 10:
            return 2

        n = len(vals)
        max_period = min(n // 2, 100)

        acf = np.correlate(vals - np.mean(vals), vals - np.mean(vals), mode='full')
        acf = acf[n-1:] / acf[n-1]

        peaks = signal.find_peaks(acf[1:max_period], distance=2)[0] + 1
        if len(peaks) > 0:
            return peaks[0]
        return min(12, n // 4)

    def _estimate_trend(self, vals: np.ndarray) -> np.ndarray:
        window = self.period if self.period % 2 == 1 else self.period + 1
        trend = np.convolve(vals, np.ones(window)/window, mode='same')

        half = window // 2
        trend[:half] = trend[half]
        trend[-half:] = trend[-half-1]
        return trend

    def _estimate_seasonal(self, detrended: np.ndarray) -> np.ndarray:
        n = len(detrended)
        seasonal = np.zeros(n)

        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])

        return seasonal

class TrendAnalyzer:
    @staticmethod
    def detect_trend(data: TimeSeriesData) -> Dict[str, Any]:
        vals = data.values
        n = len(vals)
        x = np.arange(n)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, vals)

        mk_stat, mk_p = TrendAnalyzer._mann_kendall(vals)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
            "std_err": std_err,
            "mann_kendall_stat": mk_stat,
            "mann_kendall_p": mk_p,
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        }

    @staticmethod
    def _mann_kendall(vals: np.ndarray) -> Tuple[float, float]:
        n = len(vals)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(vals[j] - vals[i])

        var_s = n * (n-1) * (2*n+5) / 18
        if s > 0:
            z = (s - 1) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / math.sqrt(var_s)
        else:
            z = 0

        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    @staticmethod
    def detect_changepoints(data: TimeSeriesData, penalty: float = 10.0) -> List[int]:
        vals = data.values
        n = len(vals)
        if n < 4:
            return []

        costs = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                segment = vals[i:j+1]
                costs[i, j] = np.sum((segment - np.mean(segment))**2)

        dp = np.full(n, np.inf)
        prev = np.full(n, -1, dtype=int)
        dp[0] = 0

        for i in range(1, n):
            for j in range(max(0, i-1), i):
                cost = dp[j] + costs[j, i] + penalty
                if cost < dp[i]:
                    dp[i] = cost
                    prev[i] = j

        changepoints = []
        idx = n - 1
        while prev[idx] != -1:
            changepoints.append(idx)
            idx = prev[idx]
        changepoints.reverse()
        return changepoints

class SeasonalAnalyzer:
    @staticmethod
    def seasonal_strength(data: TimeSeriesData) -> float:
        decomp = Decomposer().decompose(data)
        var_seasonal = np.var(decomp.seasonal)
        var_residual = np.var(decomp.residual)
        return var_seasonal / (var_seasonal + var_residual + 1e-8)

    @staticmethod
    def find_period(data: TimeSeriesData, max_period: int = None) -> int:
        vals = data.values
        n = len(vals)
        if max_period is None:
            max_period = min(n // 2, 100)

        acf = np.correlate(vals - np.mean(vals), vals - np.mean(vals), mode='full')
        acf = acf[n-1:] / acf[n-1]

        peaks = signal.find_peaks(acf[1:max_period], distance=2)[0] + 1
        if len(peaks) > 0:
            return peaks[0]
        return 1

    @staticmethod
    def seasonal_profile(data: TimeSeriesData, period: int = None) -> Dict[int, float]:
        if period is None:
            period = SeasonalAnalyzer.find_period(data)

        vals = data.values
        n = len(vals)
        profile = {}

        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                profile[i] = np.mean(vals[indices])

        return profile

class SpectralAnalyzer:
    @staticmethod
    def power_spectrum(data: TimeSeriesData) -> Tuple[np.ndarray, np.ndarray]:
        vals = data.values
        n = len(vals)

        fft_vals = fft(vals)
        freqs = fftfreq(n, d=1.0)
        psd = np.abs(fft_vals)**2

        pos_mask = freqs > 0
        return freqs[pos_mask], psd[pos_mask]

    @staticmethod
    def dominant_frequencies(data: TimeSeriesData, top_k: int = 5) -> List[Tuple[float, float]]:
        freqs, psd = SpectralAnalyzer.power_spectrum(data)
        idx = np.argsort(psd)[::-1][:top_k]
        return [(freqs[i], psd[i]) for i in idx]

    @staticmethod
    def periodogram(data: TimeSeriesData) -> Tuple[np.ndarray, np.ndarray]:
        vals = data.values
        freqs, psd = signal.periodogram(vals, fs=1.0)
        return freqs, psd

class Forecaster:
    def __init__(self, method: ForecastMethod = ForecastMethod.NAIVE):
        self.method = method
        self.fitted = False
        self.model_params = {}

    def fit(self, data: TimeSeriesData):
        if self.method == ForecastMethod.NAIVE:
            self._fit_naive(data)
        elif self.method == ForecastMethod.MEAN:
            self._fit_mean(data)
        elif self.method == ForecastMethod.DRIFT:
            self._fit_drift(data)
        elif self.method == ForecastMethod.SES:
            self._fit_ses(data)
        elif self.method == ForecastMethod.HOLT:
            self._fit_holt(data)
        elif self.method == ForecastMethod.HOLT_WINTERS:
            self._fit_holt_winters(data)
        self.fitted = True

    def predict(self, horizon: int) -> ForecastResult:
        if not self.fitted:
            raise ValueError("Model not fitted")

        if self.method == ForecastMethod.NAIVE:
            return self._predict_naive(horizon)
        elif self.method == ForecastMethod.MEAN:
            return self._predict_mean(horizon)
        elif self.method == ForecastMethod.DRIFT:
            return self._predict_drift(horizon)
        elif self.method == ForecastMethod.SES:
            return self._predict_ses(horizon)
        elif self.method == ForecastMethod.HOLT:
            return self._predict_holt(horizon)
        elif self.method == ForecastMethod.HOLT_WINTERS:
            return self._predict_holt_winters(horizon)

        return ForecastResult(np.arange(horizon), np.zeros(horizon), method=self.method)

    def _fit_naive(self, data: TimeSeriesData):
        self.model_params["last_value"] = data.values[-1]

    def _predict_naive(self, horizon: int) -> ForecastResult:
        vals = np.full(horizon, self.model_params["last_value"])
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.NAIVE)

    def _fit_mean(self, data: TimeSeriesData):
        self.model_params["mean"] = np.mean(data.values)

    def _predict_mean(self, horizon: int) -> ForecastResult:
        vals = np.full(horizon, self.model_params["mean"])
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.MEAN)

    def _fit_drift(self, data: TimeSeriesData):
        vals = data.values
        n = len(vals)
        self.model_params["slope"] = (vals[-1] - vals[0]) / (n - 1) if n > 1 else 0
        self.model_params["last_value"] = vals[-1]

    def _predict_drift(self, horizon: int) -> ForecastResult:
        slope = self.model_params["slope"]
        last = self.model_params["last_value"]
        vals = last + slope * np.arange(1, horizon + 1)
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.DRIFT)

    def _fit_ses(self, data: TimeSeriesData):
        vals = data.values
        alpha = 0.3
        level = vals[0]
        for v in vals[1:]:
            level = alpha * v + (1 - alpha) * level
        self.model_params["level"] = level
        self.model_params["alpha"] = alpha

    def _predict_ses(self, horizon: int) -> ForecastResult:
        level = self.model_params["level"]
        vals = np.full(horizon, level)
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.SES)

    def _fit_holt(self, data: TimeSeriesData):
        vals = data.values
        alpha, beta = 0.3, 0.1
        level = vals[0]
        trend = vals[1] - vals[0] if len(vals) > 1 else 0

        for v in vals[1:]:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend

        self.model_params["level"] = level
        self.model_params["trend"] = trend

    def _predict_holt(self, horizon: int) -> ForecastResult:
        level = self.model_params["level"]
        trend = self.model_params["trend"]
        vals = level + trend * np.arange(1, horizon + 1)
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.HOLT)

    def _fit_holt_winters(self, data: TimeSeriesData):
        vals = data.values
        period = Decomposer()._estimate_period(vals)
        alpha, beta, gamma = 0.3, 0.1, 0.1

        level = np.mean(vals[:period])
        trend = (np.mean(vals[period:2*period]) - np.mean(vals[:period])) / period if len(vals) > 2*period else 0
        seasonal = vals[:period] - level

        for i, v in enumerate(vals):
            prev_level = level
            level = alpha * (v - seasonal[i % period]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[i % period] = gamma * (v - level) + (1 - gamma) * seasonal[i % period]

        self.model_params["level"] = level
        self.model_params["trend"] = trend
        self.model_params["seasonal"] = seasonal
        self.model_params["period"] = period

    def _predict_holt_winters(self, horizon: int) -> ForecastResult:
        level = self.model_params["level"]
        trend = self.model_params["trend"]
        seasonal = self.model_params["seasonal"]
        period = self.model_params["period"]

        vals = np.array([level + (i+1)*trend + seasonal[i % period] for i in range(horizon)])
        return ForecastResult(np.arange(horizon), vals, method=ForecastMethod.HOLT_WINTERS)

class AnomalyDetector:
    @staticmethod
    def zscore(data: TimeSeriesData, threshold: float = 3.0) -> AnomalyResult:
        vals = data.values
        mean = np.mean(vals)
        std = np.std(vals) + 1e-8
        scores = np.abs((vals - mean) / std)
        indices = np.where(scores > threshold)[0]
        return AnomalyResult(indices, scores, threshold, "zscore",
                            data.timestamps[indices] if len(indices) > 0 else None,
                            vals[indices] if len(indices) > 0 else None)

    @staticmethod
    def modified_zscore(data: TimeSeriesData, threshold: float = 3.5) -> AnomalyResult:
        vals = data.values
        median = np.median(vals)
        mad = np.median(np.abs(vals - median)) + 1e-8
        scores = 0.6745 * np.abs(vals - median) / mad
        indices = np.where(scores > threshold)[0]
        return AnomalyResult(indices, scores, threshold, "modified_zscore",
                            data.timestamps[indices] if len(indices) > 0 else None,
                            vals[indices] if len(indices) > 0 else None)

    @staticmethod
    def iqr(data: TimeSeriesData, multiplier: float = 1.5) -> AnomalyResult:
        vals = data.values
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        scores = np.zeros_like(vals)
        scores[vals < lower] = (lower - vals[vals < lower]) / iqr
        scores[vals > upper] = (vals[vals > upper] - upper) / iqr
        indices = np.where(scores > 0)[0]
        return AnomalyResult(indices, scores, multiplier, "iqr",
                            data.timestamps[indices] if len(indices) > 0 else None,
                            vals[indices] if len(indices) > 0 else None)

    @staticmethod
    def isolation_forest(data: TimeSeriesData, contamination: float = 0.1) -> AnomalyResult:
        try:
            from sklearn.ensemble import IsolationForest
            vals = data.values.reshape(-1, 1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            scores = -clf.fit_predict(vals)
            anomaly_scores = -clf.score_samples(vals)
            indices = np.where(scores == 1)[0]
            return AnomalyResult(indices, anomaly_scores, contamination, "isolation_forest",
                                data.timestamps[indices] if len(indices) > 0 else None,
                                vals[indices] if len(indices) > 0 else None)
        except ImportError:
            return AnomalyResult(np.array([]), np.array([]), contamination, "isolation_forest")

    @staticmethod
    def rolling_std(data: TimeSeriesData, window: int = 10, threshold: float = 3.0) -> AnomalyResult:
        vals = data.values
        scores = np.zeros_like(vals)

        for i in range(window, len(vals)):
            window_vals = vals[i-window:i]
            mean = np.mean(window_vals)
            std = np.std(window_vals) + 1e-8
            scores[i] = abs(vals[i] - mean) / std

        indices = np.where(scores > threshold)[0]
        return AnomalyResult(indices, scores, threshold, "rolling_std",
                            data.timestamps[indices] if len(indices) > 0 else None,
                            vals[indices] if len(indices) > 0 else None)

class TimeSeriesMetrics:
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(TimeSeriesMetrics.mse(y_true, y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        mae = TimeSeriesMetrics.mae(y_true, y_pred)
        scale = np.mean(np.abs(np.diff(y_train)))
        return mae / (scale + 1e-8)

    @staticmethod
    def evaluate_forecast(data: TimeSeriesData, forecast: ForecastResult,
                         train_data: TimeSeriesData = None) -> Dict[str, float]:
        actual = data.values[-len(forecast.values):]
        pred = forecast.values

        metrics = {
            "mae": TimeSeriesMetrics.mae(actual, pred),
            "mse": TimeSeriesMetrics.mse(actual, pred),
            "rmse": TimeSeriesMetrics.rmse(actual, pred),
            "mape": TimeSeriesMetrics.mape(actual, pred),
            "smape": TimeSeriesMetrics.smape(actual, pred),
        }

        if train_data is not None:
            metrics["mase"] = TimeSeriesMetrics.mase(actual, pred, train_data.values)

        return metrics

class TimeSeriesPipeline:
    def __init__(self):
        self.preprocessors: List[Callable] = []
        self.decomposer: Optional[Decomposer] = None
        self.forecaster: Optional[Forecaster] = None
        self.anomaly_detector: Optional[Callable] = None

    def add_preprocessor(self, func: Callable):
        self.preprocessors.append(func)

    def set_decomposer(self, decomposer: Decomposer):
        self.decomposer = decomposer

    def set_forecaster(self, forecaster: Forecaster):
        self.forecaster = forecaster

    def set_anomaly_detector(self, detector: Callable):
        self.anomaly_detector = detector

    def process(self, data: TimeSeriesData) -> TimeSeriesData:
        result = data
        for proc in self.preprocessors:
            result = proc(result)
        return result

    def decompose(self, data: TimeSeriesData) -> Optional[DecompositionResult]:
        if self.decomposer:
            return self.decomposer.decompose(data)
        return None

    def forecast(self, data: TimeSeriesData, horizon: int) -> Optional[ForecastResult]:
        if self.forecaster:
            self.forecaster.fit(data)
            return self.forecaster.predict(horizon)
        return None

    def detect_anomalies(self, data: TimeSeriesData) -> Optional[AnomalyResult]:
        if self.anomaly_detector:
            return self.anomaly_detector(data)
        return None

    def run_full(self, data: TimeSeriesData, horizon: int = 10) -> Dict[str, Any]:
        processed = self.process(data)
        decomposition = self.decompose(processed)
        forecast = self.forecast(processed, horizon)
        anomalies = self.detect_anomalies(processed)

        return {
            "processed_data": processed,
            "decomposition": decomposition,
            "forecast": forecast,
            "anomalies": anomalies,
            "metrics": TimeSeriesMetrics.evaluate_forecast(data, forecast) if forecast else {}
        }

def create_pipeline() -> TimeSeriesPipeline:
    pipeline = TimeSeriesPipeline()
    pipeline.add_preprocessor(lambda d: TimeSeriesPreprocessor.fill_missing(d))
    pipeline.add_preprocessor(lambda d: TimeSeriesPreprocessor.remove_outliers(d, "iqr"))
    pipeline.set_decomposer(Decomposer())
    pipeline.set_forecaster(Forecaster(ForecastMethod.HOLT_WINTERS))
    pipeline.set_anomaly_detector(AnomalyDetector.modified_zscore)
    return pipeline