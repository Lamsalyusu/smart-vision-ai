from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, Counter, deque
import warnings


class TimeSeriesFrequency(Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class DecompositionType(Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class StationarityTest(Enum):
    ADF = "adf"
    KPSS = "kpss"
    PP = "pp"
    ZIVOT_ANDREWS = "zivot_andrews"


@dataclass(frozen=True, slots=True)
class TimeSeriesData:
    timestamps: np.ndarray
    values: np.ndarray
    frequency: Optional[TimeSeriesFrequency] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.values)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[float, float]:
        return self.timestamps[idx], self.values[idx]


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    decomposition_type: DecompositionType


@dataclass(frozen=True, slots=True)
class StationarityTestResult:
    test: StationarityTest
    statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]


@dataclass(frozen=True, slots=True)
class ARIMAOrder:
    p: int
    d: int
    q: int
    seasonal_order: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True, slots=True)
class ARIMAResult:
    order: ARIMAOrder
    coefficients: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    residuals: np.ndarray
    fitted_values: np.ndarray


class TimeSeriesPreprocessor:
    @staticmethod
    def resample(
        data: TimeSeriesData,
        target_frequency: TimeSeriesFrequency,
        method: str = "mean"
    ) -> TimeSeriesData:
        return data

    @staticmethod
    def interpolate_missing(
        data: TimeSeriesData,
        method: str = "linear",
        limit: int = None
    ) -> TimeSeriesData:
        values = data.values.copy()
        mask = np.isnan(values)

        if not np.any(mask):
            return data

        indices = np.arange(len(values))
        valid = ~mask

        if method == "linear":
            values[mask] = np.interp(indices[mask], indices[valid], values[valid])
        elif method == "nearest":
            values[mask] = values[valid[np.searchsorted(indices[valid], indices[mask], side='left') - 1]]
        elif method == "spline":
            from scipy.interpolate import UnivariateSpline
            spl = UnivariateSpline(indices[valid], values[valid], k=3)
            values[mask] = spl(indices[mask])

        return TimeSeriesData(
            timestamps=data.timestamps,
            values=values,
            frequency=data.frequency,
            metadata=data.metadata
        )

    @staticmethod
    def remove_outliers(
        data: TimeSeriesData,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> TimeSeriesData:
        values = data.values.copy()

        if method == "iqr":
            q1 = np.nanpercentile(values, 25)
            q3 = np.nanpercentile(values, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = (values < lower) | (values > upper)
            values[outliers] = np.nan
        elif method == "zscore":
            z_scores = np.abs((values - np.nanmean(values)) / (np.nanstd(values) + 1e-8))
            values[z_scores > threshold] = np.nan
        elif method == "modified_zscore":
            median = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values - median))
            modified_z = 0.6745 * np.abs(values - median) / (mad + 1e-8)
            values[modified_z > threshold] = np.nan

        return TimeSeriesData(
            timestamps=data.timestamps,
            values=values,
            frequency=data.frequency,
            metadata=data.metadata
        )

    @staticmethod
    def difference(
        data: TimeSeriesData,
        order: int = 1,
        lag: int = 1
    ) -> TimeSeriesData:
        values = data.values.copy()

        for _ in range(order):
            diff = np.diff(values, n=lag)
            values = np.concatenate([np.full(lag, np.nan), diff])

        return TimeSeriesData(
            timestamps=data.timestamps,
            values=values,
            frequency=data.frequency,
            metadata=data.metadata
        )

    @staticmethod
    def log_transform(data: TimeSeriesData) -> TimeSeriesData:
        values = data.values.copy()
        values[values <= 0] = np.nan
        values = np.log(values)
        return TimeSeriesData(
            timestamps=data.timestamps,
            values=values,
            frequency=data.frequency,
            metadata=data.metadata
        )

    @staticmethod
    def box_cox_transform(
        data: TimeSeriesData,
        lambda_param: float = None
    ) -> Tuple[TimeSeriesData, float]:
        values = data.values.copy()
        values[values <= 0] = np.nan

        if lambda_param is None:
            lambda_param = 0.0

        if abs(lambda_param) < 1e-8:
            transformed = np.log(values)
        else:
            transformed = (values**lambda_param - 1) / lambda_param

        return TimeSeriesData(
            timestamps=data.timestamps,
            values=transformed,
            frequency=data.frequency,
            metadata=data.metadata
        ), lambda_param


class SeasonalDecomposition:
    def __init__(
        self,
        period: int = None,
        decomposition_type: DecompositionType = DecompositionType.ADDITIVE,
        robust: bool = False
    ):
        self.period = period
        self.decomposition_type = decomposition_type
        self.robust = robust

    def decompose(self, data: TimeSeriesData) -> DecompositionResult:
        n = len(data.values)
        period = self.period or self._estimate_period(data)

        if self.decomposition_type == DecompositionType.ADDITIVE:
            return self._additive_decomposition(data, period)
        else:
            return self._multiplicative_decomposition(data, period)

    def _estimate_period(self, data: TimeSeriesData) -> int:
        n = len(data.values)
        if n < 20:
            return min(4, n // 2)

        acf = self._autocorr(data.values, nlags=min(50, n // 2))
        peaks = self._find_peaks(acf[1:], min_distance=period//2 if (period := 4) else 2)

        if len(peaks) > 0:
            return peaks[0] + 1

        if data.frequency:
            freq_map = {
                TimeSeriesFrequency.SECOND: 60,
                TimeSeriesFrequency.MINUTE: 60,
                TimeSeriesFrequency.HOUR: 24,
                TimeSeriesFrequency.DAY: 7,
                TimeSeriesFrequency.WEEK: 52,
                TimeSeriesFrequency.MONTH: 12,
                TimeSeriesFrequency.QUARTER: 4,
                TimeSeriesFrequency.YEAR: 1
            }
            return freq_map.get(data.frequency, 4)

        return 4

    def _autocorr(self, x: np.ndarray, nlags: int = 40) -> np.ndarray:
        n = len(x)
        x = x - np.nanmean(x)
        acf = np.zeros(nlags + 1)
        var = np.nanvar(x)

        for k in range(nlags + 1):
            if k == 0:
                acf[k] = 1.0
            else:
                acf[k] = np.nanmean(x[:-k] * x[k:]) / var
        return acf

    def _find_peaks(self, x: np.ndarray, min_distance: int = 2) -> List[int]:
        peaks = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        return peaks

    def _additive_decomposition(self, data: TimeSeriesData, period: int) -> DecompositionResult:
        n = len(data.values)
        values = data.values.copy()

        trend = self._moving_average(values, period)
        detrended = values - trend

        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal[indices] = np.nanmean(detrended[indices])

        seasonal = seasonal - np.nanmean(seasonal)
        residual = detrended - seasonal

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            decomposition_type=DecompositionType.ADDITIVE
        )

    def _multiplicative_decomposition(self, data: TimeSeriesData, period: int) -> DecompositionResult:
        n = len(data.values)
        values = data.values.copy()
        values[values <= 0] = np.nan

        trend = self._moving_average(values, period)
        detrended = values / trend

        seasonal = np.ones(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal[indices] = np.nanmean(detrended[indices])

        seasonal = seasonal / np.nanmean(seasonal)
        residual = detrended / seasonal

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            decomposition_type=DecompositionType.MULTIPLICATIVE
        )

    def _moving_average(self, values: np.ndarray, window: int) -> np.ndarray:
        if window % 2 == 0:
            window += 1

        pad = window // 2
        padded = np.pad(values, (pad, pad), mode='edge')
        kernel = np.ones(window) / window
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed


class StationarityTests:
    @staticmethod
    def adf_test(data: TimeSeriesData, max_lag: int = None) -> StationarityTestResult:
        values = data.values
        values = values[~np.isnan(values)]

        if len(values) < 10:
            return StationarityTestResult(
                test=StationarityTest.ADF,
                statistic=0.0,
                p_value=1.0,
                is_stationary=False,
                critical_values={}
            )

        n = len(values)
        max_lag = max_lag or int(12 * (n / 100)**0.25)

        best_adf = float('inf')
        best_lag = 0

        for lag in range(max_lag + 1):
            try:
                y = values[lag:]
                x = np.column_stack([
                    np.ones(len(values) - lag),
                    values[:-lag] if lag > 0 else np.zeros(len(values)),
                    np.arange(len(values) - lag)
                ])

                if lag > 0:
                    for l in range(1, lag):
                        x = np.column_stack([x, np.diff(values, n=l, prepend=np.nan)[lag:]])

                x = x[~np.any(np.isnan(x), axis=1)]
                y = y[:len(x)]

                if len(x) > x.shape[1]:
                    coeffs = np.linalg.lstsq(x, y, rcond=None)[0]
                    residuals = y - x @ coeffs
                    sigma2 = np.var(residuals)
                    se = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[1, 1])
                    adf_stat = (coeffs[1] - 1) / (np.sqrt(np.diag(np.linalg.inv(x.T @ x))[1]) + 1e-8)

                    if adf_stat < best_adf:
                        best_adf = adf_stat
                        best_lag = lag
            except:
                continue

        critical_values = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57
        }

        p_value = 0.01 if best_adf < critical_values['1%'] else \
                  0.05 if best_adf < critical_values['5%'] else \
                  0.10 if best_adf < critical_values['10%'] else 0.5

        return StationarityTestResult(
            test=StationarityTest.ADF,
            statistic=best_adf,
            p_value=p_value,
            is_stationary=best_adf < critical_values['5%'],
            critical_values=critical_values
        )

    @staticmethod
    def kpss_test(data: TimeSeriesData, regression: str = 'c') -> StationarityTestResult:
        values = data.values
        values = values[~np.isnan(values)]

        if len(values) < 10:
            return StationarityTestResult(
                test=StationarityTest.KPSS,
                statistic=0.0,
                p_value=1.0,
                is_stationary=True,
                critical_values={}
            )

        n = len(values)
        y = values - np.mean(values)
        s = np.cumsum(y)
        eta = np.sum(s**2) / (n**2 * np.var(values))

        if regression == 'ct':
            t = np.arange(n)
            trend = np.column_stack([np.ones(n), t])
            beta = np.linalg.lstsq(trend, s, rcond=None)[0]
            s_detrended = s - trend @ beta
            eta = np.sum(s_detrended**2) / (n**2 * np.var(values))

        critical_values = {
            '1%': 0.739,
            '5%': 0.463,
            '10%': 0.347
        } if regression == 'c' else {
            '1%': 0.216,
            '5%': 0.146,
            '10%': 0.119
        }

        p_value = 0.01 if eta > list(critical_values.values())[0] else \
                  0.05 if eta > list(critical_values.values())[1] else \
                  0.10 if eta > list(critical_values.values())[2] else 0.5

        return StationarityTestResult(
            test=StationarityTest.KPSS,
            statistic=eta,
            p_value=p_value,
            is_stationary=eta < list(critical_values.values())[1],
            critical_values=critical_values
        )


class ARIMAModel:
    def __init__(self, order: ARIMAOrder):
        self.order = order
        self.coefficients: Optional[np.ndarray] = None
        self.fitted_values: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None

    def fit(self, data: TimeSeriesData, method: str = 'css-mle') -> ARIMAResult:
        p, d, q = self.order.p, self.order.d, self.order.q
        values = data.values.copy()
        values = values[~np.isnan(values)]

        for _ in range(d):
            values = np.diff(values)

        n = len(values)
        if n < max(p, q) + 10:
            raise ValueError("Insufficient data for ARIMA fitting")

        X, y = self._build_regression_matrix(values, p, q)

        if len(X) == 0:
            raise ValueError("Insufficient data for regression")

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        self.coefficients = coeffs
        fitted = X @ coeffs
        residuals = values[p+q:] - fitted

        self.fitted_values = np.concatenate([np.full(p+q, np.nan), fitted])
        self.residuals = np.concatenate([np.full(p+q, np.nan), residuals])

        log_likelihood = -0.5 * len(fitted) * np.log(2 * np.pi * np.var(self.residuals[~np.isnan(self.residuals)]))
        aic = -2 * log_likelihood + 2 * (p + q + 1)
        bic = -2 * log_likelihood + np.log(len(fitted)) * (p + q + 1)

        return ARIMAResult(
            order=self.order,
            coefficients=coeffs,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            residuals=self.residuals,
            fitted_values=self.fitted_values
        )

    def _build_regression_matrix(self, values: np.ndarray, p: int, q: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(values)
        if n < p + q + 1:
            return np.array([]), np.array([])

        X = []
        y = []

        for i in range(p + q, n):
            row = []
            for j in range(1, p + 1):
                row.append(values[i - j])
            for j in range(1, q + 1):
                row.append(0)

            X.append(row)
            y.append(values[i])

        return np.array(X), np.array(y)

    def forecast(self, steps: int = 10) -> np.ndarray:
        if self.coefficients is None:
            return np.zeros(steps)

        p, d, q = self.order.p, self.order.d, self.order.q
        history = np.zeros(0)

        if hasattr(self, 'fitted_values') and self.fitted_values is not None:
            valid_fitted = self.fitted_values[~np.isnan(self.fitted_values)]
            history = np.concatenate([self.residuals[~np.isnan(self.residuals)], self.fitted_values[~np.isnan(self.fitted_values)]])

        forecasts = []
        for _ in range(steps):
            pred = 0
            for i in range(1, p + 1):
                if len(history) >= i:
                    pred += self.coefficients[i - 1] * history[-i]

            for i in range(1, q + 1):
                if len(self.residuals) >= i and not np.isnan(self.residuals[-i]):
                    pred += self.coefficients[p + i - 1] * self.residuals[-i]

            forecasts.append(pred)
            history = np.append(history, pred)

        return np.array(forecasts)


class TimeSeriesForecaster:
    def __init__(
        self,
        model_type: str = "arima",
        order: ARIMAOrder = None,
        seasonal_order: Tuple[int, int, int, int] = None
    ):
        self.model_type = model_type
        self.order = order or ARIMAOrder(1, 1, 1)
        self.seasonal_order = seasonal_order
        self.model: Optional[ARIMAModel] = None

    def fit(self, data: TimeSeriesData) -> ARIMAResult:
        if self.model_type == "arima":
            self.model = ARIMAModel(self.order)
            return self.model.fit(data)
        elif self.model_type == "sarima":
            pass
        elif self.model_type == "ets":
            pass
        return ARIMAResult(
            order=self.order,
            coefficients=np.array([]),
            log_likelihood=0.0,
            aic=0.0,
            bic=0.0,
            residuals=np.array([]),
            fitted_values=np.array([])
        )

    def forecast(self, steps: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None:
            return np.zeros(steps), np.zeros(steps), np.zeros(steps)

        forecast = self.model.forecast(steps)
        lower = forecast - 1.96 * np.std(self.model.residuals[~np.isnan(self.model.residuals)])
        upper = forecast + 1.96 * np.std(self.model.residuals[~np.isnan(self.model.residuals)])

        return forecast, lower, upper


class TimeSeriesClustering:
    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "dtw",
        max_iter: int = 100
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None

    def fit(self, data: List[TimeSeriesData]):
        n = len(data)
        if n < self.n_clusters:
            self.labels_ = np.arange(n)
            return self

        if self.metric == "dtw":
            distances = self._compute_dtw_distances(data)
        elif self.metric == "euclidean":
            lengths = [len(d.values) for d in data]
            max_len = max(lengths)
            padded = np.array([
                np.pad(d.values, (0, max_len - len(d.values)), mode='constant', constant_values=np.nan)
                for d in data
            ])
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    valid = ~np.isnan(padded[i]) & ~np.isnan(padded[j])
                    if np.any(valid):
                        distances[i, j] = np.sqrt(np.nansum((padded[i][valid] - padded[j][valid])**2))
                    else:
                        distances[i, j] = np.inf
                    distances[j, i] = distances[i, j]

        else:
            distances = np.random.rand(n, n)

        self.labels_ = self._kmedoids(distances)
        return self

    def _compute_dtw_distances(self, data: List[TimeSeriesData]) -> np.ndarray:
        n = len(data)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = self._dtw_distance(data[i].values, data[j].values)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _dtw_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        n, m = len(x), len(y)
        if n == 0 or m == 0:
            return float('inf')

        dtw = np.full((n+1, m+1), np.inf)
        dtw[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = (x[i-1] - y[j-1])**2
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

        return math.sqrt(dtw[n, m])

    def _kmedoids(self, distances: np.ndarray) -> np.ndarray:
        n = distances.shape[0]
        medoids = np.random.choice(n, self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            labels = np.argmin(distances[:, medoids], axis=1)

            new_medoids = medoids.copy()
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                if len(cluster_points) == 0:
                    continue

                costs = np.sum(distances[np.ix_(cluster_points, cluster_points)], axis=1)
                best_medoid = cluster_points[np.argmin(costs)]
                new_medoids[k] = best_medoid

            if np.array_equal(medoids, new_medoids):
                break
            medoids = new_medoids

        return np.array([np.where(medoids == i)[0][0] if i in medoids else 0 for i in range(len(distances))])


class TimeSeriesMetrics:
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred)**2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        mae = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        return float(mae / (mae_naive + 1e-8))

    @staticmethod
    def aic(n: int, k: int, mse: float) -> float:
        return float(n * math.log(mse) + 2 * k)

    @staticmethod
    def bic(n: int, k: int, mse: float) -> float:
        return float(n * math.log(mse) + k * math.log(n))


class TimeSeriesVisualizer:
    @staticmethod
    def plot_decomposition(
        data: TimeSeriesData,
        result: DecompositionResult,
        image_size: Tuple[int, int] = (1200, 800)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n = len(data.values)
        if n == 0:
            return vis

        panel_height = image_size[1] // 4
        panel_width = image_size[0] - 100

        for panel_idx, (name, values) in enumerate([
            ("Original", data.values),
            ("Trend", result.trend),
            ("Seasonal", result.seasonal),
            ("Residual", result.residual)
        ]):
            y_start = panel_idx * panel_height + 50
            y_end = y_start + panel_height - 20

            if panel_idx == 0:
                cv2.putText(vis, name, (50, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(vis, name, (50, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            valid = ~np.isnan(values)
            if not np.any(valid):
                continue

            valid_indices = np.where(valid)[0]
            x_vals = np.linspace(50, image_size[0] - 50, len(valid_indices)).astype(int)
            y_vals = np.interp(
                np.linspace(0, 1, len(valid_indices)),
                [0, 1],
                [y_end - 20, y_start + 20]
            ).astype(int)

            for i in range(len(valid_indices) - 1):
                cv2.line(vis, (x_vals[i], y_vals[i]), (x_vals[i+1], y_vals[i+1]), (0, 255, 0), 1)

        return vis

    @staticmethod
    def plot_forecast(
        historical: TimeSeriesData,
        forecast: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        image_size: Tuple[int, int] = (1200, 600)
    ) -> np.ndarray:
        import cv2
        vis = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        n_hist = len(historical.values)
        n_fore = len(forecast)
        total = n_hist + n_fore

        x_vals = np.linspace(50, image_size[0] - 50, total).astype(int)
        y_min = min(np.nanmin(historical.values), np.nanmin(forecast))
        y_max = max(np.nanmax(historical.values), np.nanmax(forecast))
        y_range = y_max - y_min if y_max > y_min else 1

        for i in range(n_hist - 1):
            if not np.isnan(historical.values[i]) and not np.isnan(historical.values[i+1]):
                y1 = int((1 - (historical.values[i] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                y2 = int((1 - (historical.values[i+1] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                cv2.line(vis, (int(x_vals[i]), y1), (int(x_vals[i+1]), y2), (0, 255, 0), 2)

        for i in range(n_fore - 1):
            if not np.isnan(forecast[i]) and not np.isnan(forecast[i+1]):
                y1 = int((1 - (forecast[i] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                y2 = int((1 - (forecast[i+1] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                x1 = int(x_vals[n_hist + i])
                x2 = int(x_vals[n_hist + i + 1])
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for i in range(n_fore - 1):
            if not np.isnan(lower[i]) and not np.isnan(upper[i]) and not np.isnan(lower[i+1]) and not np.isnan(upper[i+1]):
                y1_low = int((1 - (lower[i] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                y2_low = int((1 - (lower[i+1] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                y1_up = int((1 - (upper[i] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                y2_up = int((1 - (upper[i+1] - y_min) / (y_max - y_min + 1e-6)) * (image_size[1] - 100)) + 50
                x1 = int(x_vals[n_hist + i])
                x2 = int(x_vals[n_hist + i + 1])
                pts = np.array([[x1, y1_low], [x2, y2_low], [x2, y2_up], [x1, y1_up]], np.int32)
                cv2.fillPoly(vis, [pts], (0, 0, 255, 50))

        return vis


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
