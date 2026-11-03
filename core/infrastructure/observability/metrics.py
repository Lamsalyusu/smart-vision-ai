from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# TODO: Add support for exponential histograms (Prometheus native)
# FIXME: Label cardinality can explode - need cardinality limits
# XXX: Summary quantiles are approximate - not exact
# HACK: Using threading.Lock instead of asyncio.Lock for sync/async compat


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass(frozen=True, slots=True)
class LabelSet:
    labels: Tuple[Tuple[str, str], ...] = ()

    def __getitem__(self, key: str) -> Optional[str]:
        for k, v in self.labels:
            if k == key:
                return v
        return None

    def __contains__(self, key: str) -> bool:
        return any(k == key for k, _ in self.labels)

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "LabelSet":
        return cls(labels=tuple(sorted(d.items())))


@dataclass
class MetricSample:
    name: str
    metric_type: MetricType
    labels: LabelSet
    value: float
    timestamp: float = field(default_factory=time.time)
    exemplar: Optional[Dict[str, str]] = None


@dataclass
class HistogramBucket:
    le: float
    count: int = 0


@dataclass
class SummaryQuantile:
    quantile: float
    value: float
    error: float


class Metric:
    def __init__(self, name: str, metric_type: MetricType, description: str = "", unit: str = ""):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.unit = unit
        self._lock = threading.Lock()

    def record(self, value: float, labels: Dict[str, str] = None):
        raise NotImplementedError

    def collect(self) -> List[MetricSample]:
        raise NotImplementedError


class Counter(Metric):
    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, MetricType.COUNTER, description, unit)
        self._values: Dict[LabelSet, float] = defaultdict(float)
        self._total = 0.0

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None):
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            self._values[label_set] += value
            self._total += value

    def record(self, value: float, labels: Dict[str, str] = None):
        self.inc(value, labels)

    def get(self, labels: Dict[str, str] = None) -> float:
        with self._lock:
            if labels is None:
                return self._total
            return self._values.get(LabelSet.from_dict(labels), 0.0)

    def collect(self) -> List[MetricSample]:
        with self._lock:
            samples = []
            for labels, value in self._values.items():
                samples.append(MetricSample(
                    name=self.name,
                    metric_type=self.metric_type,
                    labels=labels,
                    value=value,
                ))
            samples.append(MetricSample(
                name=self.name,
                metric_type=self.metric_type,
                labels=LabelSet(),
                value=self._total,
            ))
            return samples


class Gauge(Metric):
    def __init__(self, name: str, description: str = "", unit: str = ""):
        super().__init__(name, MetricType.GAUGE, description, unit)
        self._values: Dict[LabelSet, float] = {}
        self._callbacks: List[Callable[[], float]] = []

    def set(self, value: float, labels: Dict[str, str] = None):
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            self._values[label_set] = value

    def inc(self, value: float = 1.0, labels: Dict[str, str] = None):
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            self._values[label_set] = self._values.get(label_set, 0.0) + value

    def dec(self, value: float = 1.0, labels: Dict[str, str] = None):
        self.inc(-value, labels)

    def set_callback(self, callback: Callable[[], float], labels: Dict[str, str] = None):
        with self._lock:
            self._callbacks.append((callback, LabelSet.from_dict(labels or {})))

    def record(self, value: float, labels: Dict[str, str] = None):
        self.set(value, labels)

    def get(self, labels: Dict[str, str] = None) -> float:
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            return self._values.get(label_set, 0.0)

    def collect(self) -> List[MetricSample]:
        with self._lock:
            samples = []
            for callback, labels in self._callbacks:
                try:
                    value = callback()
                    samples.append(MetricSample(
                        name=self.name,
                        metric_type=self.metric_type,
                        labels=labels,
                        value=value,
                    ))
                except Exception as e:
                    logger.warning(f"Gauge callback failed: {e}")

            for labels, value in self._values.items():
                samples.append(MetricSample(
                    name=self.name,
                    metric_type=self.metric_type,
                    labels=labels,
                    value=value,
                ))
            return samples


class Histogram(Metric):
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: List[float] = None,
    ):
        super().__init__(name, MetricType.HISTOGRAM, description, unit)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._buckets: Dict[LabelSet, List[int]] = defaultdict(lambda: [0] * len(self.buckets))
        self._counts: Dict[LabelSet, int] = defaultdict(int)
        self._sums: Dict[LabelSet, float] = defaultdict(float)

    def observe(self, value: float, labels: Dict[str, str] = None):
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            for i, le in enumerate(self.buckets):
                if value <= le:
                    self._buckets[label_set][i] += 1
            self._counts[label_set] += 1
            self._sums[label_set] += value

    def record(self, value: float, labels: Dict[str, str] = None):
        self.observe(value, labels)

    def collect(self) -> List[MetricSample]:
        with self._lock:
            samples = []
            for labels, bucket_counts in self._buckets.items():
                count = self._counts[labels]
                sum_val = self._sums[labels]
                for i, le in enumerate(self.buckets):
                    samples.append(MetricSample(
                        name=f"{self.name}_bucket",
                        metric_type=MetricType.COUNTER,
                        labels=LabelSet.from_dict({**dict(labels.labels), "le": str(le)}),
                        value=bucket_counts[i],
                    ))
                samples.append(MetricSample(
                    name=f"{self.name}_count",
                    metric_type=MetricType.COUNTER,
                    labels=labels,
                    value=count,
                ))
                samples.append(MetricSample(
                    name=f"{self.name}_sum",
                    metric_type=MetricType.COUNTER,
                    labels=labels,
                    value=sum_val,
                ))
            return samples


class Summary(Metric):
    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        quantiles: List[float] = None,
        max_age_seconds: float = 600.0,
        max_size: int = 500,
    ):
        super().__init__(name, MetricType.SUMMARY, description, unit)
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.max_age = max_age_seconds
        self.max_size = max_size
        self._values: Dict[LabelSet, List[Tuple[float, float]]] = defaultdict(list)

    def observe(self, value: float, labels: Dict[str, str] = None):
        with self._lock:
            label_set = LabelSet.from_dict(labels or {})
            now = time.time()
            self._values[label_set].append((now, value))
            self._cleanup(label_set)

    def record(self, value: float, labels: Dict[str, str] = None):
        self.observe(value, labels)

    def _cleanup(self, labels: LabelSet):
        now = time.time()
        values = self._values[labels]
        cutoff = now - self.max_age
        self._values[labels] = [(t, v) for t, v in values if t > cutoff][-self.max_size:]

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(q * (len(sorted_values) - 1))
        return sorted_values[idx]

    def collect(self) -> List[MetricSample]:
        with self._lock:
            samples = []
            for labels, values_with_time in self._values.items():
                values = [v for _, v in values_with_time]
                count = len(values)
                sum_val = sum(values)

                samples.append(MetricSample(
                    name=f"{self.name}_count",
                    metric_type=MetricType.COUNTER,
                    labels=labels,
                    value=count,
                ))
                samples.append(MetricSample(
                    name=f"{self.name}_sum",
                    metric_type=MetricType.COUNTER,
                    labels=labels,
                    value=sum_val,
                ))

                for q in self.quantiles:
                    quantile_val = self._quantile(values, q)
                    samples.append(MetricSample(
                        name=self.name,
                        metric_type=MetricType.SUMMARY,
                        labels=LabelSet.from_dict({**dict(labels.labels), "quantile": str(q)}),
                        value=quantile_val,
                    ))
            return samples


class MetricsRegistry:
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str = "", unit: str = "") -> Counter:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description, unit)
            return self._metrics[name]

    def gauge(self, name: str, description: str = "", unit: str = "") -> Gauge:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description, unit)
            return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: List[float] = None,
    ) -> Histogram:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, unit, buckets)
            return self._metrics[name]

    def summary(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        quantiles: List[float] = None,
    ) -> Summary:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Summary(name, description, unit, quantiles)
            return self._metrics[name]

    def get(self, name: str) -> Optional[Metric]:
        with self._lock:
            return self._metrics.get(name)

    def collect_all(self) -> List[MetricSample]:
        with self._lock:
            all_samples = []
            for metric in self._metrics.values():
                all_samples.extend(metric.collect())
            return all_samples

    def get_metric_names(self) -> List[str]:
        with self._lock:
            return list(self._metrics.keys())


class PrometheusExporter:
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry

    def generate_metrics(self) -> str:
        lines = []
        for sample in self.registry.collect_all():
            metric_name = sample.name
            if sample.labels.labels:
                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in sample.labels.labels) + "}"
            else:
                label_str = ""
            lines.append(f"{metric_name}{label_str} {sample.value}")
        return "\n".join(lines) + "\n"

    def generate_text(self) -> bytes:
        return self.generate_metrics().encode()


class StatsDExporter:
    def __init__(self, host: str = "localhost", port: int = 8125, prefix: str = ""):
        self.host = host
        self.port = port
        self.prefix = prefix
        self._sock = None

    async def export(self, samples: List[MetricSample]):
        import socket
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        for sample in samples:
            name = sample.name
            if self.prefix:
                name = f"{self.prefix}.{name}"
            if sample.labels.labels:
                tags = ",".join(f"{k}:{v}" for k, v in sample.labels.labels)
                line = f"{name}:{sample.value}|{sample.metric_type.value[0]}|#{tags}"
            else:
                line = f"{name}:{sample.value}|{sample.metric_type.value[0]}"

            self._sock.sendto(line.encode(), (self.host, self.port))


class MetricsServer:
    def __init__(
        self,
        registry: MetricsRegistry,
        port: int = 9090,
        path: str = "/metrics",
    ):
        self.registry = registry
        self.port = port
        self.path = path
        self.exporter = PrometheusExporter(registry)
        self._running = False
        self._server = None

    async def start(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get(self.path, self._handle_metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        self._running = True

    async def stop(self):
        self._running = False

    async def _handle_metrics(self, request):
        from aiohttp import web
        return web.Response(
            text=self.exporter.generate_metrics(),
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )


class HealthCheck:
    def __init__(self):
        self._checks: Dict[str, Callable[[], Awaitable[Tuple[bool, str]]]] = {}

    def add_check(self, name: str, check: Callable[[], Awaitable[Tuple[bool, str]]]):
        self.checks[name] = check

    async def run(self) -> Dict[str, Any]:
        results = {}
        overall = True
        for name, check in self.checks.items():
            try:
                healthy, message = await check()
                results[name] = {"healthy": healthy, "message": message}
                if not healthy:
                    overall = False
            except Exception as e:
                results[name] = {"healthy": False, "message": str(e)}
                overall = False
        return {"healthy": overall, "checks": results}


# Global metrics registry
_global_registry: Optional[MetricsRegistry] = None
_global_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    global _global_registry
    with _global_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry()
        return _global_registry


def get_counter(name: str, description: str = "", unit: str = "") -> Counter:
    return get_metrics_registry().counter(name, description, unit)


def get_gauge(name: str, description: str = "", unit: str = "") -> Gauge:
    return get_metrics_registry().gauge(name, description, unit)


def get_histogram(name: str, description: str = "", unit: str = "", buckets: List[float] = None) -> Histogram:
    return get_metrics_registry().histogram(name, description, unit, buckets)


def get_summary(name: str, description: str = "", unit: str = "", quantiles: List[float] = None) -> Summary:
    return get_metrics_registry().summary(name, description, unit, quantiles)


# Built-in metrics
REQUEST_COUNT = get_counter("http_requests_total", "Total HTTP requests", "1")
REQUEST_DURATION = get_histogram("http_request_duration_seconds", "HTTP request duration", "s")
ACTIVE_CONNECTIONS = get_gauge("active_connections", "Number of active connections", "1")
MEMORY_USAGE = get_gauge("process_memory_bytes", "Process memory usage", "bytes")
CPU_USAGE = get_gauge("process_cpu_percent", "Process CPU usage", "percent")
GC_COUNT = get_counter("gc_collections_total", "Total garbage collections", "1")
GC_DURATION = get_histogram("gc_duration_seconds", "GC duration", "s")


class SystemMetricsCollector:
    def __init__(self, registry: MetricsRegistry = None):
        self.registry = registry or get_metrics_registry()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, interval: float = 10.0):
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(interval))

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self, interval: float):
        import psutil
        process = psutil.Process()
        while True:
            try:
                mem = process.memory_info()
                MEMORY_USAGE.set(mem.rss)
                CPU_USAGE.set(process.cpu_percent())
                # GC stats would need gc module
            except Exception as e:
                logger.warning(f"Failed to collect system metrics: {e}")
            await asyncio.sleep(interval)


class SLIManager:
    def __init__(self, registry: MetricsRegistry = None):
        self.registry = registry or get_metrics_registry()
        self.slos: Dict[str, SLO] = {}

    def add_slo(self, slo: "SLO"):
        self.slos[slo.name] = slo

    async def evaluate(self) -> Dict[str, Any]:
        results = {}
        for name, slo in self.slos.items():
            results[name] = await slo.evaluate()
        return results


class SLO:
    def __init__(
        self,
        name: str,
        description: str,
        target: float,
        window_seconds: float,
        good_events_query: Callable[[], int],
        total_events_query: Callable[[], int],
    ):
        self.name = name
        self.description = description
        self.target = target
        self.window_seconds = window_seconds
        self.good_events_query = good_events_query
        self.total_events_query = total_events_query

    async def evaluate(self) -> Dict[str, Any]:
        good = self.good_events_query()
        total = self.total_events_query()
        if total == 0:
            return {"compliance": 1.0, "target": self.target, "error_budget_remaining": 1.0}
        compliance = good / total
        error_budget = 1 - self.target
        error_budget_remaining = (compliance - self.target) / error_budget if error_budget > 0 else 1.0
        return {
            "compliance": compliance,
            "target": self.target,
            "error_budget_remaining": max(0, error_budget_remaining),
            "good_events": good,
            "total_events": total,
        }


def create_metrics_stack(
    port: int = 9090,
    path: str = "/metrics",
    collect_system: bool = True,
) -> Tuple[MetricsRegistry, MetricsServer, Optional[SystemMetricsCollector]]:
    registry = MetricsRegistry()
    server = MetricsServer(registry, port=port)
    collector = SystemMetricsCollector() if collect_system else None
    return registry, server, collector