from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import queue
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: str = ""


@dataclass(frozen=True, slots=True)
class Alert:
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class MetricCollector:
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lock = threading.RLock()

    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self.metrics[key].append(Metric(
                name=name,
                metric_type=MetricType.COUNTER,
                value=self.counters[key],
                labels=labels or {},
                timestamp=time.time()
            ))

    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self.metrics[key].append(Metric(
                name=name,
                metric_type=MetricType.GAUGE,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            ))

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            if len(self.histograms[key]) > self.max_samples:
                self.histograms[key] = self.histograms[key][-self.max_samples:]

            self.metrics[key].append(Metric(
                name=name,
                metric_type=MetricType.HISTOGRAM,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            ))

    def record_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)

            if len(self.histograms[key]) >= 2:
                vals = self.histograms[key]
                self.summaries[key] = {
                    'count': len(vals),
                    'sum': sum(vals),
                    'min': min(vals),
                    'max': max(vals),
                    'mean': statistics.mean(vals),
                    'std': statistics.stdev(vals) if len(vals) > 1 else 0,
                    'p50': self._percentile(vals, 50),
                    'p90': self._percentile(vals, 90),
                    'p95': self._percentile(vals, 95),
                    'p99': self._percentile(vals, 99)
                }

            self.metrics[key].append(Metric(
                name=name,
                metric_type=MetricType.SUMMARY,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            ))

    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        self.record_histogram(name, duration_ms, labels)

    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        with self.lock:
            key = self._make_key(name, labels)
            return self.counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        with self.lock:
            key = self._make_key(name, labels)
            return self.gauges.get(key)

    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        with self.lock:
            key = self._make_key(name, labels)
            values = self.histograms.get(key, [])

            if not values:
                return {}

            return {
                'count': len(values),
                'sum': sum(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'p50': self._percentile(values, 50),
                'p90': self._percentile(values, 90),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }

    def get_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        with self.lock:
            key = self._make_key(name, labels)
            return self.summaries.get(key, {})

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _percentile(self, values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        with self.lock:
            return {k: list(v) for k, v in self.metrics.items()}


class AlertManager:
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.rules: List[AlertRule] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()

    def add_rule(self, rule: 'AlertRule'):
        with self.lock:
            self.rules.append(rule)

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        with self.lock:
            self.notification_handlers.append(handler)

    def fire_alert(self, alert: Alert):
        with self.lock:
            self.alerts[alert.alert_id] = alert

            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception:
                    pass

    def resolve_alert(self, alert_id: str):
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = time.time()

    def get_active_alerts(self) -> List[Alert]:
        with self.lock:
            return [a for a in self.alerts.values() if not a.resolved]

    def get_all_alerts(self) -> List[Alert]:
        with self.lock:
            return list(self.alerts.values())

    def evaluate_rules(self, metrics: Dict[str, float]):
        with self.lock:
            for rule in self.rules:
                if rule.condition(metrics):
                    alert = Alert(
                        alert_id=f"{rule.name}_{int(time.time())}",
                        name=rule.name,
                        level=rule.level,
                        message=rule.message,
                        source=rule.source,
                        labels=rule.labels
                    )
                    self.fire_alert(alert)


class AlertRule:
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, float]], bool],
        level: AlertLevel,
        message: str,
        source: str = "monitoring",
        labels: Dict[str, str] = None
    ):
        self.name = name
        self.condition = condition
        self.level = level
        self.message = message
        self.source = source
        self.labels = labels or {}


class HealthChecker:
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.lock = threading.RLock()

    def register_check(self, name: str, check_fn: Callable[[], HealthCheck]):
        with self.lock:
            self.checks[name] = check_fn

    def run_check(self, name: str) -> HealthCheck:
        with self.lock:
            check_fn = self.checks.get(name)
            if not check_fn:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Check not found"
                )

            result = check_fn()
            self.last_results[name] = result
            return result

    def run_all_checks(self) -> Dict[str, HealthCheck]:
        with self.lock:
            results = {}
            for name, check_fn in self.checks.items():
                try:
                    result = check_fn()
                    results[name] = result
                except Exception as e:
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(e)}"
                    )
            self.last_results = results
            return results

    def get_overall_health(self) -> HealthStatus:
        with self.lock:
            if not self.last_results:
                return HealthStatus.UNKNOWN

            statuses = [r.status for r in self.last_results.values()]

            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            elif all(s == HealthStatus.HEALTHY for s in statuses):
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNKNOWN


class MetricsExporter:
    def __init__(self, collector: MetricCollector):
        self.collector = collector

    def export_prometheus(self) -> str:
        lines = []

        for key, metrics in self.collector.metrics.items():
            if not metrics:
                continue

            metric = metrics[-1]
            name = metric.name.replace('.', '_').replace('-', '_')

            if metric.metric_type == MetricType.COUNTER:
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {metric.value}")
            elif metric.metric_type == MetricType.GAUGE:
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {metric.value}")
            elif metric.metric_type == MetricType.HISTOGRAM:
                stats = self.collector.get_histogram_stats(metric.name, metric.labels)
                if stats:
                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f"{name}_count {stats['count']}")
                    lines.append(f"{name}_sum {stats['sum']}")
                    for p in [50, 90, 95, 99]:
                        lines.append(f"{name}_bucket{{le=\"{p}\"}} {stats.get(f'p{p}', 0)}")
            elif metric.metric_type == MetricType.SUMMARY:
                summary = self.collector.get_summary(metric.name, metric.labels)
                if summary:
                    lines.append(f"# TYPE {name} summary")
                    lines.append(f"{name}_count {summary.get('count', 0)}")
                    lines.append(f"{name}_sum {summary.get('sum', 0)}")

        return "\n".join(lines)

    def export_json(self) -> str:
        data = {}
        for key, metrics in self.collector.metrics.items():
            if not metrics:
                continue
            metric = metrics[-1]
            data[key] = {
                'name': metric.name,
                'type': metric.metric_type.value,
                'value': metric.value,
                'labels': metric.labels,
                'timestamp': metric.timestamp
            }
        return json.dumps(data, indent=2)

    def export_influxdb(self) -> str:
        lines = []

        for key, metrics in self.collector.metrics.items():
            if not metrics:
                continue
            metric = metrics[-1]

            tags = ",".join(f"{k}={v}" for k, v in metric.labels.items())
            tag_str = f",{tags}" if tags else ""

            if metric.metric_type == MetricType.COUNTER:
                lines.append(f"{metric.name}{tag_str} value={metric.value} {int(metric.timestamp * 1e9)}")
            elif metric.metric_type == MetricType.GAUGE:
                lines.append(f"{metric.name}{tag_str} value={metric.value} {int(metric.timestamp * 1e9)}")
            elif metric.metric_type == MetricType.HISTOGRAM:
                stats = self._get_histogram_stats(metric.name, metric.labels)
                if stats:
                    lines.append(f"{metric.name}{tag_str} count={stats['count']},sum={stats['sum']},mean={stats['mean']},std={stats['std']} {int(metric.timestamp * 1e9)}")

        return "\n".join(lines)

    def _get_histogram_stats(self, name: str, labels: Dict[str, str]) -> Dict[str, float]:
        key = name
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            key = f"{name}{{{label_str}}}"

        values = []
        return {}


class MonitoringDashboard:
    def __init__(
        self,
        collector: MetricCollector,
        alert_manager: AlertManager,
        health_checker: HealthChecker
    ):
        self.collector = collector
        self.alert_manager = alert_manager
        self.health_checker = health_checker

    def get_dashboard_data(self) -> Dict[str, Any]:
        overall_health = self.health_checker.get_overall_health()
        active_alerts = self.alert_manager.get_active_alerts()

        metrics_summary = {}
        for key, metrics in self.collector.metrics.items():
            if not metrics:
                continue
            metric = metrics[-1]
            metrics_summary[metric.name] = {
                'type': metric.metric_type.value,
                'value': metric.value,
                'labels': metric.labels,
                'timestamp': metric.timestamp
            }

        return {
            'overall_health': overall_health.value,
            'active_alerts': len(active_alerts),
            'alerts': [
                {
                    'id': a.alert_id,
                    'name': a.name,
                    'level': a.level.value,
                    'message': a.message,
                    'source': a.source,
                    'timestamp': a.timestamp
                } for a in active_alerts
            ],
            'metrics': metrics_summary,
            'timestamp': time.time()
        }

    def get_metric_history(self, name: str, labels: Dict[str, str] = None, limit: int = 100) -> List[Dict]:
        key = self.collector._make_key(name, labels)
        metrics = self.collector.metrics.get(key, [])

        return [
            {
                'timestamp': m.timestamp,
                'value': m.value,
                'labels': m.labels
            } for m in list(metrics)[-limit:]
        ]


class MonitoringServer:
    def __init__(
        self,
        collector: MetricCollector,
        alert_manager: AlertManager,
        health_checker: HealthChecker,
        port: int = 9090
    ):
        self.collector = collector
        self.alert_manager = alert_manager
        self.health_checker = health_checker
        self.port = port
        self.dashboard = MonitoringDashboard(collector, alert_manager, health_checker)
        self.exporter = MetricsExporter(collector)

    def start(self):
        import threading
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()

    def _run_server(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json

        collector = self.collector
        alert_manager = self.alert_manager
        health_checker = self.health_checker
        exporter = self.exporter

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(exporter.export_prometheus().encode())
                elif self.path == "/metrics/json":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(exporter.export_json().encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    health = health_checker.get_overall_health()
                    checks = health_checker.run_all_checks()
                    self.wfile.write(json.dumps({
                        'status': health.value,
                        'checks': {k: {
                            'status': v.status.value,
                            'message': v.message,
                            'timestamp': v.timestamp
                        } for k, v in checks.items()}
                    }).encode())
                elif self.path == "/dashboard":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    dashboard = MonitoringDashboard(collector, alert_manager, health_checker)
                    self.wfile.write(json.dumps(dashboard.get_dashboard_data()).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer(("0.0.0.0", self.port), Handler)
        server.serve_forever()


class MonitoringSystem:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.collector = MetricCollector(
            max_samples=self.config.get('max_samples', 10000)
        )
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()

        self.dashboard = MonitoringDashboard(
            self.collector, self.alert_manager, self.health_checker
        )
        self.exporter = MetricsExporter(self.collector)
        self.server = MonitoringServer(
            self.collector, self.alert_manager, self.health_checker,
            self.config.get('port', 9090)
        )

        self._setup_default_checks()
        self._setup_default_alerts()

    def _setup_default_checks(self):
        self.health_checker.register_check("system", self._check_system)
        self.health_checker.register_check("memory", self._check_memory)
        self.health_checker.register_check("disk", self._check_disk)
        self.health_checker.register_check("cpu", self._check_cpu)

    def _setup_default_alerts(self):
        self.alert_manager.add_rule(AlertRule(
            name="high_cpu",
            condition=lambda m: m.get("cpu_usage", 0) > 90,
            level=AlertLevel.WARNING,
            message="CPU usage above 90%",
            source="system"
        ))

        self.alert_manager.add_rule(AlertRule(
            name="high_memory",
            condition=lambda m: m.get("memory_usage", 0) > 90,
            level=AlertLevel.WARNING,
            message="Memory usage above 90%",
            source="system"
        ))

        self.alert_manager.add_rule(AlertRule(
            name="high_disk",
            condition=lambda m: m.get("disk_usage", 0) > 90,
            level=AlertLevel.CRITICAL,
            message="Disk usage above 90%",
            source="system"
        ))

    def _check_system(self) -> HealthCheck:
        return HealthCheck(
            name="system",
            status=HealthStatus.HEALTHY,
            message="System operational"
        )

    def _check_memory(self) -> HealthCheck:
        import psutil
        mem = psutil.virtual_memory()
        status = HealthStatus.HEALTHY
        if mem.percent > 90:
            status = HealthStatus.CRITICAL
        elif mem.percent > 75:
            status = HealthStatus.DEGRADED

        return HealthCheck(
            name="memory",
            status=status,
            message=f"Memory usage: {mem.percent}%"
        )

    def _check_disk(self) -> HealthCheck:
        import psutil
        disk = psutil.disk_usage('/')
        percent = disk.used / disk.total * 100
        status = HealthStatus.HEALTHY
        if percent > 90:
            status = HealthStatus.CRITICAL
        elif percent > 80:
            status = HealthStatus.DEGRADED

        return HealthCheck(
            name="disk",
            status=status,
            message=f"Disk usage: {percent:.1f}%"
        )

    def _check_cpu(self) -> HealthCheck:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        status = HealthStatus.HEALTHY
        if cpu > 90:
            status = HealthStatus.CRITICAL
        elif cpu > 75:
            status = HealthStatus.DEGRADED

        return HealthCheck(
            name="cpu",
            status=status,
            message=f"CPU usage: {cpu}%"
        )

    def start(self):
        self.server.start()

    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, labels: Dict = None):
        if metric_type == MetricType.COUNTER:
            self.collector.record_counter(name, value, labels)
        elif metric_type == MetricType.GAUGE:
            self.collector.record_gauge(name, value, labels)
        elif metric_type == MetricType.HISTOGRAM:
            self.collector.record_histogram(name, value, labels)
        elif metric_type == MetricType.SUMMARY:
            self.collector.record_summary(name, value, labels)
        elif metric_type == MetricType.TIMER:
            self.collector.record_timer(name, value, labels)

    def get_dashboard_data(self) -> Dict[str, Any]:
        return self.dashboard.get_dashboard_data()

    def get_metric_history(self, name: str, labels: Dict = None, limit: int = 100) -> List[Dict]:
        return self.dashboard.get_metric_history(name, labels, limit)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
