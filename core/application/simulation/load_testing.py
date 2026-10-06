from __future__ import annotations

import numpy as np
import time
import threading
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import random
import json
import uuid


class LoadPattern(Enum):
    CONSTANT = "constant"
    STEP = "step"
    RAMP = "ramp"
    SPIKE = "spike"
    SINE = "sine"
    RANDOM = "random"
    BURST = "burst"


class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_COUNT = "connection_count"


@dataclass(frozen=True, slots=True)
class LoadConfig:
    pattern: LoadPattern
    duration_seconds: float
    target_rps: float
    ramp_up_seconds: float = 0.0
    ramp_down_seconds: float = 0.0
    spike_multiplier: float = 1.0
    spike_interval: float = 0.0
    burst_size: int = 0
    burst_interval: float = 0.0


@dataclass(frozen=True, slots=True)
class RequestResult:
    request_id: str
    timestamp: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
    response_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadTestResult:
    test_id: str
    config: LoadConfig
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    latency_stats: Dict[str, float]
    throughput_stats: Dict[str, float]
    error_stats: Dict[str, float]
    percentile_latencies: Dict[str, float]
    time_series: List[Dict[str, float]]


class LoadGenerator:
    def __init__(self, config: LoadConfig):
        self.config = config
        self.running = False
        self.results: List[RequestResult] = []
        self.lock = threading.Lock()
        self.current_rps = 0.0

    def generate_rps(self, elapsed: float) -> float:
        if self.config.pattern == LoadPattern.CONSTANT:
            return self.config.target_rps
        elif self.config.pattern == LoadPattern.STEP:
            if elapsed < self.config.ramp_up_seconds:
                return self.config.target_rps * (elapsed / self.config.ramp_up_seconds)
            return self.config.target_rps
        elif self.config.pattern == LoadPattern.RAMP:
            if elapsed < self.config.ramp_up_seconds:
                return self.config.target_rps * (elapsed / self.config.ramp_up_seconds)
            elif elapsed > self.config.duration_seconds - self.config.ramp_down_seconds:
                return self.config.target_rps * (1 - (elapsed - (self.config.duration_seconds - self.config.ramp_down_seconds)) / self.config.ramp_down_seconds)
            return self.config.target_rps
        elif self.config.pattern == LoadPattern.SPIKE:
            base = self.config.target_rps
            if self.config.spike_interval > 0:
                spike_phase = (elapsed % self.config.spike_interval) / self.config.spike_interval
                if spike_phase < 0.1:
                    return base * self.config.spike_multiplier
            return base
        elif self.config.pattern == LoadPattern.BURST:
            if self.config.burst_interval > 0:
                burst_phase = (elapsed % self.config.burst_interval) / self.config.burst_interval
                if burst_phase < 0.05:
                    return self.config.target_rps * self.config.burst_size
            return self.config.target_rps
        elif self.config.pattern == LoadPattern.SINE:
            return self.config.target_rps * (1 + 0.5 * math.sin(2 * math.pi * elapsed / 60))
        elif self.config.pattern == LoadPattern.RANDOM:
            return self.config.target_rps * (0.5 + random.random())
        return self.config.target_rps

    def run(self, request_func: Callable[[], RequestResult]) -> LoadTestResult:
        self.running = True
        self.results = []
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        request_id = 0
        next_request_time = start_time

        while time.time() < end_time and self.running:
            current_time = time.time()
            elapsed = current_time - start_time

            rps = self.generate_rps(elapsed)
            self.current_rps = rps
            interval = 1.0 / rps if rps > 0 else 1.0

            if current_time >= next_request_time:
                request_id += 1
                try:
                    result = request_func()
                    result = RequestResult(
                        request_id=str(request_id),
                        timestamp=current_time,
                        latency_ms=result.latency_ms,
                        success=result.success,
                        error=result.error,
                        response_size=result.response_size,
                        metadata=result.metadata,
                    )
                    with self.lock:
                        self.results.append(result)
                except Exception as e:
                    with self.lock:
                        self.results.append(RequestResult(
                            request_id=str(request_id),
                            timestamp=current_time,
                            latency_ms=0,
                            success=False,
                            error=str(e),
                        ))
                next_request_time = current_time + interval
            else:
                sleep_time = min(0.001, next_request_time - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        actual_end = time.time()
        return self._compute_results(start_time, actual_end)

    def _compute_results(self, start_time: float, end_time: float) -> LoadTestResult:
        with self.lock:
            results = self.results[:]

        if not results:
            return LoadTestResult(
                test_id=str(uuid.uuid4()),
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                latency_stats={},
                throughput_stats={},
                error_stats={},
                percentile_latencies={},
                time_series=[],
            )

        latencies = [r.latency_ms for r in results if r.success]
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        latency_stats = {
            'mean': statistics.mean(latencies) if latencies else 0,
            'median': statistics.median(latencies) if latencies else 0,
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min': min(latencies) if latencies else 0,
            'max': max(latencies) if latencies else 0,
        }

        percentiles = {}
        if latencies:
            sorted_lat = sorted(latencies)
            for p in [50, 75, 90, 95, 99, 99.9]:
                idx = int(len(sorted_lat) * p / 100)
                percentiles[f'p{p}'] = sorted_lat[min(idx, len(sorted_lat) - 1)]

        duration = end_time - start_time
        throughput_stats = {
            'total_rps': len(results) / duration,
            'success_rps': successful / duration,
            'target_rps': self.config.target_rps,
        }

        error_stats = {
            'error_rate': failed / len(results) if results else 0,
            'total_errors': failed,
            'error_types': self._count_error_types(results),
        }

        time_series = self._generate_time_series(results, start_time, duration)

        return LoadTestResult(
            test_id=str(uuid.uuid4()),
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            error_stats=error_stats,
            percentile_latencies=percentiles,
            time_series=time_series,
        )

    def _count_error_types(self, results: List[RequestResult]) -> Dict[str, int]:
        errors = defaultdict(int)
        for r in results:
            if not r.success and r.error:
                errors[r.error] += 1
        return dict(errors)

    def _generate_time_series(self, results: List[RequestResult], start_time: float, duration: float, bucket_size: float = 1.0) -> List[Dict[str, float]]:
        num_buckets = int(duration / bucket_size) + 1
        buckets = {i: {'latencies': [], 'success': 0, 'total': 0} for i in range(num_buckets)}

        for r in results:
            bucket_idx = int((r.timestamp - start_time) / bucket_size)
            if 0 <= bucket_idx < num_buckets:
                buckets[bucket_idx]['total'] += 1
                if r.success:
                    buckets[bucket_idx]['success'] += 1
                    buckets[bucket_idx]['latencies'].append(r.latency_ms)

        series = []
        for i in range(num_buckets):
            bucket = buckets[i]
            series.append({
                'time': start_time + i * bucket_size,
                'rps': bucket['total'] / bucket_size,
                'success_rate': bucket['success'] / bucket['total'] if bucket['total'] > 0 else 0,
                'mean_latency': statistics.mean(bucket['latencies']) if bucket['latencies'] else 0,
            })
        return series


class AsyncLoadGenerator:
    def __init__(self, config: LoadConfig):
        self.config = config
        self.running = False
        self.results: List[RequestResult] = []
        self.lock = asyncio.Lock()

    async def run(self, request_func: Callable[[], asyncio.Future]) -> LoadTestResult:
        self.running = True
        self.results = []
        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        request_id = 0
        next_request_time = start_time

        while time.time() < end_time and self.running:
            current_time = time.time()
            elapsed = current_time - start_time

            rps = self._generate_rps(elapsed)
            interval = 1.0 / rps if rps > 0 else 1.0

            if current_time >= next_request_time:
                request_id += 1
                try:
                    result = await request_func()
                    async with self.lock:
                        self.results.append(RequestResult(
                            request_id=str(request_id),
                            timestamp=current_time,
                            latency_ms=result.latency_ms,
                            success=result.success,
                            error=result.error,
                            response_size=result.response_size,
                            metadata=result.metadata,
                        ))
                except Exception as e:
                    async with self.lock:
                        self.results.append(RequestResult(
                            request_id=str(request_id),
                            timestamp=current_time,
                            latency_ms=0,
                            success=False,
                            error=str(e),
                        ))
                next_request_time = current_time + interval
            else:
                await asyncio.sleep(min(0.001, next_request_time - current_time))

        actual_end = time.time()
        return self._compute_results(start_time, actual_end)

    def _generate_rps(self, elapsed: float) -> float:
        if self.config.pattern == LoadPattern.CONSTANT:
            return self.config.target_rps
        elif self.config.pattern == LoadPattern.RAMP:
            if elapsed < self.config.ramp_up_seconds:
                return self.config.target_rps * (elapsed / self.config.ramp_up_seconds)
            return self.config.target_rps
        return self.config.target_rps

    def _compute_results(self, start_time: float, end_time: float) -> LoadTestResult:
        results = self.results
        if not results:
            return LoadTestResult(
                test_id=str(uuid.uuid4()),
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                latency_stats={},
                throughput_stats={},
                error_stats={},
                percentile_latencies={},
                time_series=[],
            )

        latencies = [r.latency_ms for r in results if r.success]
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        latency_stats = {
            'mean': statistics.mean(latencies) if latencies else 0,
            'median': statistics.median(latencies) if latencies else 0,
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min': min(latencies) if latencies else 0,
            'max': max(latencies) if latencies else 0,
        }

        percentiles = {}
        if latencies:
            sorted_lat = sorted(latencies)
            for p in [50, 75, 90, 95, 99, 99.9]:
                idx = int(len(sorted_lat) * p / 100)
                percentiles[f'p{p}'] = sorted_lat[min(idx, len(sorted_lat) - 1)]

        duration = end_time - start_time
        throughput_stats = {
            'total_rps': len(results) / duration,
            'success_rps': successful / duration,
            'target_rps': self.config.target_rps,
        }

        return LoadTestResult(
            test_id=str(uuid.uuid4()),
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            latency_stats=latency_stats,
            throughput_stats=throughput_stats,
            error_stats={'error_rate': failed / len(results), 'total_errors': failed},
            percentile_latencies=percentiles,
            time_series=[],
        )


class ChaosInjector:
    def __init__(self):
        self.faults: Dict[str, Dict] = {}
        self.active = False

    def add_latency_fault(self, name: str, latency_ms: float, probability: float = 1.0):
        self.faults[name] = {
            'type': 'latency',
            'latency_ms': latency_ms,
            'probability': probability,
        }

    def add_error_fault(self, name: str, error_message: str, probability: float = 1.0):
        self.faults[name] = {
            'type': 'error',
            'error_message': error_message,
            'probability': probability,
        }

    def add_timeout_fault(self, name: str, timeout_ms: float, probability: float = 1.0):
        self.faults[name] = {
            'type': 'timeout',
            'timeout_ms': timeout_ms,
            'probability': probability,
        }

    def add_disconnect_fault(self, name: str, duration_ms: float, probability: float = 1.0):
        self.faults[name] = {
            'type': 'disconnect',
            'duration_ms': duration_ms,
            'probability': probability,
        }

    def inject(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for fault in self.faults.values():
                if random.random() < fault['probability']:
                    if fault['type'] == 'latency':
                        time.sleep(fault['latency_ms'] / 1000.0)
                    elif fault['type'] == 'error':
                        raise Exception(fault['error_message'])
                    elif fault['type'] == 'timeout':
                        time.sleep(fault['timeout_ms'] / 1000.0)
                        raise TimeoutError("Injected timeout")
                    elif fault['type'] == 'disconnect':
                        time.sleep(fault['duration_ms'] / 1000.0)
                        raise ConnectionError("Injected disconnect")
            return func(*args, **kwargs)
        return wrapper


class LoadTestSuite:
    def __init__(self):
        self.tests: List[Tuple[str, LoadGenerator, Callable]] = []

    def add_test(self, name: str, config: LoadConfig, request_func: Callable):
        generator = LoadGenerator(config)
        self.tests.append((name, generator, request_func))

    def run_all(self) -> Dict[str, LoadTestResult]:
        results = {}
        for name, generator, request_func in self.tests:
            print(f"Running test: {name}")
            result = generator.run(request_func)
            results[name] = result
            print(f"  Completed: {result.total_requests} requests, {result.successful_requests} successful, {result.failed_requests} failed")
        return results

    def generate_report(self, results: Dict[str, LoadTestResult]) -> str:
        report = ["Load Test Report", "=" * 50, ""]
        for name, result in results.items():
            report.append(f"Test: {name}")
            report.append(f"  Duration: {result.end_time - result.start_time:.2f}s")
            report.append(f"  Total Requests: {result.total_requests}")
            report.append(f"  Successful: {result.successful_requests}")
            report.append(f"  Failed: {result.failed_requests}")
            report.append(f"  Error Rate: {result.error_stats.get('error_rate', 0):.2%}")
            report.append(f"  Mean Latency: {result.latency_stats.get('mean', 0):.2f}ms")
            report.append(f"  P50: {result.percentile_latencies.get('p50', 0):.2f}ms")
            report.append(f"  P95: {result.percentile_latencies.get('p95', 0):.2f}ms")
            report.append(f"  P99: {result.percentile_latencies.get('p99', 0):.2f}ms")
            report.append(f"  Throughput: {result.throughput_stats.get('total_rps', 0):.2f} rps")
            report.append("")
        return "\n".join(report)


class PerformanceBenchmark:
    def __init__(self):
        self.benchmarks: Dict[str, Dict] = {}

    def benchmark_function(self, name: str, func: Callable, iterations: int = 1000, *args, **kwargs) -> Dict:
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        return {
            'name': name,
            'iterations': iterations,
            'mean_ms': statistics.mean(times_ms),
            'median_ms': statistics.median(times_ms),
            'stdev_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            'min_ms': min(times_ms),
            'max_ms': max(times_ms),
            'p95_ms': sorted(times_ms)[int(len(times_ms) * 0.95)],
            'p99_ms': sorted(times_ms)[int(len(times_ms) * 0.99)],
            'ops_per_sec': 1000 / statistics.mean(times_ms) if statistics.mean(times_ms) > 0 else 0,
        }

    def compare(self, name: str, functions: Dict[str, Callable], iterations: int = 1000, *args, **kwargs) -> Dict:
        results = {}
        for name, func in functions.items():
            results[name] = self.benchmark_function(name, func, iterations, *args, **kwargs)
        return {
            'comparison': name,
            'results': results,
            'fastest': min(results.items(), key=lambda x: x[1]['mean_ms'])[0] if results else None,
        }

    def profile_memory(self, func: Callable, *args, **kwargs) -> Dict:
        import tracemalloc
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'result': result,
        }


class SoakTest:
    def __init__(self, config: LoadConfig):
        self.config = config
        self.running = False
        self.results: List[LoadTestResult] = []

    def run(self, request_func: Callable, num_iterations: int = 10) -> List[LoadTestResult]:
        self.results = []
        for i in range(num_iterations):
            print(f"Starting soak iteration {i+1}/{num_iterations}")
            generator = LoadGenerator(self.config)
            result = generator.run(request_func)
            self.results.append(result)
            print(f"  Iteration {i+1}: {result.total_requests} requests, {result.error_stats.get('error_rate', 0):.2%} error rate")
        return self.results

    def analyze_degradation(self) -> Dict:
        if len(self.results) < 2:
            return {}

        first = self.results[0]
        last = self.results[-1]

        latency_degradation = (last.latency_stats.get('mean', 0) - first.latency_stats.get('mean', 0)) / max(first.latency_stats.get('mean', 1), 1)
        throughput_degradation = (last.throughput_stats.get('success_rps', 0) - first.throughput_stats.get('success_rps', 0)) / max(first.throughput_stats.get('success_rps', 1), 1)
        error_rate_change = last.error_stats.get('error_rate', 0) - first.error_stats.get('error_rate', 0)

        return {
            'latency_degradation_pct': latency_degradation * 100,
            'throughput_degradation_pct': throughput_degradation * 100,
            'error_rate_change_pct': error_rate_change * 100,
            'degraded': latency_degradation > 0.1 or throughput_degradation < -0.1 or error_rate_change > 0.01,
        }