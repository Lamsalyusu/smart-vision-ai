from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict, deque
import hashlib
import random


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class Backend:
    backend_id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class LoadBalancer:
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        health_check_interval: float = 10.0,
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.backends: Dict[str, Backend] = {}
        self.connections: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.round_robin_index = 0
        self.consistent_hash_ring: Dict[int, str] = {}
        self.virtual_nodes = 150
        self.lock = threading.RLock()

    def add_backend(self, backend: Backend):
        with self.lock:
            self.backends[backend.backend_id] = backend
            self._rebuild_hash_ring()

    def remove_backend(self, backend_id: str):
        with self.lock:
            self.backends.pop(backend_id, None)
            self.connections.pop(backend_id, None)
            self.response_times.pop(backend_id, None)
            self._rebuild_hash_ring()

    def get_backend(self, client_ip: str = None) -> Optional[Backend]:
        with self.lock:
            healthy_backends = [b for b in self.backends.values() if b.healthy]
            if not healthy_backends:
                return None

            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash(healthy_backends, client_ip)
            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return self._consistent_hash(healthy_backends, client_ip)

            return healthy_backends[0]

    def _round_robin(self, backends: List[Backend]) -> Backend:
        backend = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return backend

    def _weighted_round_robin(self, backends: List[Backend]) -> Backend:
        total_weight = sum(b.weight for b in backends)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for b in backends:
            cumulative += b.weight
            if r <= cumulative:
                return b
        return backends[-1]

    def _least_connections(self, backends: List[Backend]) -> Backend:
        return min(backends, key=lambda b: self.connections[b.backend_id])

    def _least_response_time(self, backends: List[Backend]) -> Backend:
        def avg_response_time(b: Backend) -> float:
            times = self.response_times[b.backend_id]
            return sum(times) / len(times) if times else float('inf')
        return min(backends, key=avg_response_time)

    def _ip_hash(self, backends: List[Backend], client_ip: str) -> Backend:
        if not client_ip:
            return backends[0]
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return backends[hash_val % len(backends)]

    def _consistent_hash(self, backends: List[Backend], client_ip: str) -> Backend:
        if not client_ip or not self.consistent_hash_ring:
            return backends[0]
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        for ring_hash in sorted(self.consistent_hash_ring.keys()):
            if ring_hash >= hash_val:
                backend_id = self.consistent_hash_ring[ring_hash]
                return next((b for b in backends if b.backend_id == backend_id), backends[0])
        first_hash = min(self.consistent_hash_ring.keys())
        backend_id = self.consistent_hash_ring[first_hash]
        return next((b for b in backends if b.backend_id == backend_id), backends[0])

    def _rebuild_hash_ring(self):
        self.consistent_hash_ring.clear()
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            for i in range(self.virtual_nodes):
                key = f"{backend.backend_id}:{i}"
                hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
                self.consistent_hash_ring[hash_val] = backend.backend_id

    def record_connection(self, backend_id: str, delta: int = 1):
        with self.lock:
            self.connections[backend_id] += delta

    def record_response_time(self, backend_id: str, response_time_ms: float):
        with self.lock:
            self.response_times[backend_id].append(response_time_ms)

    def health_check(self, backend_id: str, healthy: bool):
        with self.lock:
            if backend_id in self.backends:
                self.backends[backend_id].healthy = healthy
                if not healthy:
                    self._rebuild_hash_ring()


class AsyncLoadBalancer:
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
    ):
        self.strategy = strategy
        self.backends: Dict[str, Backend] = {}
        self.connections: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = asyncio.Lock()

    async def add_backend(self, backend: Backend):
        async with self.lock:
            self.backends[backend.backend_id] = backend

    async def remove_backend(self, backend_id: str):
        async with self.lock:
            self.backends.pop(backend_id, None)
            self.connections.pop(backend_id, None)
            self.response_times.pop(backend_id, None)

    async def get_backend(self, client_ip: str = None) -> Optional[Backend]:
        async with self.lock:
            healthy_backends = [b for b in self.backends.values() if b.healthy]
            if not healthy_backends:
                return None

            if self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_backends, key=lambda b: self.connections[b.backend_id])
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                def avg_rt(b):
                    times = self.response_times[b.backend_id]
                    return sum(times) / len(times) if times else float('inf')
                return min(healthy_backends, key=avg_rt)
            elif self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return healthy_backends[0]

            return healthy_backends[0]

    async def record_connection(self, backend_id: str, delta: int = 1):
        async with self.lock:
            self.connections[backend_id] += delta

    async def record_response_time(self, backend_id: str, rt: float):
        async with self.lock:
            self.response_times[backend_id].append(rt)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AdvancedCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        expected_exception: type = Exception,
        fallback: Optional[Callable] = None,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.fallback = fallback

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()

    def call(self, func: Callable, *args, **kwargs):
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    if self.fallback:
                        return self.fallback(*args, **kwargs)
                    raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            if self.fallback:
                return self.fallback(*args, **kwargs)
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

    def get_state(self) -> CircuitBreakerState:
        with self.lock:
            return self.state

    def reset(self):
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0


class AsyncAdvancedCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        expected_exception: type = Exception,
        fallback: Optional[Callable] = None,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.fallback = fallback

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = asyncio.Lock()

    async def call(self, coro):
        async with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    if self.fallback:
                        return await self.fallback()
                    raise Exception("Circuit breaker is open")

        try:
            result = await coro
            await self._on_success()
            return result
        except self.expected_exception:
            await self._on_failure()
            if self.fallback:
                return await self.fallback()
            raise

    async def _on_success(self):
        async with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    async def _on_failure(self):
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[type, ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def execute(self, func: Callable, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    if self.jitter:
                        delay *= (0.5 + random.random())
                    time.sleep(delay)

        raise last_exception

    async def execute_async(self, coro_func: Callable, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await coro_func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    if self.jitter:
                        delay *= (0.5 + random.random())
                    await asyncio.sleep(delay)

        raise last_exception


class Bulkhead:
    def __init__(self, max_concurrent: int = 10, max_queue: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.semaphore = threading.Semaphore(max_concurrent)
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.active = 0
        self.lock = threading.Lock()

    def execute(self, func: Callable, *args, **kwargs):
        if not self.semaphore.acquire(blocking=False):
            raise Exception("Bulkhead full")

        with self.lock:
            self.active += 1

        try:
            return func(*args, **kwargs)
        finally:
            with self.lock:
                self.active -= 1
            self.semaphore.release()

    async def execute_async(self, coro_func: Callable, *args, **kwargs):
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with semaphore:
            return await coro_func(*args, **kwargs)


class TimeoutManager:
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout

    def execute_with_timeout(self, func: Callable, timeout: float = None, *args, **kwargs):
        timeout = timeout or self.default_timeout
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Operation timed out after {timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    async def execute_async_with_timeout(self, coro, timeout: float = None):
        timeout = timeout or self.default_timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout}s")


class ResourcePool(Generic[T]):
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        validate: Callable[[T], bool] = None,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.validate = validate or (lambda x: True)

        self.available: queue.Queue = queue.Queue()
        self.in_use: Set[T] = set()
        self.all_resources: List[T] = []
        self.lock = threading.RLock()

        for _ in range(min_size):
            resource = factory()
            self.available.put(resource)
            self.all_resources.append(resource)

    def acquire(self, timeout: float = None) -> T:
        start = time.time()
        while True:
            try:
                resource = self.available.get_nowait()
                if self.validate(resource):
                    with self.lock:
                        self.in_use.add(resource)
                    return resource
                else:
                    self._replace_resource(resource)
            except queue.Empty:
                with self.lock:
                    if len(self.all_resources) < self.max_size:
                        resource = self.factory()
                        self.in_use.add(resource)
                        self.all_resources.append(resource)
                        return resource

            if timeout and time.time() - start > timeout:
                raise TimeoutError("Resource pool exhausted")

            time.sleep(0.01)

    def release(self, resource: T):
        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                if self.validate(resource):
                    self.available.put(resource)
                else:
                    self._replace_resource(resource)

    def _replace_resource(self, resource: T):
        with self.lock:
            if resource in self.all_resources:
                self.all_resources.remove(resource)
            try:
                new_resource = self.factory()
                self.all_resources.append(new_resource)
                self.available.put(new_resource)
            except:
                pass

    def close_all(self):
        with self.lock:
            for resource in self.all_resources:
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                except:
                    pass
            self.all_resources.clear()
            while not self.available.empty():
                try:
                    self.available.get_nowait()
                except:
                    pass
