from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import hashlib
import random
import heapq
import socket
import selectors
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio
import logging
import warnings

class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_BANDWIDTH = "least_bandwidth"
    ADAPTIVE = "adaptive"
    LEAST_REQUESTS = "least_requests"

class HealthCheckType(Enum):
    TCP = "tcp"
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    CUSTOM = "custom"

class ServerState(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"

@dataclass(frozen=True, slots=True)
class Server:
    id: str
    host: str
    port: int
    weight: int = 1
    max_connections: int = 1000
    current_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    total_response_time: float = 0.0
    state: ServerState = ServerState.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: str = "/health"
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def address(self) -> Tuple[str, int]:
        return (self.host, self.port)

    @property
    def avg_response_time(self) -> float:
        return self.total_response_time / max(self.total_requests, 1)

    @property
    def error_rate(self) -> float:
        return self.total_errors / max(self.total_requests, 1)

    @property
    def load_score(self) -> float:
        if self.max_connections <= 0:
            return float('inf')
        return self.current_connections / self.max_connections

@dataclass(frozen=True, slots=True)
class HealthCheckConfig:
    type: HealthCheckType = HealthCheckType.HTTP
    interval: float = 10.0
    timeout: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    path: str = "/health"
    expected_status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class LoadBalancerConfig:
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_CONNECTIONS
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    sticky_sessions: bool = False
    session_cookie: str = "LB_SESSION"
    session_timeout: float = 3600
    max_retries: int = 3
    retry_timeout: float = 1.0
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout: float = 30.0
    slow_start: bool = True
    slow_start_duration: float = 10.0
    enable_compression: bool = True
    compression_types: List[str] = field(default_factory=lambda: ["gzip", "deflate"])
    rate_limit: int = 10000
    rate_limit_window: float = 60.0

class HealthChecker:
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def check(self, server: Server) -> bool:
        if self.config.type == HealthCheckType.TCP:
            return self._tcp_check(server)
        elif self.config.type in (HealthCheckType.HTTP, HealthCheckType.HTTPS):
            return self._http_check(server)
        elif self.config.type == HealthCheckType.GRPC:
            return self._grpc_check(server)
        else:
            return self._custom_check(server)

    def _tcp_check(self, server: Server) -> bool:
        try:
            sock = socket.create_connection(server.address, timeout=self.config.timeout)
            sock.close()
            return True
        except:
            return False

    def _http_check(self, server: Server) -> bool:
        try:
            import urllib.request
            url = f"http{'s' if self.config.type == HealthCheckType.HTTPS else ''}://{server.host}:{server.port}{self.config.path}"
            req = urllib.request.Request(url, headers=self.config.headers)
            response = urllib.request.urlopen(req, timeout=self.config.timeout)
            return response.status == self.config.expected_status
        except:
            return False

    def _grpc_check(self, server: Server) -> bool:
        try:
            import grpc
            from grpc_health.v1 import health_pb2, health_pb2_grpc
            channel = grpc.insecure_channel(f"{server.host}:{server.port}")
            stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            response = stub.Check(request, timeout=self.config.timeout)
            return response.status == health_pb2.HealthCheckResponse.SERVING
        except:
            return False

    def _custom_check(self, server: Server) -> bool:
        return True

    def start(self, servers: Dict[str, Server], callback: Callable[[Server, bool], None]):
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, args=(servers, callback), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _check_loop(self, servers: Dict[str, Server], callback: Callable[[Server, bool], None]):
        while self.running:
            for server in servers.values():
                if not self.running:
                    break
                healthy = self.check(server)
                callback(server, healthy)
            time.sleep(self.config.interval)

class CircuitBreaker:
    def __init__(self, threshold: float = 0.5, timeout: float = 30.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"
        self._lock = threading.Lock()

    def record_success(self):
        with self._lock:
            self.success_count += 1
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            total = self.failure_count + self.success_count
            if total > 10 and self.failure_count / total >= self.threshold:
                self.state = "open"

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half_open"
                    return True
                return False
            else:
                return True

    def reset(self):
        with self._lock:
            self.failure_count = 0
            self.success_count = 0
            self.state = "closed"

class RateLimiter:
    def __init__(self, max_requests: int, window: float):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.time()
            window_start = now - self.window

            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()

            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return True
            return False

    def get_remaining(self, key: str) -> int:
        with self._lock:
            now = time.time()
            window_start = now - self.window
            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()
            return max(0, self.max_requests - len(self.requests[key]))

    def reset(self, key: str):
        with self._lock:
            self.requests[key].clear()

class ConsistentHashRing:
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
        self._lock = threading.RLock()

    def add_node(self, node: str):
        with self._lock:
            self.nodes.add(node)
            for i in range(self.replicas):
                key = self._hash(f"{node}:{i}")
                self.ring[key] = node
            self._rebuild_sorted_keys()

    def remove_node(self, node: str):
        with self._lock:
            self.nodes.discard(node)
            for i in range(self.replicas):
                key = self._hash(f"{node}:{i}")
                self.ring.pop(key, None)
            self._rebuild_sorted_keys()

    def get_node(self, key: str) -> Optional[str]:
        with self._lock:
            if not self.ring:
                return None
            hash_key = self._hash(key)
            idx = bisect_right(self.sorted_keys, hash_key)
            if idx >= len(self.sorted_keys):
                idx = 0
            return self.ring[self.sorted_keys[idx]]

    def get_nodes(self, key: str, count: int) -> List[str]:
        with self._lock:
            if not self.ring or count <= 0:
                return []
            hash_key = self._hash(key)
            idx = bisect_right(self.sorted_keys, hash_key)
            result = []
            seen = set()
            for i in range(len(self.sorted_keys)):
                node = self.ring[self.sorted_keys[(idx + i) % len(self.sorted_keys)]]
                if node not in seen:
                    seen.add(node)
                    result.append(node)
                    if len(result) >= count:
                        break
            return result

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _rebuild_sorted_keys(self):
        self.sorted_keys = sorted(self.ring.keys())

def bisect_right(a: List[int], x: int) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

class LoadBalancer:
    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        self.servers: Dict[str, Server] = {}
        self.algorithm = self.config.algorithm
        self.health_checker = HealthChecker(self.config.health_check)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiter = RateLimiter(self.config.rate_limit, self.config.rate_limit_window)
        self.consistent_hash = ConsistentHashRing()
        self.session_map: Dict[str, str] = {}
        self.session_lock = threading.RLock()
        self.stats: Dict[str, Any] = defaultdict(int)
        self.round_robin_index = 0
        self._lock = threading.RLock()

    def add_server(self, server: Server):
        with self._lock:
            self.servers[server.id] = server
            self.circuit_breakers[server.id] = CircuitBreaker(
                self.config.circuit_breaker_threshold,
                self.config.circuit_breaker_timeout
            )
            self.consistent_hash.add_node(server.id)

    def remove_server(self, server_id: str):
        with self._lock:
            if server_id in self.servers:
                del self.servers[server_id]
            if server_id in self.circuit_breakers:
                del self.circuit_breakers[server_id]
            self.consistent_hash.remove_node(server_id)

    def get_server(self, client_ip: str = None, session_id: str = None) -> Optional[Server]:
        with self._lock:
            healthy_servers = [s for s in self.servers.values() if s.state == ServerState.HEALTHY]
            if not healthy_servers:
                return None

            if self.config.sticky_sessions and session_id:
                with self.session_lock:
                    if session_id in self.session_map:
                        server_id = self.session_map[session_id]
                        if server_id in self.servers and self.servers[server_id].state == ServerState.HEALTHY:
                            return self.servers[server_id]

            server = self._select_server(healthy_servers, client_ip, session_id)

            if server and self.config.sticky_sessions and session_id:
                with self.session_lock:
                    self.session_map[session_id] = server.id

            return server

    def _select_server(self, servers: List[Server], client_ip: str = None, session_id: str = None) -> Optional[Server]:
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return self._round_robin(servers)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(servers)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(servers)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections(servers)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time(servers)
        elif self.algorithm == LoadBalancingAlgorithm.IP_HASH:
            return self._ip_hash(servers, client_ip)
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            return self._consistent_hash(servers, client_ip or session_id)
        elif self.algorithm == LoadBalancingAlgorithm.RANDOM:
            return self._random(servers)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_RANDOM:
            return self._weighted_random(servers)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_BANDWIDTH:
            return self._least_bandwidth(servers)
        elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return self._adaptive(servers)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_REQUESTS:
            return self._least_requests(servers)
        return servers[0] if servers else None

    def _round_robin(self, servers: List[Server]) -> Server:
        server = servers[self.round_robin_index % len(servers)]
        self.round_robin_index += 1
        return server

    def _weighted_round_robin(self, servers: List[Server]) -> Server:
        total_weight = sum(s.weight for s in servers)
        r = random.randint(1, total_weight)
        for server in servers:
            r -= server.weight
            if r <= 0:
                return server
        return servers[-1]

    def _least_connections(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.current_connections)

    def _weighted_least_connections(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.current_connections / max(s.weight, 1))

    def _least_response_time(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.avg_response_time)

    def _ip_hash(self, servers: List[Server], client_ip: str) -> Server:
        if not client_ip:
            return self._round_robin(servers)
        hash_val = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return servers[hash_val % len(servers)]

    def _consistent_hash(self, servers: List[Server], key: str) -> Server:
        if not key:
            return self._round_robin(servers)
        server_id = self.consistent_hash.get_node(key)
        if server_id and server_id in self.servers:
            return self.servers[server_id]
        return self._round_robin(servers)

    def _random(self, servers: List[Server]) -> Server:
        return random.choice(servers)

    def _weighted_random(self, servers: List[Server]) -> Server:
        total_weight = sum(s.weight for s in servers)
        r = random.uniform(0, total_weight)
        for server in servers:
            r -= server.weight
            if r <= 0:
                return server
        return servers[-1]

    def _least_bandwidth(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.metadata.get('bandwidth_usage', 0))

    def _adaptive(self, servers: List[Server]) -> Server:
        scores = []
        for server in servers:
            cb = self.circuit_breakers.get(server.id)
            if cb and not cb.can_execute():
                continue

            load_score = server.load_score
            error_score = server.error_rate
            latency_score = server.avg_response_time / 1000.0
            health_score = 1.0 if server.state == ServerState.HEALTHY else 0.0

            score = (0.4 * load_score + 0.3 * error_score +
                    0.2 * latency_score + 0.1 * (1 - health_score))
            scores.append((score, server))

        if not scores:
            return self._least_connections(servers)

        return min(scores, key=lambda x: x[0])[1]

    def _least_requests(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.total_requests)

    def record_request(self, server_id: str, response_time: float, error: bool = False):
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                server.current_connections += 1
                server.total_requests += 1
                server.total_response_time += response_time
                if error:
                    server.total_errors += 1

                cb = self.circuit_breakers.get(server_id)
                if cb:
                    if error:
                        cb.record_failure()
                    else:
                        cb.record_success()

    def record_response(self, server_id: str):
        with self._lock:
            if server_id in self.servers:
                self.servers[server_id].current_connections = max(0, self.servers[server_id].current_connections - 1)

    def start_health_checks(self):
        def health_callback(server: Server, healthy: bool):
            with self._lock:
                if healthy:
                    server.consecutive_successes += 1
                    server.consecutive_failures = 0
                    if server.consecutive_successes >= self.config.health_check.healthy_threshold:
                        server.state = ServerState.HEALTHY
                else:
                    server.consecutive_failures += 1
                    server.consecutive_successes = 0
                    if server.consecutive_failures >= self.config.health_check.unhealthy_threshold:
                        server.state = ServerState.UNHEALTHY
                server.last_health_check = time.time()

        self.health_checker.start(self.servers, health_callback)

    def stop_health_checks(self):
        self.health_checker.stop()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_servers": len(self.servers),
                "healthy_servers": sum(1 for s in self.servers.values() if s.state == ServerState.HEALTHY),
                "unhealthy_servers": sum(1 for s in self.servers.values() if s.state == ServerState.UNHEALTHY),
                "total_connections": sum(s.current_connections for s in self.servers.values()),
                "total_requests": sum(s.total_requests for s in self.servers.values()),
                "total_errors": sum(s.total_errors for s in self.servers.values()),
                "avg_response_time": np.mean([s.avg_response_time for s in self.servers.values()]) if self.servers else 0,
                "algorithm": self.algorithm.value,
                **self.stats
            }

class LoadBalancerPool:
    def __init__(self):
        self.balancers: Dict[str, LoadBalancer] = {}
        self._lock = threading.RLock()

    def create_balancer(self, name: str, config: LoadBalancerConfig = None) -> LoadBalancer:
        with self._lock:
            balancer = LoadBalancer(config)
            self.balancers[name] = balancer
            return balancer

    def get_balancer(self, name: str) -> Optional[LoadBalancer]:
        with self._lock:
            return self.balancers.get(name)

    def remove_balancer(self, name: str) -> bool:
        with self._lock:
            if name in self.balancers:
                self.balancers[name].stop_health_checks()
                del self.balancers[name]
                return True
            return False

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {name: balancer.get_stats() for name, balancer in self.balancers.items()}

class AsyncLoadBalancer(LoadBalancer):
    def __init__(self, config: LoadBalancerConfig = None):
        super().__init__(config)
        self.async_servers: Dict[str, Server] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def get_server_async(self, client_ip: str = None, session_id: str = None) -> Optional[Server]:
        return self.get_server(client_ip, session_id)

    async def execute_request(self, server_id: str, request: Any) -> Any:
        server = self.servers.get(server_id)
        if not server:
            raise ValueError(f"Server {server_id} not found")

        start = time.time()
        try:
            result = await self._send_request(server, request)
            self.record_request(server_id, time.time() - start, error=False)
            return result
        except Exception as e:
            self.record_request(server_id, time.time() - start, error=True)
            raise
        finally:
            self.record_response(server_id)

    async def _send_request(self, server: Server, request: Any) -> Any:
        await asyncio.sleep(0.001)
        return {"status": "ok", "server": server.id}

class ServiceDiscovery:
    def __init__(self):
        self.services: Dict[str, Dict[str, Server]] = defaultdict(dict)
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

    def register(self, service_name: str, server: Server):
        with self._lock:
            self.services[service_name][server.id] = server
            for watcher in self.watchers[service_name]:
                watcher(service_name, "register", server)

    def deregister(self, service_name: str, server_id: str):
        with self._lock:
            if service_name in self.services and server_id in self.services[service_name]:
                server = self.services[service_name].pop(server_id)
                for watcher in self.watchers[service_name]:
                    watcher(service_name, "deregister", server)

    def get_servers(self, service_name: str) -> List[Server]:
        with self._lock:
            return list(self.services.get(service_name, {}).values())

    def get_healthy_servers(self, service_name: str) -> List[Server]:
        with self._lock:
            return [s for s in self.services.get(service_name, {}).values()
                   if s.state == ServerState.HEALTHY]

    def watch(self, service_name: str, callback: Callable):
        with self._lock:
            self.watchers[service_name].append(callback)

    def unwatch(self, service_name: str, callback: Callable):
        with self._lock:
            if callback in self.watchers[service_name]:
                self.watchers[service_name].remove(callback)

def create_load_balancer(algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_CONNECTIONS,
                         config: LoadBalancerConfig = None) -> LoadBalancer:
    if config is None:
        config = LoadBalancerConfig(algorithm=algorithm)
    else:
        config.algorithm = algorithm
    return LoadBalancer(config)

def create_balancer_pool() -> LoadBalancerPool:
    return LoadBalancerPool()