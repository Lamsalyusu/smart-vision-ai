from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import socket
import ssl
import threading
import asyncio
import json
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager


class ProtocolType(Enum):
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"
    MQTT = "mqtt"
    COAP = "coap"
    QUIC = "quic"
    WEBSOCKET_SECURE = "wss"


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    protocol: ProtocolType = ProtocolType.TCP
    max_connections: int = 1000
    timeout: float = 30.0
    keepalive: float = 60.0
    buffer_size: int = 8192
    enable_ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""
    compression: bool = False


@dataclass(frozen=True, slots=True)
class Connection:
    id: str
    remote_addr: Tuple[str, int]
    local_addr: Tuple[str, int]
    state: ConnectionState = ConnectionState.DISCONNECTED
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_received: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Packet:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    destination: str = ""
    payload: bytes = b""
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    ack_sequence: int = 0
    flags: int = 0


class NetworkInterface(ABC):
    @abstractmethod
    async def connect(self, host: str, port: int) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        pass

    @abstractmethod
    async def send(self, data: bytes) -> bool:
        pass

    @abstractmethod
    async def receive(self) -> Optional[bytes]:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass


class TCPConnection(NetworkInterface):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.state = ConnectionState.DISCONNECTED
        self.connection_id = str(uuid.uuid4())

    async def connect(self, host: str, port: int) -> bool:
        try:
            if self.config.enable_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                self.reader, self.writer = await asyncio.open_connection(host, port, ssl=ssl_context)
            else:
                self.reader, self.writer = await asyncio.open_connection(host, port)

            self.state = ConnectionState.CONNECTED
            return True
        except Exception:
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> bool:
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.state = ConnectionState.DISCONNECTED
        return True

    async def send(self, data: bytes) -> bool:
        if self.writer and self.state == ConnectionState.CONNECTED:
            try:
                self.writer.write(data)
                await self.writer.drain()
                return True
            except Exception:
                self.state = ConnectionState.ERROR
        return False

    async def receive(self) -> Optional[bytes]:
        if self.reader and self.state == ConnectionState.CONNECTED:
            try:
                data = await self.reader.read(self.config.buffer_size)
                return data if data else None
            except Exception:
                self.state = ConnectionState.ERROR
        return None

    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED


class UDPConnection(NetworkInterface):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.state = ConnectionState.DISCONNECTED
        self.remote_addr: Optional[Tuple[str, int]] = None

    async def connect(self, host: str, port: int) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.config.timeout)
            self.remote_addr = (host, port)
            self.state = ConnectionState.CONNECTED
            return True
        except Exception:
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> bool:
        if self.socket:
            self.socket.close()
        self.state = ConnectionState.DISCONNECTED
        return True

    async def send(self, data: bytes) -> bool:
        if self.socket and self.remote_addr:
            try:
                self.socket.sendto(data, self.remote_addr)
                return True
            except Exception:
                self.state = ConnectionState.ERROR
        return False

    async def receive(self) -> Optional[bytes]:
        if self.socket:
            try:
                data, addr = self.socket.recvfrom(self.config.buffer_size)
                self.remote_addr = addr
                return data
            except socket.timeout:
                return None
            except Exception:
                self.state = ConnectionState.ERROR
        return None

    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED


class WebSocketConnection(NetworkInterface):
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.websocket = None
        self.state = ConnectionState.DISCONNECTED

    async def connect(self, host: str, port: int) -> bool:
        try:
            import websockets
            protocol = "wss" if self.config.enable_ssl else "ws"
            uri = f"{protocol}://{host}:{port}"
            self.websocket = await websockets.connect(uri)
            self.state = ConnectionState.CONNECTED
            return True
        except Exception:
            self.state = ConnectionState.ERROR
            return False

    async def disconnect(self) -> bool:
        if self.websocket:
            await self.websocket.close()
        self.state = ConnectionState.DISCONNECTED
        return True

    async def send(self, data: bytes) -> bool:
        if self.websocket and self.state == ConnectionState.CONNECTED:
            try:
                await self.websocket.send(data)
                return True
            except Exception:
                self.state = ConnectionState.ERROR
        return False

    async def receive(self) -> Optional[bytes]:
        if self.websocket and self.state == ConnectionState.CONNECTED:
            try:
                data = await self.websocket.recv()
                return data if isinstance(data, bytes) else data.encode()
            except Exception:
                self.state = ConnectionState.ERROR
        return None

    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED


class NetworkServer:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.server = None
        self.connections: Dict[str, Connection] = {}
        self.handlers: Dict[str, Callable] = {}
        self.running = False

    async def start(self):
        if self.config.protocol == ProtocolType.TCP:
            if self.config.enable_ssl:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(self.config.ssl_cert, self.config.ssl_key)
                self.server = await asyncio.start_server(
                    self._handle_client, self.config.host, self.config.port, ssl=ssl_context
                )
            else:
                self.server = await asyncio.start_server(
                    self._handle_client, self.config.host, self.config.port
                )
        elif self.config.protocol == ProtocolType.WEBSOCKET:
            import websockets
            self.server = await websockets.serve(
                self._handle_websocket, self.config.host, self.config.port
            )

        self.running = True

    async def stop(self):
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        conn_id = str(uuid.uuid4())
        connection = Connection(
            id=conn_id,
            remote_addr=addr,
            local_addr=writer.get_extra_info('sockname'),
            state=ConnectionState.CONNECTED
        )
        self.connections[conn_id] = connection

        try:
            while self.running:
                data = await reader.read(self.config.buffer_size)
                if not data:
                    break

                connection.bytes_received += len(data)
                connection.last_activity = time.time()

                await self._process_message(conn_id, data)

        except Exception:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            del self.connections[conn_id]

    async def _handle_websocket(self, websocket, path):
        conn_id = str(uuid.uuid4())
        connection = Connection(
            id=conn_id,
            remote_addr=("websocket", 0),
            local_addr=("websocket", 0),
            state=ConnectionState.CONNECTED
        )
        self.connections[conn_id] = connection

        try:
            async for message in websocket:
                data = message if isinstance(message, bytes) else message.encode()
                connection.bytes_received += len(data)
                connection.last_activity = time.time()
                await self._process_message(conn_id, data)
        except Exception:
            pass
        finally:
            del self.connections[conn_id]

    async def _process_message(self, conn_id: str, data: bytes):
        try:
            message = json.loads(data.decode())
            msg_type = message.get('type', 'unknown')

            if msg_type in self.handlers:
                response = await self.handlers[msg_type](message)
                if response:
                    await self.send(conn_id, response)
        except Exception:
            pass

    def register_handler(self, msg_type: str, handler: Callable):
        self.handlers[msg_type] = handler

    async def send(self, conn_id: str, data: Any) -> bool:
        if conn_id in self.connections:
            connection = self.connections[conn_id]
            try:
                if isinstance(data, (dict, list)):
                    payload = json.dumps(data).encode()
                else:
                    payload = data if isinstance(data, bytes) else str(data).encode()

                return True
            except Exception:
                pass
        return False

    async def broadcast(self, data: Any, exclude: List[str] = None):
        exclude = exclude or []
        for conn_id in list(self.connections.keys()):
            if conn_id not in exclude:
                await self.send(conn_id, data)


class NetworkClient:
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.connection: Optional[NetworkInterface] = None
        self.handlers: Dict[str, Callable] = {}
        self.receive_task: Optional[asyncio.Task] = None
        self.running = False

    async def connect(self) -> bool:
        if self.config.protocol == ProtocolType.TCP:
            self.connection = TCPConnection(self.config)
        elif self.config.protocol == ProtocolType.UDP:
            self.connection = UDPConnection(self.config)
        elif self.config.protocol == ProtocolType.WEBSOCKET:
            self.connection = WebSocketConnection(self.config)
        else:
            return False

        success = await self.connection.connect(self.config.host, self.config.port)
        if success:
            self.running = True
            self.receive_task = asyncio.create_task(self._receive_loop())
        return success

    async def disconnect(self) -> bool:
        self.running = False
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
        if self.connection:
            return await self.connection.disconnect()
        return False

    async def send(self, data: Any) -> bool:
        if self.connection and self.connection.is_connected():
            if isinstance(data, (dict, list)):
                payload = json.dumps(data).encode()
            else:
                payload = data if isinstance(data, bytes) else str(data).encode()
            return await self.connection.send(payload)
        return False

    async def _receive_loop(self):
        while self.running and self.connection and self.connection.is_connected():
            data = await self.connection.receive()
            if data:
                try:
                    message = json.loads(data.decode())
                    msg_type = message.get('type', 'unknown')
                    if msg_type in self.handlers:
                        await self.handlers[msg_type](message)
                except Exception:
                    pass
            await asyncio.sleep(0.001)

    def register_handler(self, msg_type: str, handler: Callable):
        self.handlers[msg_type] = handler


class LoadBalancer:
    def __init__(self, algorithm: str = "round_robin"):
        self.algorithm = algorithm
        self.servers: List[Tuple[str, int]] = []
        self.weights: List[float] = []
        self.current_index = 0
        self.connections: Dict[str, int] = defaultdict(int)

    def add_server(self, host: str, port: int, weight: float = 1.0):
        self.servers.append((host, port))
        self.weights.append(weight)

    def remove_server(self, host: str, port: int):
        for i, (h, p) in enumerate(self.servers):
            if h == host and p == port:
                del self.servers[i]
                del self.weights[i]
                break

    def get_server(self) -> Optional[Tuple[str, int]]:
        if not self.servers:
            return None

        if self.algorithm == "round_robin":
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server
        elif self.algorithm == "least_connections":
            min_conn = min(self.connections.get(f"{h}:{p}", 0) for h, p in self.servers)
            for h, p in self.servers:
                if self.connections.get(f"{h}:{p}", 0) == min_conn:
                    return (h, p)
        elif self.algorithm == "weighted":
            total = sum(self.weights)
            r = random.uniform(0, total)
            cumulative = 0
            for i, w in enumerate(self.weights):
                cumulative += w
                if r <= cumulative:
                    return self.servers[i]

        return self.servers[0]

    def record_connection(self, host: str, port: int, delta: int = 1):
        self.connections[f"{host}:{port}"] = max(0, self.connections.get(f"{host}:{port}", 0) + delta)


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"

    def call(self, func: Callable, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def execute(self, func: Callable, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    if self.jitter:
                        delay *= random.uniform(0.5, 1.5)
                    time.sleep(delay)

        raise last_exception


class NetworkMetrics:
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.packets_sent = 0
        self.packets_received = 0
        self.errors = 0
        self.latencies: List[float] = []
        self.lock = threading.Lock()

    def record_sent(self, bytes_count: int):
        with self.lock:
            self.bytes_sent += bytes_count
            self.packets_sent += 1

    def record_received(self, bytes_count: int):
        with self.lock:
            self.bytes_received += bytes_count
            self.packets_received += 1

    def record_latency(self, latency: float):
        with self.lock:
            self.latencies.append(latency)
            if len(self.latencies) > 10000:
                self.latencies = self.latencies[-10000:]

    def record_error(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'bytes_sent': self.bytes_sent,
                'bytes_received': self.bytes_received,
                'packets_sent': self.packets_sent,
                'packets_received': self.packets_received,
                'errors': self.errors,
                'avg_latency': np.mean(self.latencies) if self.latencies else 0,
                'p50_latency': np.percentile(self.latencies, 50) if self.latencies else 0,
                'p95_latency': np.percentile(self.latencies, 95) if self.latencies else 0,
                'p99_latency': np.percentile(self.latencies, 99) if self.latencies else 0
            }


import random
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
