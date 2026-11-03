from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import ssl
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import asyncio
import websockets
import logging

logger = logging.getLogger(__name__)

# TODO: Add support for WebSocket compression (permessage-deflate)
# FIXME: Connection cleanup on abrupt disconnect is not always reliable
# XXX: No built-in support for message acknowledgments/retries
# HACK: Using asyncio.Queue for pub/sub - not suitable for high-throughput scenarios
# FIXME: Reconnection logic in WebSocketClient is basic - no exponential backoff config


class ConnectionState(Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class MessageType(Enum):
    TEXT = "text"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class WebSocketMessage:
    type: MessageType
    payload: Union[str, bytes]
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionInfo:
    connection_id: str
    remote_address: str
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    subscriptions: Set[str] = field(default_factory=set)
    user_id: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)


class WebSocketServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        ssl_context: Optional[ssl.SSLContext] = None,
        max_connections: int = 10000,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        max_message_size: int = 16 * 1024 * 1024,
    ):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.max_connections = max_connections
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_message_size = max_message_size

        self._connections: Dict[str, ConnectionInfo] = {}
        self._websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._connection_handlers: List[Callable] = []
        self._disconnection_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._running = False
        self._server: Optional[websockets.WebSocketServer] = None
        self._lock = asyncio.Lock()

    async def start(self):
        if self._running:
            return

        self._running = True
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ssl=self.ssl_context,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            max_size=self.max_message_size,
            max_queue=1000,
        )
        logger.info(f"WebSocket server started on {self.host}:{self.port}")

    async def stop(self):
        self._running = False

        for conn_id, ws in list(self._websockets.items()):
            try:
                await ws.close(1001, "Server shutting down")
            except Exception:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped")

    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        if len(self._connections) >= self.max_connections:
            await websocket.close(1013, "Server at capacity")
            return

        connection_id = str(uuid.uuid4())
        remote_addr = websocket.remote_address[0] if websocket.remote_address else "unknown"

        conn_info = ConnectionInfo(
            connection_id=connection_id,
            remote_address=remote_addr,
            state=ConnectionState.CONNECTED,
        )

        self._connections[connection_id] = conn_info
        self._websockets[connection_id] = websocket

        for handler in self._connection_handlers:
            try:
                await handler(connection_id, conn_info)
            except Exception as e:
                logger.error(f"Connection handler error: {e}")

        try:
            async for message in websocket:
                await self._handle_message(connection_id, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Connection {connection_id} error: {e}")
            for handler in self._error_handlers:
                try:
                    await handler(connection_id, e)
                except Exception:
                    pass
        finally:
            await self._cleanup_connection(connection_id)

    async def _handle_message(self, connection_id: str, message: Union[str, bytes]):
        conn_info = self._connections.get(connection_id)
        if not conn_info:
            return

        conn_info.last_activity = time.time()

        if isinstance(message, bytes):
            ws_message = WebSocketMessage(
                type=MessageType.BINARY,
                payload=message,
            )
        else:
            ws_message = WebSocketMessage(
                type=MessageType.TEXT,
                payload=message,
            )

        try:
            data = json.loads(ws_message.payload) if isinstance(ws_message.payload, str) else {}
            msg_type = data.get("type", "message")

            handler = self._message_handlers.get(msg_type)
            if handler:
                await handler(connection_id, data)
            else:
                await self._default_message_handler(connection_id, data)
        except json.JSONDecodeError:
            await self._handle_raw_message(connection_id, ws_message)
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _handle_raw_message(self, connection_id: str, message: WebSocketMessage):
        pass

    async def _default_message_handler(self, connection_id: str, data: Dict[str, Any]):
        pass

    async def _cleanup_connection(self, connection_id: str):
        conn_info = self._connections.pop(connection_id, None)
        ws = self._websockets.pop(connection_id, None)

        if conn_info:
            conn_info.state = ConnectionState.DISCONNECTED

        for handler in self._disconnection_handlers:
            try:
                await handler(connection_id, conn_info)
            except Exception as e:
                logger.error(f"Disconnection handler error: {e}")

        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

        for topic, subscribers in self._subscriptions.items():
            subscribers.discard(connection_id)

    async def send(self, connection_id: str, message: Union[str, bytes, Dict], message_type: MessageType = MessageType.TEXT) -> bool:
        ws = self._websockets.get(connection_id)
        if not ws or ws.closed:
            return False

        try:
            if isinstance(message, dict):
                payload = json.dumps(message)
            elif isinstance(message, bytes):
                payload = message
            else:
                payload = str(message)

            if isinstance(payload, str):
                await ws.send(payload)
            else:
                await ws.send(payload)

            return True
        except Exception as e:
            logger.error(f"Send error for {connection_id}: {e}")
            return False

    async def broadcast(self, message: Union[str, Dict], topic: str = None, exclude: Set[str] = None) -> int:
        if isinstance(message, dict):
            payload = json.dumps(message)
        else:
            payload = str(message)

        exclude = exclude or set()
        sent = 0

        if topic:
            subscribers = self._subscriptions.get(topic, set()).copy()
        else:
            subscribers = set(self._websockets.keys())

        for conn_id in subscribers:
            if conn_id in exclude:
                continue
            if await self.send(conn_id, message):
                sent += 1

        return sent

    async def subscribe(self, connection_id: str, topic: str) -> bool:
        if connection_id not in self._connections:
            return False

        self._subscriptions[topic].add(connection_id)
        self._connections[connection_id].subscriptions.add(topic)
        return True

    async def unsubscribe(self, connection_id: str, topic: str) -> bool:
        if connection_id in self._subscriptions.get(topic, set()):
            self._subscriptions[topic].discard(connection_id)

        if connection_id in self._connections:
            self._connections[connection_id].subscriptions.discard(topic)

        return True

    def on_connect(self, handler: Callable):
        self._connection_handlers.append(handler)

    def on_disconnect(self, handler: Callable):
        self._disconnection_handlers.append(handler)

    def on_message(self, message_type: str, handler: Callable):
        self._message_handlers[message_type] = handler

    def on_error(self, handler: Callable):
        self._error_handlers.append(handler)

    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        return self._connections.get(connection_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_connections": len(self._connections),
            "connected": sum(1 for c in self._connections.values() if c.state == ConnectionState.CONNECTED),
            "subscriptions": {topic: len(subs) for topic, subs in self._subscriptions.items()},
        }


class WebSocketClient:
    def __init__(
        self,
        uri: str,
        ssl_context: Optional[ssl.SSLContext] = None,
        ping_interval: float = 20.0,
        reconnect: bool = True,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        self.uri = uri
        self.ssl_context = ssl_context
        self.ping_interval = ping_interval
        self.reconnect = reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts

        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_attempts = 0
        self._message_handlers: Dict[str, Callable] = {}
        self._connection_handler: Optional[Callable] = None
        self._disconnection_handler: Optional[Callable] = None
        self._running_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        try:
            self._websocket = await websockets.connect(
                self.uri,
                ssl=self.ssl_context,
                ping_interval=20,
                ping_timeout=10,
            )
            self._reconnect_attempts = 0

            if self._connection_handler:
                await self._connection_handler()

            self._running = True
            self._running_task = asyncio.create_task(self._run())
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def _run(self):
        while self._running:
            try:
                async for message in self._websocket:
                    await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                if self._reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                    self._reconnect_attempts += 1
                    logger.info(f"Reconnecting... attempt {self._reconnect_attempts}")
                    await asyncio.sleep(self.reconnect_interval)
                    await self.connect()
                else:
                    self._running = False
        except Exception as e:
            logger.error(f"Client error: {e}")
            if self._reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
                self._reconnect_attempts += 1
                await asyncio.sleep(self.reconnect_interval)
                await self.connect()

    async def _handle_message(self, message: Union[str, bytes]):
        try:
            data = json.loads(message) if isinstance(message, str) else {}
            msg_type = data.get("type", "message")
            handler = self._message_handlers.get(msg_type)
            if handler:
                await handler(data)
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def send(self, message: Union[str, Dict], message_type: str = "message"):
        if not self._websocket or self._websocket.closed:
            return False

        try:
            if isinstance(message, dict):
                message["type"] = message.get("type", "message")
                payload = json.dumps(message)
            else:
                payload = str(message)

            await self._websocket.send(payload)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    async def disconnect(self):
        self._running = False
        if self._running_task:
            self._running_task.cancel()
        if self._websocket:
            await self._websocket.close()

    def on_message(self, message_type: str, handler: Callable):
        self._message_handlers[message_type] = handler

    def on_connect(self, handler: Callable):
        self._connection_handler = handler

    def on_disconnect(self, handler: Callable):
        self._disconnection_handler = handler


class PubSubManager:
    def __init__(self):
        self._topics: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        async with self._lock:
            self._topics[topic].add(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue):
        async with self._lock:
            if topic in self._topics:
                self._topics[topic].discard(queue)

    async def publish(self, topic: str, message: Any):
        async with self._lock:
            if topic in self._topics:
                for queue in self._topics[topic].copy():
                    try:
                        queue.put_nowait(message)
                    except asyncio.QueueFull:
                        pass


class StreamProcessor:
    def __init__(
        self,
        input_topic: str,
        output_topic: str,
        processor: Callable[[Any], Any],
        pubsub: PubSubManager,
    ):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.processor = processor
        self.pubsub = pubsub
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        queue = await self.pubsub.subscribe(self.input_topic)
        self._task = asyncio.create_task(self._run(queue))

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self, queue: asyncio.Queue):
        while True:
            try:
                message = await queue.get()
                result = await self.processor(message)
                if self.output_topic:
                    await self.pubsub.publish(self.output_topic, result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream processor error: {e}")


class RateLimiter:
    def __init__(self, max_requests: int, window: float):
        self.max_requests = max_requests
        self.window = window
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        async with self._lock:
            now = time.time()
            window_start = now - self.window

            self._requests[key] = [t for t in self._requests[key] if t > window_start]

            if len(self._requests[key]) < self.max_requests:
                self._requests[key].append(now)
                return True
            return False

    async def get_remaining(self, key: str) -> int:
        async with self._lock:
            now = time.time()
            window_start = now - self.window
            self._requests[key] = [t for t in self._requests[key] if t > window_start]
            return max(0, self.max_requests - len(self._requests[key]))


class ConnectionPool:
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self._connections: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, factory: Callable[[], Any]) -> Any:
        async with self._lock:
            if key in self._connections:
                return self._connections[key]

            if len(self._connections) >= self.max_connections:
                oldest = min(self._connections.items(), key=lambda x: x[1].get('created_at', 0))
                del self._connections[oldest[0]]

            conn = await factory()
            self._connections[key] = conn
            return conn

    async def release(self, key: str):
        async with self._lock:
            if key in self._connections:
                del self._connections[key]

    async def close_all(self):
        async with self._lock:
            for conn in self._connections.values():
                if hasattr(conn, 'close'):
                    await conn.close()
            self._connections.clear()


class LoadBalancer:
    def __init__(self):
        self._endpoints: List[Tuple[str, int]] = []
        self._current = 0
        self._lock = asyncio.Lock()

    def add_endpoint(self, host: str, port: int, weight: int = 1):
        self._endpoints.append((host, port, weight))

    async def get_endpoint(self) -> Tuple[str, int]:
        async with self._lock:
            if not self._endpoints:
                raise ValueError("No endpoints available")

            return self._endpoints[self._current % len(self._endpoints)][:2]


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        async with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")

        try:
            if inspect.iscoroutinefunction(func):
                result = await func()
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, func)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        self._state = "closed"

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    async def execute(self, func: Callable, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay,
                    )
                    if self.jitter:
                        delay *= random.uniform(0.5, 1.5)
                    await asyncio.sleep(delay)

        raise last_exception


class MessageQueue:
    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers: List[asyncio.Queue] = []

    async def put(self, item: Any):
        await self._queue.put(item)
        for sub in self._subscribers:
            try:
                sub.put_nowait(item)
            except asyncio.QueueFull:
                pass

    async def get(self) -> Any:
        return await self._queue.get()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def qsize(self) -> int:
        return self._queue.qsize()


class EventBus:
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_type: str, handler: Callable):
        async with self._lock:
            self._handlers[event_type].append(handler)

    async def unsubscribe(self, event_type: str, handler: Callable):
        async with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    async def publish(self, event_type: str, data: Any):
        handlers = self._handlers.get(event_type, []).copy()
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    await asyncio.get_event_loop().run_in_executor(None, lambda: handler(data))
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def publish_many(self, events: List[Tuple[str, Any]]):
        for event_type, data in events:
            await self.publish(event_type, data)