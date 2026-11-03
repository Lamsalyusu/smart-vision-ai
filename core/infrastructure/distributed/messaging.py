from __future__ import annotations

# SECURITY WARNING: This module supports SerializationFormat.PICKLE via pickle.dumps/pickle.loads.
# pickle is NOT secure against maliciously crafted data and can lead to arbitrary code execution.
# Only use PICKLE for trusted internal payloads. Prefer SerializationFormat.JSON for any data
# that may originate from untrusted sources (network, user input, persistent storage).

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union, Set
from enum import Enum
from abc import ABC, abstractmethod
import time
import threading
import queue
import uuid
import json
import pickle
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"
    STREAM = "stream"

class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

class SerializationFormat(Enum):
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    RAW = "raw"

class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"

@dataclass(frozen=True, slots=True)
class Message:
    id: str
    type: MessageType
    topic: str
    payload: Any
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    ttl: Optional[float] = None
    priority: int = 0
    compressed: bool = False
    encryption: bool = False

    def __post_init__(self):
        if not self.id:
            object.__setattr__(self, 'id', str(uuid.uuid4()))

@dataclass(frozen=True, slots=True)
class Subscription:
    id: str
    topic: str
    callback: Callable[[Message], None]
    filter: Optional[Callable[[Message], bool]] = None
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    auto_ack: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    dead_letter_topic: Optional[str] = None

@dataclass(frozen=True, slots=True)
class BrokerConfig:
    host: str = "localhost"
    port: int = 5672
    max_connections: int = 1000
    max_message_size: int = 10 * 1024 * 1024
    default_ttl: float = 3600
    heartbeat_interval: float = 30
    cleanup_interval: float = 60
    persistence_enabled: bool = False
    persistence_path: str = "./data"
    cluster_enabled: bool = False
    cluster_nodes: List[str] = field(default_factory=list)

class Serializer:
    def __init__(self, format: SerializationFormat = SerializationFormat.JSON,
                 compression: CompressionType = CompressionType.NONE):
        self.format = format
        self.compression = compression

    def serialize(self, message: Message) -> bytes:
        if self.format == SerializationFormat.JSON:
            data = {
                "id": message.id,
                "type": message.type.value,
                "topic": message.topic,
                "payload": message.payload,
                "headers": message.headers,
                "timestamp": message.timestamp,
                "correlation_id": message.correlation_id,
                "reply_to": message.reply_to,
                "delivery_guarantee": message.delivery_guarantee.value,
                "ttl": message.ttl,
                "priority": message.priority
            }
            payload = json.dumps(data).encode('utf-8')
        elif self.format == SerializationFormat.PICKLE:
            payload = pickle.dumps(message)
        elif self.format == SerializationFormat.RAW:
            if isinstance(message.payload, bytes):
                payload = message.payload
            elif isinstance(message.payload, str):
                payload = message.payload.encode('utf-8')
            else:
                payload = str(message.payload).encode('utf-8')
        else:
            payload = json.dumps(message.__dict__, default=str).encode('utf-8')

        if self.compression == CompressionType.GZIP:
            import gzip
            payload = gzip.compress(payload)
        elif self.compression == CompressionType.LZ4:
            import lz4.frame
            payload = lz4.frame.compress(payload)
        elif self.compression == CompressionType.ZSTD:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor()
            payload = cctx.compress(payload)

        return payload

    def deserialize(self, data: bytes) -> Message:
        payload = data

        if self.compression == CompressionType.GZIP:
            import gzip
            payload = gzip.decompress(payload)
        elif self.compression == CompressionType.LZ4:
            import lz4.frame
            payload = lz4.frame.decompress(payload)
        elif self.compression == CompressionType.ZSTD:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            payload = dctx.decompress(payload)

        if self.format == SerializationFormat.JSON:
            obj = json.loads(payload.decode('utf-8'))
            return Message(
                id=obj["id"],
                type=MessageType(obj["type"]),
                topic=obj["topic"],
                payload=obj["payload"],
                headers=obj.get("headers", {}),
                timestamp=obj.get("timestamp", time.time()),
                correlation_id=obj.get("correlation_id"),
                reply_to=obj.get("reply_to"),
                delivery_guarantee=DeliveryGuarantee(obj.get("delivery_guarantee", "at_least_once")),
                ttl=obj.get("ttl"),
                priority=obj.get("priority", 0)
            )
        elif self.format == SerializationFormat.PICKLE:
            return pickle.loads(payload)
        else:
            return Message(
                id=str(uuid.uuid4()),
                type=MessageType.EVENT,
                topic="unknown",
                payload=payload.decode('utf-8', errors='ignore')
            )

class MessageQueue:
    def __init__(self, maxsize: int = 0, priority: bool = True):
        self.maxsize = maxsize
        self.priority = priority
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize) if priority else queue.Queue(maxsize)
        self._lock = threading.RLock()
        self._size = 0

    def put(self, message: Message, block: bool = True, timeout: float = None) -> bool:
        try:
            if self.priority:
                self._queue.put((-message.priority, time.time(), message), block, timeout)
            else:
                self._queue.put(message, block, timeout)
            with self._lock:
                self._size += 1
            return True
        except queue.Full:
            return False

    def get(self, block: bool = True, timeout: float = None) -> Optional[Message]:
        try:
            if self.priority:
                _, _, msg = self._queue.get(block, timeout)
            else:
                msg = self._queue.get(block, timeout)
            with self._lock:
                self._size -= 1
            return msg
        except queue.Empty:
            return None

    def get_nowait(self) -> Optional[Message]:
        return self.get(block=False)

    def qsize(self) -> int:
        with self._lock:
            return self._size

    def empty(self) -> bool:
        return self.qsize() == 0

    def full(self) -> bool:
        return self.maxsize > 0 and self.qsize() >= self.maxsize

    def clear(self):
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._size = 0

class TopicManager:
    def __init__(self):
        self.topics: Dict[str, MessageQueue] = {}
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.pattern_subscriptions: List[Tuple[str, Subscription]] = []
        self._lock = threading.RLock()

    def create_topic(self, topic: str, maxsize: int = 0, priority: bool = True) -> MessageQueue:
        with self._lock:
            if topic not in self.topics:
                self.topics[topic] = MessageQueue(maxsize, priority)
            return self.topics[topic]

    def get_topic(self, topic: str) -> Optional[MessageQueue]:
        with self._lock:
            return self.topics.get(topic)

    def delete_topic(self, topic: str) -> bool:
        with self._lock:
            if topic in self.topics:
                del self.topics[topic]
                if topic in self.subscriptions:
                    del self.subscriptions[topic]
                return True
            return False

    def subscribe(self, subscription: Subscription) -> str:
        with self._lock:
            self.subscriptions[subscription.topic].append(subscription)
            if '*' in subscription.topic or '?' in subscription.topic:
                self.pattern_subscriptions.append((subscription.topic, subscription))
            return subscription.id

    def unsubscribe(self, topic: str, subscription_id: str) -> bool:
        with self._lock:
            if topic in self.subscriptions:
                self.subscriptions[topic] = [s for s in self.subscriptions[topic] if s.id != subscription_id]
                self.pattern_subscriptions = [(p, s) for p, s in self.pattern_subscriptions if s.id != subscription_id]
                return True
            return False

    def get_subscriptions(self, topic: str) -> List[Subscription]:
        with self._lock:
            subs = self.subscriptions.get(topic, []).copy()
            for pattern, sub in self.pattern_subscriptions:
                if self._match_pattern(pattern, topic):
                    subs.append(sub)
            return subs

    def _match_pattern(self, pattern: str, topic: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)

    def publish(self, message: Message) -> int:
        with self._lock:
            queue = self.topics.get(message.topic)
            if queue is None:
                queue = self.create_topic(message.topic)
            queue.put(message)
            return 1

    def get_all_topics(self) -> List[str]:
        with self._lock:
            return list(self.topics.keys())

class ConnectionManager:
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.connections: Dict[str, Connection] = {}
        self.selector = selectors.DefaultSelector()
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self._lock = threading.RLock()
        self.serializer = Serializer()

    def start(self) -> bool:
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.host, self.config.port))
            self.server_socket.listen(self.config.max_connections)
            self.server_socket.setblocking(False)
            self.selector.register(self.server_socket, selectors.EVENT_READ, self._accept)
            self.running = True
            return True
        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            return False

    def stop(self):
        self.running = False
        if self.server_socket:
            self.selector.unregister(self.server_socket)
            self.server_socket.close()
        for conn in list(self.connections.values()):
            conn.close()
        self.selector.close()

    def _accept(self, sock: socket.socket, mask: int):
        conn, addr = sock.accept()
        conn.setblocking(False)
        connection = Connection(conn, addr, self)
        with self._lock:
            self.connections[connection.id] = connection
        self.selector.register(conn, selectors.EVENT_READ, connection.read)

    def broadcast(self, message: Message, exclude: Set[str] = None):
        exclude = exclude or set()
        data = self.serializer.serialize(message)
        for conn in self.connections.values():
            if conn.id not in exclude:
                conn.send(data)

    def get_connection_count(self) -> int:
        with self._lock:
            return len(self.connections)

class Connection:
    def __init__(self, sock: socket.socket, addr: Tuple[str, int], manager: ConnectionManager):
        self.id = str(uuid.uuid4())
        self.sock = sock
        self.addr = addr
        self.manager = manager
        self.buffer = bytearray()
        self.connected = True
        self.last_heartbeat = time.time()
        self.subscriptions: Set[str] = set()
        self.pending_acks: Dict[str, Future] = {}

    def read(self, sock: socket.socket, mask: int):
        try:
            data = sock.recv(4096)
            if not data:
                self.close()
                return

            self.buffer.extend(data)
            self._process_buffer()
        except Exception as e:
            logging.error(f"Read error: {e}")
            self.close()

    def _process_buffer(self):
        while len(self.buffer) >= 4:
            length = struct.unpack('!I', self.buffer[:4])[0]
            if len(self.buffer) < 4 + length:
                break

            message_data = bytes(self.buffer[4:4+length])
            self.buffer = self.buffer[4+length:]

            try:
                message = self.manager.serializer.deserialize(message_data)
                self._handle_message(message)
            except Exception as e:
                logging.error(f"Message parse error: {e}")

    def _handle_message(self, message: Message):
        if message.type == MessageType.HEARTBEAT:
            self.last_heartbeat = time.time()
            self.send(self.manager.serializer.serialize(Message(
                id=str(uuid.uuid4()),
                type=MessageType.ACK,
                topic="",
                payload={"original_id": message.id}
            )))
        elif message.type == MessageType.ACK:
            if message.correlation_id in self.pending_acks:
                future = self.pending_acks.pop(message.correlation_id)
                future.set_result(message)
        else:
            self.manager.topic_manager.publish(message)

    def send(self, data: bytes):
        try:
            length = struct.pack('!I', len(data))
            self.sock.sendall(length + data)
        except Exception as e:
            logging.error(f"Send error: {e}")
            self.close()

    def close(self):
        if self.connected:
            self.connected = False
            try:
                self.manager.selector.unregister(self.sock)
            except:
                pass
            try:
                self.sock.close()
            except:
                pass
            with self.manager._lock:
                if self.id in self.manager.connections:
                    del self.manager.connections[self.id]

class MessageBroker:
    def __init__(self, config: BrokerConfig = None):
        self.config = config or BrokerConfig()
        self.topic_manager = TopicManager()
        self.connection_manager = ConnectionManager(self.config)
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.stats: Dict[str, Any] = defaultdict(int)
        self._cleanup_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not self.connection_manager.start():
            return False

        self.running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        return True

    def stop(self):
        self.running = False
        self.connection_manager.stop()
        self.executor.shutdown(wait=True)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

    def publish(self, topic: str, payload: Any, **kwargs) -> str:
        message = Message(
            id=str(uuid.uuid4()),
            type=kwargs.get('msg_type', MessageType.EVENT),
            topic=topic,
            payload=payload,
            headers=kwargs.get('headers', {}),
            correlation_id=kwargs.get('correlation_id'),
            reply_to=kwargs.get('reply_to'),
            delivery_guarantee=kwargs.get('delivery_guarantee', DeliveryGuarantee.AT_LEAST_ONCE),
            ttl=kwargs.get('ttl', self.config.default_ttl),
            priority=kwargs.get('priority', 0)
        )
        count = self.topic_manager.publish(message)
        self.stats['messages_published'] += count
        return message.id

    def subscribe(self, topic: str, callback: Callable[[Message], None], **kwargs) -> str:
        subscription = Subscription(
            id=str(uuid.uuid4()),
            topic=topic,
            callback=callback,
            filter=kwargs.get('filter'),
            delivery_guarantee=kwargs.get('delivery_guarantee', DeliveryGuarantee.AT_LEAST_ONCE),
            auto_ack=kwargs.get('auto_ack', True),
            max_retries=kwargs.get('max_retries', 3),
            retry_delay=kwargs.get('retry_delay', 1.0),
            dead_letter_topic=kwargs.get('dead_letter_topic')
        )
        sub_id = self.topic_manager.subscribe(subscription)
        self.stats['active_subscriptions'] += 1
        return sub_id

    def unsubscribe(self, topic: str, subscription_id: str) -> bool:
        result = self.topic_manager.unsubscribe(topic, subscription_id)
        if result:
            self.stats['active_subscriptions'] -= 1
        return result

    def request_response(self, topic: str, payload: Any, timeout: float = 30) -> Optional[Message]:
        correlation_id = str(uuid.uuid4())
        reply_topic = f"reply.{correlation_id}"

        future = Future()

        def reply_handler(msg: Message):
            if not future.done():
                future.set_result(msg)

        self.subscribe(reply_topic, reply_handler)

        self.publish(topic, payload, correlation_id=correlation_id, reply_to=reply_topic)

        try:
            return future.result(timeout=timeout)
        except:
            return None
        finally:
            self.unsubscribe(reply_topic, reply_topic)

    def _cleanup_loop(self):
        while self.running:
            time.sleep(self.config.cleanup_interval)
            self._cleanup_expired_messages()
            self._cleanup_stale_connections()

    def _cleanup_expired_messages(self):
        current_time = time.time()
        for topic, queue in self.topic_manager.topics.items():
            pass

    def _cleanup_stale_connections(self):
        current_time = time.time()
        for conn in list(self.connection_manager.connections.values()):
            if current_time - conn.last_heartbeat > self.config.heartbeat_interval * 3:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "topics": len(self.topic_manager.topics),
            "connections": self.connection_manager.get_connection_count(),
            "subscriptions": sum(len(s) for s in self.topic_manager.subscriptions.values())
        }

class DistributedMessageBus:
    def __init__(self, node_id: str, cluster_nodes: List[str] = None):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes or []
        self.local_broker = MessageBroker()
        self.remote_brokers: Dict[str, MessageBroker] = {}
        self.running = False

    def start(self) -> bool:
        return self.local_broker.start()

    def stop(self):
        self.local_broker.stop()

    def publish_local(self, topic: str, payload: Any, **kwargs) -> str:
        return self.local_broker.publish(topic, payload, **kwargs)

    def publish_cluster(self, topic: str, payload: Any, **kwargs) -> Dict[str, str]:
        results = {}
        results[self.node_id] = self.publish_local(topic, payload, **kwargs)
        return results

    def subscribe(self, topic: str, callback: Callable[[Message], None], **kwargs) -> str:
        return self.local_broker.subscribe(topic, callback, **kwargs)

class EventSourcing:
    def __init__(self, broker: MessageBroker, event_store_topic: str = "events"):
        self.broker = broker
        self.event_store_topic = event_store_topic
        self.snapshots: Dict[str, Any] = {}
        self.snapshot_interval = 100
        self.event_counts: Dict[str, int] = defaultdict(int)

    def append_event(self, aggregate_id: str, event_type: str, payload: Any) -> str:
        event = {
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "payload": payload,
            "timestamp": time.time(),
            "version": self.event_counts[aggregate_id] + 1
        }
        self.event_counts[aggregate_id] += 1
        return self.broker.publish(self.event_store_topic, event)

    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Dict]:
        return []

    def create_snapshot(self, aggregate_id: str, state: Any):
        self.snapshots[aggregate_id] = {
            "state": state,
            "version": self.event_counts[aggregate_id],
            "timestamp": time.time()
        }

    def get_snapshot(self, aggregate_id: str) -> Optional[Dict]:
        return self.snapshots.get(aggregate_id)

class CQRS:
    def __init__(self, command_bus: MessageBroker, query_bus: MessageBroker):
        self.command_bus = command_bus
        self.query_bus = query_bus
        self.command_handlers: Dict[str, Callable] = {}
        self.query_handlers: Dict[str, Callable] = {}

    def register_command_handler(self, command_type: str, handler: Callable):
        self.command_handlers[command_type] = handler
        self.command_bus.subscribe(f"command.{command_type}", self._handle_command)

    def register_query_handler(self, query_type: str, handler: Callable):
        self.query_handlers[query_type] = handler
        self.query_bus.subscribe(f"query.{query_type}", self._handle_query)

    def send_command(self, command_type: str, payload: Any) -> str:
        return self.command_bus.publish(f"command.{command_type}", payload, msg_type=MessageType.COMMAND)

    def send_query(self, query_type: str, payload: Any) -> Optional[Message]:
        return self.query_bus.request_response(f"query.{query_type}", payload)

    def _handle_command(self, message: Message):
        handler = self.command_handlers.get(message.topic.replace("command.", ""))
        if handler:
            try:
                handler(message.payload)
            except Exception as e:
                logging.error(f"Command handler error: {e}")

    def _handle_query(self, message: Message):
        handler = self.query_handlers.get(message.topic.replace("query.", ""))
        if handler:
            try:
                result = handler(message.payload)
                if message.reply_to:
                    self.command_bus.publish(message.reply_to, result, msg_type=MessageType.RESPONSE,
                                            correlation_id=message.correlation_id)
            except Exception as e:
                logging.error(f"Query handler error: {e}")

class MessageRouter:
    def __init__(self):
        self.routes: Dict[str, List[Callable[[Message], bool]]] = defaultdict(list)
        self.default_route: Optional[Callable[[Message], bool]] = None

    def add_route(self, pattern: str, filter_fn: Callable[[Message], bool]):
        self.routes[pattern].append(filter_fn)

    def set_default_route(self, filter_fn: Callable[[Message], bool]):
        self.default_route = filter_fn

    def route(self, message: Message) -> List[str]:
        destinations = []
        for pattern, filters in self.routes.items():
            if self._match(pattern, message.topic):
                for f in filters:
                    if f(message):
                        destinations.append(pattern)
                        break
        if not destinations and self.default_route:
            if self.default_route(message):
                destinations.append("default")
        return destinations

    def _match(self, pattern: str, topic: str) -> bool:
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)

class DeadLetterQueue:
    def __init__(self, broker: MessageBroker, dlq_topic: str = "dead_letter"):
        self.broker = broker
        self.dlq_topic = dlq_topic
        self.failed_messages: List[Tuple[Message, str, int]] = []
        self.max_retries = 3

    def handle_failure(self, message: Message, error: str, retry_count: int):
        if retry_count >= self.max_retries:
            self.failed_messages.append((message, error, retry_count))
            self.broker.publish(self.dlq_topic, {
                "original_message": message.__dict__,
                "error": error,
                "retry_count": retry_count,
                "failed_at": time.time()
            })
        else:
            time.sleep(2 ** retry_count)
            self.broker.publish(message.topic, message.payload, **{
                k: v for k, v in message.__dict__.items()
                if k not in ['id', 'timestamp']
            })

    def get_failed_messages(self) -> List[Tuple[Message, str, int]]:
        return self.failed_messages.copy()

    def replay(self, index: int) -> bool:
        if 0 <= index < len(self.failed_messages):
            message, _, _ = self.failed_messages[index]
            self.broker.publish(message.topic, message.payload)
            return True
        return False

def create_message_broker(config: BrokerConfig = None) -> MessageBroker:
    return MessageBroker(config)

def create_distributed_bus(node_id: str, cluster_nodes: List[str] = None) -> DistributedMessageBus:
    return DistributedMessageBus(node_id, cluster_nodes)
