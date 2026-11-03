from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict, deque
import threading
import queue
import uuid
import json
import hashlib
import copy


class QueueType(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"
    DELAY = "delay"
    SCHEDULED = "scheduled"
    BATCH = "batch"
    DEAD_LETTER = "dead_letter"
    CIRCULAR = "circular"
    WEIGHTED = "weighted"


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    QUERY = "query"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ACK = "ack"


class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.EVENT
    topic: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[float] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    retry_count: int = 0
    max_retries: int = 3
    source: str = ""
    destination: str = ""

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'topic': self.topic,
            'payload': self.payload,
            'headers': self.headers,
            'timestamp': self.timestamp,
            'priority': self.priority.value,
            'ttl': self.ttl,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'delivery_guarantee': self.delivery_guarantee.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'source': self.source,
            'destination': self.destination
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=MessageType(data.get('type', 'event')),
            topic=data.get('topic', ''),
            payload=data.get('payload'),
            headers=data.get('headers', {}),
            timestamp=data.get('timestamp', time.time()),
            priority=MessagePriority(data.get('priority', 1)),
            ttl=data.get('ttl'),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            delivery_guarantee=DeliveryGuarantee(data.get('delivery_guarantee', 'at_least_once')),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            source=data.get('source', ''),
            destination=data.get('destination', '')
        )


@dataclass(frozen=True, slots=True)
class Subscription:
    id: str
    topic: str
    callback: Callable[[Message], None]
    filter: Optional[Callable[[Message], bool]] = None
    created_at: float = field(default_factory=time.time)
    active: bool = True


class MessageSerializer(ABC):
    @abstractmethod
    def serialize(self, message: Message) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Message:
        pass


class JSONSerializer(MessageSerializer):
    def serialize(self, message: Message) -> bytes:
        return json.dumps(message.to_dict()).encode('utf-8')

    def deserialize(self, data: bytes) -> Message:
        return Message.from_dict(json.loads(data.decode('utf-8')))


class MessageBroker:
    def __init__(
        self,
        serializer: MessageSerializer = None,
        max_queue_size: int = 10000,
        cleanup_interval: float = 60.0
    ):
        self.serializer = serializer or JSONSerializer()
        self.max_queue_size = max_queue_size
        self.cleanup_interval = cleanup_interval

        self.topics: Dict[str, queue.PriorityQueue] = defaultdict(lambda: queue.PriorityQueue(maxsize=max_queue_size))
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.dead_letter_queue: queue.Queue = queue.Queue()
        self.message_history: List[Message] = []
        self.max_history = 10000

        self.lock = threading.RLock()
        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def stop(self):
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)

    def publish(self, message: Message) -> bool:
        if message.is_expired():
            return False

        with self.lock:
            if message.topic not in self.topics:
                self.topics[message.topic] = queue.PriorityQueue(maxsize=self.max_queue_size)

            try:
                priority = -message.priority.value
                self.topics[message.topic].put((priority, time.time(), message), timeout=0.1)

                self._notify_subscribers(message)
                self._add_to_history(message)

                return True
            except queue.Full:
                self._send_to_dead_letter(message, "Queue full")
                return False

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Message], None],
        filter: Optional[Callable[[Message], bool]] = None
    ) -> str:
        sub_id = str(uuid.uuid4())
        subscription = Subscription(
            id=sub_id,
            topic=topic,
            callback=callback,
            filter=filter
        )

        with self.lock:
            self.subscriptions[topic].append(subscription)

        return sub_id

    def unsubscribe(self, topic: str, subscription_id: str) -> bool:
        with self.lock:
            if topic in self.subscriptions:
                self.subscriptions[topic] = [
                    s for s in self.subscriptions[topic] if s.id != subscription_id
                ]
                return True
        return False

    def consume(self, topic: str, timeout: float = 1.0) -> Optional[Message]:
        with self.lock:
            if topic not in self.topics:
                return None

            try:
                _, _, message = self.topics[topic].get(timeout=timeout)
                return message
            except queue.Empty:
                return None

    def _notify_subscribers(self, message: Message):
        if message.topic in self.subscriptions:
            for sub in self.subscriptions[message.topic]:
                if sub.active and (sub.filter is None or sub.filter(message)):
                    try:
                        sub.callback(message)
                    except Exception:
                        pass

    def _add_to_history(self, message: Message):
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]

    def _send_to_dead_letter(self, message: Message, reason: str):
        try:
            message.headers['dlq_reason'] = reason
            message.headers['dlq_timestamp'] = str(time.time())
            self.dead_letter_queue.put(message, timeout=0.1)
        except queue.Full:
            pass

    def _cleanup_loop(self):
        while self.running:
            time.sleep(self.cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self):
        with self.lock:
            for topic, q in self.topics.items():
                valid_messages = []
                while not q.empty():
                    try:
                        priority, timestamp, message = q.get_nowait()
                        if not message.is_expired():
                            valid_messages.append((priority, timestamp, message))
                    except queue.Empty:
                        break

                for msg in valid_messages:
                    try:
                        q.put(msg, timeout=0.1)
                    except queue.Full:
                        self._send_to_dead_letter(msg[2], "Cleanup queue full")

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'topics': {topic: q.qsize() for topic, q in self.topics.items()},
                'subscriptions': {topic: len(subs) for topic, subs in self.subscriptions.items()},
                'dead_letter_size': self.dead_letter_queue.qsize(),
                'history_size': len(self.message_history)
            }


class DistributedMessageBroker:
    def __init__(
        self,
        node_id: str,
        peers: List[str],
        serializer: MessageSerializer = None
    ):
        self.node_id = node_id
        self.peers = peers
        self.local_broker = MessageBroker(serializer)
        self.remote_brokers: Dict[str, MessageBroker] = {}
        self.routing_table: Dict[str, str] = {}

        for peer in peers:
            self.routing_table[peer] = peer

    def start(self):
        self.local_broker.start()

    def stop(self):
        self.local_broker.stop()

    def publish(self, message: Message) -> bool:
        message.source = self.node_id

        if message.destination and message.destination in self.peers:
            return self._publish_remote(message)

        return self.local_broker.publish(message)

    def _publish_remote(self, message: Message) -> bool:
        try:
            data = self.local_broker.serializer.serialize(message)
            return True
        except Exception:
            return False

    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> str:
        return self.local_broker.subscribe(topic, callback)

    def register_peer(self, peer_id: str, broker: MessageBroker):
        self.peers.append(peer_id)
        self.remote_brokers[peer_id] = broker
        self.routing_table[peer_id] = peer_id


class RequestResponseClient:
    def __init__(
        self,
        broker: MessageBroker,
        default_timeout: float = 30.0
    ):
        self.broker = broker
        self.default_timeout = default_timeout
        self.pending_requests: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def request(
        self,
        topic: str,
        payload: Any,
        timeout: Optional[float] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Optional[Message]:
        correlation_id = str(uuid.uuid4())
        reply_topic = f"reply.{correlation_id}"

        response_queue = queue.Queue(maxsize=1)
        with self.lock:
            self.pending_requests[correlation_id] = response_queue

        def response_handler(msg: Message):
            try:
                response_queue.put(msg, timeout=0.1)
            except queue.Full:
                pass

        sub_id = self.broker.subscribe(reply_topic, response_handler)

        request = Message(
            type=MessageType.REQUEST,
            topic=topic,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=reply_topic,
            priority=priority,
            delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE
        )

        self.broker.publish(request)

        try:
            response = response_queue.get(timeout=timeout or self.default_timeout)
            return response
        except queue.Empty:
            return None
        finally:
            self.broker.unsubscribe(reply_topic, sub_id)
            with self.lock:
                self.pending_requests.pop(correlation_id, None)


class EventPublisher:
    def __init__(self, broker: MessageBroker, source: str):
        self.broker = broker
        self.source = source

    def publish_event(self, topic: str, payload: Any, **kwargs) -> bool:
        message = Message(
            type=MessageType.EVENT,
            topic=topic,
            payload=payload,
            source=self.source,
            **kwargs
        )
        return self.broker.publish(message)

    def publish_notification(self, topic: str, payload: Any, **kwargs) -> bool:
        message = Message(
            type=MessageType.NOTIFICATION,
            topic=topic,
            payload=payload,
            source=self.source,
            **kwargs
        )
        return self.broker.publish(message)


class CommandDispatcher:
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.handlers: Dict[str, Callable[[Message], Any]] = {}

    def register_handler(self, command: str, handler: Callable[[Message], Any]):
        self.handlers[command] = handler

        def wrapper(msg: Message):
            try:
                result = handler(msg)
                if msg.reply_to:
                    response = Message(
                        type=MessageType.RESPONSE,
                        topic=msg.reply_to,
                        payload=result,
                        correlation_id=msg.correlation_id,
                        source=self.broker.node_id if hasattr(self.broker, 'node_id') else ''
                    )
                    self.broker.publish(response)
            except Exception as e:
                if msg.reply_to:
                    error_response = Message(
                        type=MessageType.RESPONSE,
                        topic=msg.reply_to,
                        payload={'error': str(e)},
                        correlation_id=msg.correlation_id,
                        source=self.broker.node_id if hasattr(self.broker, 'node_id') else ''
                    )
                    self.broker.publish(error_response)

        self.broker.subscribe(f"command.{command}", wrapper)

    def send_command(self, command: str, payload: Any, **kwargs) -> bool:
        message = Message(
            type=MessageType.COMMAND,
            topic=f"command.{command}",
            payload=payload,
            **kwargs
        )
        return self.broker.publish(message)


class MessageRouter:
    def __init__(self):
        self.routes: Dict[str, List[Callable[[Message], bool]]] = defaultdict(list)

    def add_route(self, pattern: str, filter: Callable[[Message], bool]):
        self.routes[pattern].append(filter)

    def route(self, message: Message) -> List[str]:
        matched = []
        for pattern, filters in self.routes.items():
            if self._match_pattern(message.topic, pattern):
                for f in filters:
                    if f(message):
                        matched.append(pattern)
                        break
        return matched

    def _match_pattern(self, topic: str, pattern: str) -> bool:
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return topic.startswith(prefix)
        return topic == pattern


class MessageMetrics:
    def __init__(self):
        self.published = 0
        self.delivered = 0
        self.failed = 0
        self.latencies: List[float] = []
        self.lock = threading.Lock()

    def record_publish(self):
        with self.lock:
            self.published += 1

    def record_delivery(self, latency: float):
        with self.lock:
            self.delivered += 1
            self.latencies.append(latency)
            if len(self.latencies) > 10000:
                self.latencies = self.latencies[-10000:]

    def record_failure(self):
        with self.lock:
            self.failed += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'published': self.published,
                'delivered': self.delivered,
                'failed': self.failed,
                'success_rate': self.delivered / max(self.published, 1),
                'avg_latency': np.mean(self.latencies) if self.latencies else 0,
                'p50_latency': np.percentile(self.latencies, 50) if self.latencies else 0,
                'p95_latency': np.percentile(self.latencies, 95) if self.latencies else 0,
                'p99_latency': np.percentile(self.latencies, 99) if self.latencies else 0
            }


class MessageStore:
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.messages: Dict[str, Message] = {}
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.Lock()

    def store(self, message: Message):
        with self.lock:
            if len(self.messages) >= self.max_size:
                oldest = min(self.messages.items(), key=lambda x: x[1].timestamp)
                del self.messages[oldest[0]]
                for topic, ids in self.index.items():
                    if oldest[0] in ids:
                        ids.remove(oldest[0])

            self.messages[message.id] = message
            self.index[message.topic].append(message.id)

    def retrieve(self, message_id: str) -> Optional[Message]:
        with self.lock:
            return self.messages.get(message_id)

    def query_by_topic(self, topic: str, limit: int = 100) -> List[Message]:
        with self.lock:
            ids = self.index.get(topic, [])[-limit:]
            return [self.messages[id] for id in ids if id in self.messages]

    def query_by_time_range(self, start: float, end: float, limit: int = 100) -> List[Message]:
        with self.lock:
            results = []
            for msg in self.messages.values():
                if start <= msg.timestamp <= end:
                    results.append(msg)
                    if len(results) >= limit:
                        break
            return sorted(results, key=lambda x: x.timestamp)


class DeadLetterProcessor:
    def __init__(self, broker: MessageBroker, processor: Callable[[Message], bool]):
        self.broker = broker
        self.processor = processor
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _process_loop(self):
        while self.running:
            try:
                message = self.broker.dead_letter_queue.get(timeout=1.0)
                try:
                    if self.processor(message):
                        self.broker.publish(message)
                except Exception:
                    pass
            except queue.Empty:
                continue


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
