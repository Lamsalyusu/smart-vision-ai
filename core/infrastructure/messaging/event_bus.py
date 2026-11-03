from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict
import json


class EventType(Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    TRACKING = "tracking"
    ANOMALY = "anomaly"
    STATE_CHANGE = "state_change"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class Priority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    event_type: EventType
    source: str
    timestamp: float = field(default_factory=time.time)
    payload: Any = None
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class EventSubscription:
    subscription_id: str
    event_type: EventType
    callback: Callable[[Any], None]
    filter: Optional[Callable[[Any], bool]] = None
    created_at: float = field(default_factory=time.time)


class EventBus:
    def __init__(self):
        self.subscriptions: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=10000)
        self.lock = threading.RLock()

    def publish(self, event: Event) -> bool:
        with self.lock:
            self.event_history.append(event)
            handlers = self.subscriptions.get(event.event_type, [])
            for sub in handlers:
                if sub.filter is None or sub.filter(event):
                    try:
                        sub.callback(event)
                    except Exception:
                        pass
            return True

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Any], None],
        filter_fn: Optional[Callable[[Any], bool]] = None
    ) -> str:
        sub_id = f"sub_{int(time.time()*1000000)}"
        sub = EventSubscription(
            subscription_id=sub_id,
            event_type=event_type,
            callback=callback,
            filter=filter_fn
        )
        with self.lock:
            self.subscriptions[event_type].append(sub)
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        with self.lock:
            for event_type, subs in self.subscriptions.items():
                for i, sub in enumerate(subs):
                    if sub.subscription_id == subscription_id:
                        subs.pop(i)
                        return True
        return False

    def get_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List:
        with self.lock:
            if event_type:
                return [e for e in self.event_history if e.event_type == event_type][-limit:]
            return list(self.event_history)[-limit:]


class MessageBroker:
    def __init__(self):
        self.topics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)
        self.lock = threading.RLock()

    def publish(self, topic: str, message: Any) -> bool:
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = deque(maxlen=10000)
            try:
                self.topics[topic].append({
                    'data': message,
                    'timestamp': time.time()
                })
                for callback in self.subscriptions.get(topic, []):
                    try:
                        callback(message)
                    except Exception:
                        pass
                return True
            except Exception:
                return False

    def subscribe(self, topic: str, callback: Callable) -> str:
        sub_id = f"sub_{int(time.time()*1000000)}"
        with self.lock:
            self.subscriptions[topic].append(callback)
        return sub_id

    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        with self.lock:
            if topic in self.subscriptions:
                try:
                    self.subscriptions[topic].remove(callback)
                    return True
                except ValueError:
                    pass
        return False

    def get_messages(self, topic: str, limit: int = 100) -> List[Any]:
        with self.lock:
            if topic not in self.topics:
                return []
            return list(self.topics[topic])[-limit:]


class PubSubManager:
    def __init__(self):
        self.brokers: Dict[str, MessageBroker] = {}
        self.default_broker = MessageBroker()

    def create_broker(self, name: str) -> MessageBroker:
        broker = MessageBroker()
        self.brokers[name] = broker
        return broker

    def get_broker(self, name: str) -> Optional[MessageBroker]:
        return self.brokers.get(name)

    def publish(self, topic: str, message: Any, broker_name: str = None) -> bool:
        broker = self.brokers.get(broker_name) if broker_name else self.default_broker
        if broker:
            return broker.publish(topic, message)
        return False

    def subscribe(self, topic: str, callback: Callable, broker_name: str = None) -> str:
        broker = self.brokers.get(broker_name) if broker_name else self.default_broker
        if broker:
            return broker.subscribe(topic, callback)
        return ""

    def unsubscribe(self, topic: str, callback: Callable, broker_name: str = None) -> bool:
        broker = self.brokers.get(broker_name) if broker_name else self.default_broker
        if broker:
            return broker.unsubscribe(topic, callback)
        return False


class RequestResponseManager:
    def __init__(self, message_bus: MessageBroker):
        self.message_bus = message_bus
        self.pending_requests: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def request(
        self,
        topic: str,
        payload: Any,
        timeout: float = 30.0,
        correlation_id: str = None
    ) -> Optional[Any]:
        correlation_id = correlation_id or f"req_{int(time.time()*1000000)}"
        reply_topic = f"reply.{correlation_id}"

        response_queue = queue.Queue(maxsize=1)
        with self.lock:
            self.pending_requests[correlation_id] = response_queue

        def response_handler(msg):
            try:
                response_queue.put(msg, timeout=0.1)
            except queue.Full:
                pass

        sub_id = self.message_bus.subscribe(reply_topic, response_handler)

        request = {
            'type': 'request',
            'topic': topic,
            'payload': payload,
            'correlation_id': correlation_id,
            'reply_to': reply_topic,
            'timestamp': time.time()
        }

        self.message_bus.publish(topic, request)

        try:
            response = response_queue.get(timeout=timeout)
            return response
        except queue.Empty:
            return None
        finally:
            self.message_bus.unsubscribe(reply_topic, sub_id)
            with self.lock:
                self.pending_requests.pop(correlation_id, None)


class EventSourcing:
    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self.events: List[Dict] = []
        self.snapshots: Dict[str, Dict] = {}
        self.lock = threading.RLock()

    def append_event(
        self,
        aggregate_id: str,
        event_type: str,
        payload: Dict,
        metadata: Dict = None
    ):
        event = {
            'event_id': f"evt_{int(time.time()*1000000)}",
            'aggregate_id': aggregate_id,
            'event_type': event_type,
            'payload': payload,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]

    def get_events(self, aggregate_id: str, after_timestamp: float = 0) -> List[Dict]:
        with self.lock:
            return [e for e in self.events if e['aggregate_id'] == aggregate_id and e['timestamp'] > after_timestamp]

    def create_snapshot(self, aggregate_id: str, state: Any):
        with self.lock:
            self.snapshots[aggregate_id] = {
                'state': state,
                'timestamp': time.time()
            }

    def get_snapshot(self, aggregate_id: str) -> Optional[Any]:
        with self.lock:
            return self.snapshots.get(aggregate_id, {}).get('state')

    def replay(self, aggregate_id: str) -> Any:
        events = self.get_events(aggregate_id)
        # Replay logic would depend on the aggregate type
        return None


class CQRS:
    def __init__(self):
        self.command_handlers: Dict[str, Callable] = {}
        self.query_handlers: Dict[str, Callable] = {}
        self.event_sourcing = EventSourcing()

    def register_command_handler(self, command_type: str, handler: Callable):
        self.command_handlers[command_type] = handler

    def register_query_handler(self, query_type: str, handler: Callable):
        self.query_handlers[query_type] = handler

    def execute_command(self, command_type: str, payload: Any) -> Any:
        handler = self.command_handlers.get(command_type)
        if not handler:
            raise ValueError(f"No handler for command: {command_type}")

        result = handler(payload)

        # Could emit event here
        return result

    def execute_query(self, query_type: str, payload: Any) -> Any:
        handler = self.query_handlers.get(query_type)
        if not handler:
            raise ValueError(f"No handler for query: {query_type}")
        return handler(payload)


class SagaOrchestrator:
    def __init__(self):
        self.sagas: Dict[str, Dict] = {}

    def start_saga(self, saga_id: str, steps: List[Dict]):
        self.sagas[saga_id] = {
            'steps': steps,
            'current_step': 0,
            'status': 'running',
            'compensations': [],
            'start_time': time.time()
        }

    def execute_step(self, saga_id: str, step_result: Any = None) -> bool:
        if saga_id not in self.sagas:
            return False

        saga = self.sagas[saga_id]
        if saga['status'] != 'running':
            return False

        step = saga['steps'][saga['current_step']]
        try:
            # Execute step logic here
            pass

            saga['current_step'] += 1
            if saga['current_step'] >= len(saga['steps']):
                saga['status'] = 'completed'
            return True
        except Exception:
            saga['status'] = 'failed'
            self.compensate(saga_id)
            return False

    def compensate(self, saga_id: str):
        if saga_id not in self.sagas:
            return

        saga = self.sagas[saga_id]
        saga['status'] = 'compensating'

        for comp in reversed(saga['compensations']):
            try:
                comp()
            except Exception:
                pass

        saga['status'] = 'compensated'


class OutboxPattern:
    def __init__(self):
        self.outbox: List[Dict] = []
        self.lock = threading.RLock()

    def add(self, topic: str, message: Any):
        with self.lock:
            self.outbox.append({
                'topic': topic,
                'message': message,
                'created_at': time.time(),
                'retries': 0
            })

    def process(self, publisher: Callable[[str, Any], bool], batch_size: int = 100) -> int:
        with self.lock:
            to_process = self.outbox[:batch_size]
            processed = 0

            for item in to_process:
                try:
                    if publisher(item['topic'], item['message']):
                        self.outbox.remove(item)
                        processed += 1
                    else:
                        item['retries'] += 1
                        if item['retries'] > 3:
                            self.outbox.remove(item)
                except Exception:
                    item['retries'] += 1
                    if item['retries'] > 3:
                        self.outbox.remove(item)

            return processed


class MessageSerializer:
    @staticmethod
    def serialize(message: Any, format: str = "json") -> bytes:
        if format == "json":
            return json.dumps(message).encode()
        elif format == "msgpack":
            try:
                import msgpack
                return msgpack.packb(message)
            except ImportError:
                return json.dumps(message).encode()
        elif format == "protobuf":
            try:
                import protobuf
                return message.SerializeToString()
            except ImportError:
                return json.dumps(message).encode()
        return json.dumps(message).encode()

    @staticmethod
    def deserialize(data: bytes, format: str = "json") -> Any:
        if format == "json":
            return json.loads(data.decode())
        elif format == "msgpack":
            try:
                import msgpack
                return msgpack.unpackb(data, raw=False)
            except ImportError:
                return json.loads(data.decode())
        elif format == "protobuf":
            try:
                import protobuf
                return message.ParseFromString(data)
            except ImportError:
                return json.loads(data.decode())
        return json.loads(data.decode())


class MessageRouter:
    def __init__(self):
        self.routes: Dict[str, List[Callable]] = defaultdict(list)
        self.filters: Dict[str, Callable] = {}

    def add_route(self, pattern: str, handler: Callable):
        self.routes[pattern].append(handler)

    def add_filter(self, pattern: str, filter_fn: Callable):
        self.filters[pattern] = filter_fn

    def route(self, message: Dict) -> List[Callable]:
        matched = []
        for pattern, handlers in self.routes.items():
            if self._match_pattern(message.get('topic', ''), pattern):
                for handler in handlers:
                    if message.get('topic') in self.filters:
                        if not self.filters[message['topic']](message):
                            continue
                        matched.append(handler)
        return matched

    def _match_pattern(self, topic: str, pattern: str) -> bool:
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return topic.startswith(prefix)
        return topic == pattern


class MessageMetrics:
    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)
        self.latencies: Dict[str, List[float]] = defaultdict(list)
        self.errors: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()

    def record_send(self, topic: str):
        with self.lock:
            self.counts[topic] += 1

    def record_receive(self, topic: str, latency: float):
        with self.lock:
            self.latencies[topic].append(latency)

    def record_error(self, topic: str):
        with self.lock:
            self.errors[topic] += 1

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'counts': dict(self.counts),
                'avg_latencies': {k: sum(v)/len(v) for k, v in self.latencies.items() if v},
                'errors': dict(self.errors)
            }


import threading
import queue
from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
