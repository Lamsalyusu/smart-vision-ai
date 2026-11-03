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
import asyncio
import heapq
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager


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


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass(frozen=True, slots=True)
class QueueMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    scheduled_at: Optional[float] = None
    batch_id: Optional[str] = None

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def is_scheduled(self) -> bool:
        if self.scheduled_at is None:
            return False
        return time.time() < self.scheduled_at


@dataclass(frozen=True, slots=True)
class QueueStats:
    queue_name: str
    size: int
    max_size: int
    enqueued_total: int
    dequeued_total: int
    rejected_total: int
    expired_total: int
    avg_wait_time: float
    avg_processing_time: float
    throughput: float


class BaseQueue(ABC):
    def __init__(self, name: str, max_size: int = 0):
        self.name = name
        self.max_size = max_size
        self.stats = QueueStats(
            queue_name=name,
            size=0,
            max_size=max_size,
            enqueued_total=0,
            dequeued_total=0,
            rejected_total=0,
            expired_total=0,
            avg_wait_time=0.0,
            avg_processing_time=0.0,
            throughput=0.0
        )
        self.lock = threading.RLock()
        self._waiting: Dict[str, threading.Event] = {}

    @abstractmethod
    def enqueue(self, message: QueueMessage) -> bool:
        pass

    @abstractmethod
    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        pass

    @abstractmethod
    def peek(self) -> Optional[QueueMessage]:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def clear(self):
        pass

    def wait_for_message(self, timeout: float = None) -> bool:
        event = threading.Event()
        with self.lock:
            waiter_id = str(uuid.uuid4())
            self._waiting[waiter_id] = event

        result = event.wait(timeout)

        with self.lock:
            self._waiting.pop(waiter_id, None)

        return result

    def _notify_waiters(self):
        for event in self._waiting.values():
            event.set()
        self._waiting.clear()


class FIFOQueue(BaseQueue):
    def __init__(self, name: str, max_size: int = 0):
        super().__init__(name, max_size)
        self._queue: deque = deque()

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if self.max_size > 0 and len(self._queue) >= self.max_size:
                self.stats.rejected_total += 1
                return False

            if message.is_expired():
                self.stats.expired_total += 1
                return False

            if message.is_scheduled():
                return self._schedule_message(message)

            self._queue.append(message)
            self.stats.size = len(self._queue)
            self.stats.enqueued_total += 1
            self._notify_waiters()
            return True

    def _schedule_message(self, message: QueueMessage) -> bool:
        def delayed_enqueue():
            time.sleep(message.scheduled_at - time.time())
            self.enqueue(message)

        threading.Thread(target=delayed_enqueue, daemon=True).start()
        return True

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            if not self._queue:
                if timeout is None:
                    return None
                return self._wait_and_dequeue(timeout)

            message = self._queue.popleft()
            self.stats.size = len(self._queue)
            self.stats.dequeued_total += 1
            self.stats.avg_wait_time = time.time() - message.timestamp
            return message

    def _wait_and_dequeue(self, timeout: float) -> Optional[QueueMessage]:
        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return None

            with self.lock:
                if self._queue:
                    message = self._queue.popleft()
                    self.stats.size = len(self._queue)
                    self.stats.dequeued_total += 1
                    self.stats.avg_wait_time = time.time() - message.timestamp
                    return message

            time.sleep(0.01)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            return self._queue[0] if self._queue else None

    def size(self) -> int:
        with self.lock:
            return len(self._queue)

    def clear(self):
        with self.lock:
            self._queue.clear()
            self.stats.size = 0


class PriorityQueue(BaseQueue):
    def __init__(self, name: str, max_size: int = 0):
        super().__init__(name, max_size)
        self._heap: List[Tuple[int, float, QueueMessage]] = []

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if self.max_size > 0 and len(self._heap) >= self.max_size:
                self.stats.rejected_total += 1
                return False

            if message.is_expired():
                self.stats.expired_total += 1
                return False

            if message.is_scheduled():
                return self._schedule_message(message)

            priority = -message.priority.value
            heapq.heappush(self._heap, (priority, message.timestamp, message))
            self.stats.size = len(self._heap)
            self.stats.enqueued_total += 1
            self._notify_waiters()
            return True

    def _schedule_message(self, message: QueueMessage) -> bool:
        def delayed_enqueue():
            time.sleep(message.scheduled_at - time.time())
            self.enqueue(message)

        threading.Thread(target=delayed_enqueue, daemon=True).start()
        return True

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            if not self._heap:
                if timeout is None:
                    return None
                return self._wait_and_dequeue(timeout)

            _, _, message = heapq.heappop(self._heap)
            self.stats.size = len(self._heap)
            self.stats.dequeued_total += 1
            self.stats.avg_wait_time = time.time() - message.timestamp
            return message

    def _wait_and_dequeue(self, timeout: float) -> Optional[QueueMessage]:
        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return None

            with self.lock:
                if self._heap:
                    _, _, message = heapq.heappop(self._heap)
                    self.stats.size = len(self._heap)
                    self.stats.dequeued_total += 1
                    self.stats.avg_wait_time = time.time() - message.timestamp
                    return message

            time.sleep(0.01)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            return self._heap[0][2] if self._heap else None

    def size(self) -> int:
        with self.lock:
            return len(self._heap)

    def clear(self):
        with self.lock:
            self._heap.clear()
            self.stats.size = 0


class DelayQueue(BaseQueue):
    def __init__(self, name: str, max_size: int = 0):
        super().__init__(name, max_size)
        self._scheduled: List[Tuple[float, QueueMessage]] = []
        self._ready: deque = deque()
        self._scheduler_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

    def _scheduler(self):
        while True:
            time.sleep(0.1)
            now = time.time()
            with self.lock:
                ready = []
                for i, (scheduled_at, msg) in enumerate(self._scheduled):
                    if scheduled_at <= now:
                        ready.append((i, msg))

                for i, msg in reversed(ready):
                    self._scheduled.pop(i)
                    self._ready.append(msg)
                    self.stats.enqueued_total += 1
                    self._notify_waiters()

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if self.max_size > 0 and len(self._scheduled) + len(self._ready) >= self.max_size:
                self.stats.rejected_total += 1
                return False

            if message.is_expired():
                self.stats.expired_total += 1
                return False

            if message.scheduled_at:
                heapq.heappush(self._scheduled, (message.scheduled_at, message))
            else:
                self._ready.append(message)
                self.stats.enqueued_total += 1
                self._notify_waiters()
            return True

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            if not self._ready:
                if timeout is None:
                    return None
                return self._wait_and_dequeue(timeout)

            message = self._ready.popleft()
            self.stats.size = len(self._ready) + len(self._scheduled)
            self.stats.dequeued_total += 1
            self.stats.avg_wait_time = time.time() - message.timestamp
            return message

    def _wait_and_dequeue(self, timeout: float) -> Optional[QueueMessage]:
        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return None

            with self.lock:
                if self._ready:
                    message = self._ready.popleft()
                    self.stats.dequeued_total += 1
                    self.stats.avg_wait_time = time.time() - message.timestamp
                    return message

            time.sleep(0.01)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            return self._ready[0] if self._ready else None

    def size(self) -> int:
        with self.lock:
            return len(self._ready) + len(self._scheduled)

    def clear(self):
        with self.lock:
            self._ready.clear()
            self._scheduled.clear()


class BatchQueue(BaseQueue):
    def __init__(
        self,
        name: str,
        batch_size: int = 10,
        max_wait_time: float = 1.0,
        max_size: int = 0
    ):
        super().__init__(name, max_size)
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self._buffer: List[QueueMessage] = []
        self._last_flush = time.time()

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if self.max_size > 0 and len(self._buffer) >= self.max_size:
                self.stats.rejected_total += 1
                return False

            if message.is_expired():
                self.stats.expired_total += 1
                return False

            self._buffer.append(message)
            self.stats.enqueued_total += 1

            if len(self._buffer) >= self.batch_size:
                self._flush_buffer()

            return True

    def _flush_buffer(self):
        if self._buffer:
            pass

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            if not self._buffer:
                if timeout is None:
                    return None
                start = time.time()
                while time.time() - start < timeout:
                    if self._buffer:
                        return self._buffer.pop(0)
                    time.sleep(0.01)
                return None

            return self._buffer.pop(0)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            return self._buffer[0] if self._buffer else None

    def size(self) -> int:
        with self.lock:
            return len(self._buffer)

    def clear(self):
        with self.lock:
            self._buffer.clear()


class CircularQueue(BaseQueue):
    def __init__(self, name: str, max_size: int):
        super().__init__(name, max_size)
        self._buffer = [None] * max_size
        self.head = 0
        self.tail = 0
        self.count = 0

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if message.is_expired():
                self.stats.expired_total += 1
                return False

            self._buffer[self.tail] = message
            self.tail = (self.tail + 1) % self.max_size
            if self.count < self.max_size:
                self.count += 1
            else:
                self.head = (self.head + 1) % self.max_size

            self.stats.size = self.count
            self.stats.enqueued_total += 1
            self._notify_waiters()
            return True

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            if self.count == 0:
                if timeout is None:
                    return None
                return self._wait_and_dequeue(timeout)

            message = self._buffer[self.head]
            self._buffer[self.head] = None
            self.head = (self.head + 1) % self.max_size
            self.count -= 1
            self.stats.size = self.count
            self.stats.dequeued_total += 1
            self.stats.avg_wait_time = time.time() - message.timestamp
            return message

    def _wait_and_dequeue(self, timeout: float) -> Optional[QueueMessage]:
        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return None

            with self.lock:
                if self.count > 0:
                    return self.dequeue()

            time.sleep(0.01)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            if self.count == 0:
                return None
            return self._buffer[self.head]

    def size(self) -> int:
        with self.lock:
            return self.count

    def clear(self):
        with self.lock:
            self._buffer = [None] * self.max_size
            self.head = 0
            self.tail = 0
            self.count = 0
            self.stats.size = 0


class WeightedQueue(BaseQueue):
    def __init__(self, name: str, max_size: int = 0):
        super().__init__(name, max_size)
        self._queues: Dict[MessagePriority, deque] = {
            p: deque() for p in MessagePriority
        }

    def enqueue(self, message: QueueMessage) -> bool:
        with self.lock:
            if self.max_size > 0 and self.size() >= self.max_size:
                self.stats.rejected_total += 1
                return False

            if message.is_expired():
                self.stats.expired_total += 1
                return False

            if message.is_scheduled():
                return self._schedule_message(message)

            self._queues[message.priority].append(message)
            self.stats.enqueued_total += 1
            self._notify_waiters()
            return True

    def _schedule_message(self, message: QueueMessage) -> bool:
        def delayed_enqueue():
            time.sleep(message.scheduled_at - time.time())
            self.enqueue(message)

        threading.Thread(target=delayed_enqueue, daemon=True).start()
        return True

    def dequeue(self, timeout: float = None) -> Optional[QueueMessage]:
        with self.lock:
            for priority in sorted(MessagePriority, reverse=True):
                queue = self._queues[priority]
                if queue:
                    message = queue.popleft()
                    self.stats.dequeued_total += 1
                    self.stats.avg_wait_time = time.time() - message.timestamp
                    return message

            if timeout is None:
                return None
            return self._wait_and_dequeue(timeout)

    def _wait_and_dequeue(self, timeout: float) -> Optional[QueueMessage]:
        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                return None

            with self.lock:
                for priority in sorted(MessagePriority, reverse=True):
                    queue = self._queues[priority]
                    if queue:
                        message = queue.popleft()
                        self.stats.dequeued_total += 1
                        self.stats.avg_wait_time = time.time() - message.timestamp
                        return message

            time.sleep(0.01)

    def peek(self) -> Optional[QueueMessage]:
        with self.lock:
            for priority in sorted(MessagePriority, reverse=True):
                queue = self._queues[priority]
                if queue:
                    return queue[0]
            return None

    def size(self) -> int:
        with self.lock:
            return sum(len(q) for q in self._queues.values())

    def clear(self):
        with self.lock:
            for q in self._queues.values():
                q.clear()


class QueueManager:
    def __init__(self):
        self.queues: Dict[str, BaseQueue] = {}

    def create_queue(
        self,
        name: str,
        queue_type: QueueType = QueueType.FIFO,
        max_size: int = 0,
        **kwargs
    ) -> BaseQueue:
        if queue_type == QueueType.FIFO:
            queue = FIFOQueue(name, max_size)
        elif queue_type == QueueType.LIFO:
            queue = FIFOQueue(name, max_size)
        elif queue_type == QueueType.PRIORITY:
            queue = PriorityQueue(name, max_size)
        elif queue_type == QueueType.DELAY:
            queue = DelayQueue(name, max_size)
        elif queue_type == QueueType.BATCH:
            queue = BatchQueue(name, kwargs.get('batch_size', 10), kwargs.get('max_wait_time', 1.0), max_size)
        elif queue_type == QueueType.CIRCULAR:
            queue = CircularQueue(name, max_size)
        elif queue_type == QueueType.WEIGHTED:
            queue = WeightedQueue(name, max_size)
        else:
            queue = FIFOQueue(name, max_size)

        self.queues[name] = queue
        return queue

    def get_queue(self, name: str) -> Optional[BaseQueue]:
        return self.queues.get(name)

    def delete_queue(self, name: str) -> bool:
        if name in self.queues:
            self.queues[name].clear()
            del self.queues[name]
            return True
        return False

    def get_all_stats(self) -> Dict[str, QueueStats]:
        return {name: queue.stats for name, queue in self.queues.items()}


class QueueMetrics:
    def __init__(self):
        self.latencies: List[float] = []
        self.throughput_samples: List[float] = []
        self.lock = threading.Lock()

    def record_enqueue(self, latency: float):
        with self.lock:
            self.latencies.append(latency)

    def record_dequeue(self, latency: float):
        with self.lock:
            self.latencies.append(latency)

    def record_throughput(self, throughput: float):
        with self.lock:
            self.throughput_samples.append(throughput)

    def get_stats(self) -> Dict[str, float]:
        with self.lock:
            return {
                'avg_latency': np.mean(self.latencies) if self.latencies else 0,
                'p50_latency': np.percentile(self.latencies, 50) if self.latencies else 0,
                'p95_latency': np.percentile(self.latencies, 95) if self.latencies else 0,
                'p99_latency': np.percentile(self.latencies, 99) if self.latencies else 0,
                'avg_throughput': np.mean(self.throughput_samples) if self.throughput_samples else 0
            }


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
