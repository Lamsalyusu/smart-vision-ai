from collections import defaultdict
from dataclasses import dataclass
import math
import threading
from typing import Generic, TypeVar


Payload = TypeVar("Payload")


@dataclass(frozen=True, slots=True)
class InferenceRequest(Generic[Payload]):
    request_id: str
    model_name: str
    payload: Payload
    priority: int
    enqueued_at_seconds: float


class PriorityBatcher(Generic[Payload]):
    """Group ready requests by model, honoring priority and maximum queue delay."""

    def __init__(self, max_batch_size: int, max_queue_delay_seconds: float, max_queue_size: int) -> None:
        if max_batch_size < 1 or max_queue_size < max_batch_size:
            raise ValueError("queue size must be at least the positive maximum batch size")
        if not math.isfinite(max_queue_delay_seconds) or max_queue_delay_seconds < 0:
            raise ValueError("maximum queue delay must be finite and non-negative")
        self._max_batch_size = max_batch_size
        self._max_queue_delay = max_queue_delay_seconds
        self._max_queue_size = max_queue_size
        self._queues: dict[str, list[InferenceRequest[Payload]]] = defaultdict(list)
        self._request_ids: set[str] = set()
        self._queue_length = 0
        self._lock = threading.Lock()

    @property
    def queue_length(self) -> int:
        with self._lock:
            return self._queue_length

    def enqueue(self, request: InferenceRequest[Payload]) -> None:
        if not request.request_id or not request.model_name:
            raise ValueError("request id and model name are required")
        if not math.isfinite(request.enqueued_at_seconds) or request.enqueued_at_seconds < 0:
            raise ValueError("enqueue timestamp must be finite and non-negative")
        with self._lock:
            if request.request_id in self._request_ids:
                raise ValueError("request id is already queued")
            if self._queue_length >= self._max_queue_size:
                raise OverflowError("inference queue is full")
            self._queues[request.model_name].append(request)
            self._request_ids.add(request.request_id)
            self._queue_length += 1

    def take_ready_batch(
        self,
        now_seconds: float,
        *,
        force: bool = False,
    ) -> tuple[InferenceRequest[Payload], ...] | None:
        if not math.isfinite(now_seconds) or now_seconds < 0:
            raise ValueError("current time must be finite and non-negative")
        with self._lock:
            eligible = []
            for model_name, requests in self._queues.items():
                if not requests:
                    continue
                ordered = sorted(requests, key=lambda item: (-item.priority, item.enqueued_at_seconds))
                oldest_age = max(0.0, now_seconds - min(item.enqueued_at_seconds for item in requests))
                if force or len(requests) >= self._max_batch_size or oldest_age >= self._max_queue_delay:
                    eligible.append((ordered[0].priority, -min(item.enqueued_at_seconds for item in requests), model_name, ordered))
            if not eligible:
                return None

            _, _, model_name, ordered = max(eligible, key=lambda item: (item[0], item[1]))
            batch = tuple(ordered[: self._max_batch_size])
            selected_ids = {request.request_id for request in batch}
            self._queues[model_name] = [
                request for request in self._queues[model_name]
                if request.request_id not in selected_ids
            ]
            self._request_ids.difference_update(selected_ids)
            self._queue_length -= len(batch)
            return batch