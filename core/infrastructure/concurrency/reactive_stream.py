from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Generic, List, Optional, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class ReactiveStreamPipeline(Generic[TIn]):
    """Asynchronous Reactive Stream with bounded ring-buffer, windowed batching, and backpressure."""

    def __init__(self, capacity: int = 100) -> None:
        self._queue: asyncio.Queue[TIn] = asyncio.Queue(maxsize=capacity)
        self._is_active = True

    async def emit(self, item: TIn) -> bool:
        if not self._is_active:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            # Drop oldest item to apply backpressure flow control
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            await self._queue.put(item)
            return True

    async def batch_window(self, max_batch_size: int = 10, max_wait_seconds: float = 0.05) -> List[TIn]:
        """Aggregate stream elements within batch size or time window."""
        batch: List[TIn] = []
        try:
            # Wait for at least one item
            first = await asyncio.wait_for(self._queue.get(), timeout=max_wait_seconds)
            batch.append(first)
            self._queue.task_done()

            # Drain any remaining available items up to batch size
            while len(batch) < max_batch_size:
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
        except asyncio.TimeoutError:
            pass
        return batch

    def close(self) -> None:
        self._is_active = False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
