from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class MathTaskWorkerPool:
    """ThreadPool executor offloading heavy mathematical algorithms from the async event loop."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cv-math-worker")

    async def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run CPU-intensive function in worker thread pool without blocking event loop."""
        loop = asyncio.get_running_loop()
        if kwargs:
            from functools import partial
            target_func = partial(func, **kwargs)
            return await loop.run_in_executor(self._executor, target_func, *args)
        return await loop.run_in_executor(self._executor, func, *args)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


worker_pool = MathTaskWorkerPool()
