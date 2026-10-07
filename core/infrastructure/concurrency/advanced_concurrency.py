from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar, Set
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import weakref


T = TypeVar("T")
R = TypeVar("R")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task(Generic[T]):
    task_id: str
    func: Callable[..., T]
    args: tuple
    kwargs: dict
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[T] = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    dependencies: List[str] = field(default_factory=list)

    def __lt__(self, other: "Task") -> bool:
        return self.priority > other.priority


class TaskScheduler:
    def __init__(
        self,
        max_workers: int = None,
        max_queue_size: int = 10000,
    ):
        self.max_workers = max_workers or (threading.cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self, timeout: float = 10.0):
        self.running = False
        self.task_queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)
        self.executor.shutdown(wait=True)

    def submit(
        self,
        func: Callable[..., T],
        *args,
        priority: int = 0,
        max_retries: int = 3,
        timeout: float = 30.0,
        dependencies: List[str] = None,
        **kwargs,
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            dependencies=dependencies or [],
        )

        with self.lock:
            self.tasks[task_id] = task
            for dep in task.dependencies:
                self.dependency_graph[task_id].add(dep)
                self.reverse_deps[dep].add(task_id)

        if not task.dependencies:
            self.task_queue.put(task)
        return task_id

    def submit_batch(
        self,
        tasks: List[Dict],
    ) -> List[str]:
        return [self.submit(**t) for t in tasks]

    def get_result(self, task_id: str, timeout: float = None) -> Optional[T]:
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result

        start = time.time()
        while True:
            with self.lock:
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    if task.status == TaskStatus.COMPLETED:
                        return task.result
                    elif task.status == TaskStatus.FAILED:
                        raise task.error
                    return None

            if timeout and time.time() - start > timeout:
                return None
            time.sleep(0.01)

    def wait_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, T]:
        results = {}
        start = time.time()
        remaining = set(task_ids)

        while remaining:
            with self.lock:
                completed = remaining & set(self.completed_tasks.keys())
                for tid in completed:
                    task = self.completed_tasks[tid]
                    if task.status == TaskStatus.COMPLETED:
                        results[tid] = task.result
                    remaining.remove(tid)

            if not remaining:
                break

            if timeout and time.time() - start > timeout:
                break
            time.sleep(0.01)

        return results

    def cancel(self, task_id: str) -> bool:
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    self._remove_task(task_id)
                    return True
            return False

    def _worker_loop(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break

                with self.lock:
                    if task.status == TaskStatus.CANCELLED:
                        continue
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()

                try:
                    future = self.executor.submit(task.func, *task.args, **task.kwargs)
                    task.result = future.result(timeout=task.timeout)
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = e
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.PENDING
                        self.task_queue.put(task)
                        continue
                    task.status = TaskStatus.FAILED

                task.completed_at = time.time()

                with self.lock:
                    self.completed_tasks[task.task_id] = task
                    self._remove_task(task.task_id)
                    self._check_dependencies(task.task_id)

            except queue.Empty:
                continue
            except Exception:
                continue

    def _remove_task(self, task_id: str):
        self.tasks.pop(task_id, None)

    def _check_dependencies(self, completed_task_id: str):
        for dependent_id in self.reverse_deps.get(completed_task_id, set()):
            with self.lock:
                if dependent_id in self.tasks:
                    task = self.tasks[dependent_id]
                    task.dependencies.discard(completed_task_id)
                    if not task.dependencies:
                        self.task_queue.put(task)

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                'pending': sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                'running': sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING),
                'completed': len(self.completed_tasks),
                'queue_size': self.task_queue.qsize(),
                'active_workers': len(self.executor._threads) if hasattr(self.executor, '_threads') else 0,
            }


class AsyncTaskScheduler:
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
        self.errors: Dict[str, Exception] = {}

    async def submit(self, coro, task_id: str = None) -> str:
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        async def wrapped():
            async with self.semaphore:
                try:
                    result = await coro
                    self.results[task_id] = result
                except Exception as e:
                    self.errors[task_id] = e

        task = asyncio.create_task(wrapped())
        self.tasks[task_id] = task
        return task_id

    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        if task_id in self.results:
            return self.results[task_id]
        if task_id in self.errors:
            raise self.errors[task_id]

        if task_id in self.tasks:
            if timeout:
                await asyncio.wait_for(self.tasks[task_id], timeout=timeout)
            else:
                await self.tasks[task_id]

        if task_id in self.results:
            return self.results[task_id]
        if task_id in self.errors:
            raise self.errors[task_id]
        return None

    async def wait_all(self, task_ids: List[str], timeout: float = None) -> Dict[str, Any]:
        results = {}
        tasks = [self.tasks[tid] for tid in task_ids if tid in self.tasks]

        if timeout:
            done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
        else:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        for tid in task_ids:
            if tid in self.results:
                results[tid] = self.results[tid]
            elif tid in self.errors:
                results[tid] = self.errors[tid]

        return results

    def cancel(self, task_id: str) -> bool:
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            return True
        return False


class ActorSystem:
    def __init__(self):
        self.actors: Dict[str, "Actor"] = {}
        self.mailboxes: Dict[str, asyncio.Queue] = {}
        self.supervisors: Dict[str, "ActorSupervisor"] = {}

    async def spawn(self, actor: "Actor", actor_id: str = None) -> str:
        if actor_id is None:
            actor_id = str(uuid.uuid4())[:8]

        self.actors[actor_id] = actor
        self.mailboxes[actor_id] = asyncio.Queue()
        actor.actor_id = actor_id
        actor._system = self
        await actor.start()
        return actor_id

    async def send(self, actor_id: str, message: Any, reply_to: str = None) -> Any:
        if actor_id not in self.mailboxes:
            raise ValueError(f"Actor {actor_id} not found")

        future = asyncio.get_event_loop().create_future()
        envelope = ActorEnvelope(
            message=message,
            sender=reply_to,
            reply_future=future,
        )

        await self.mailboxes[actor_id].put(envelope)

        if future:
            return await future
        return None

    async def stop(self, actor_id: str):
        if actor_id in self.actors:
            await self.actors[actor_id].stop()
            del self.actors[actor_id]
            del self.mailboxes[actor_id]


@dataclass
class ActorEnvelope:
    message: Any
    sender: Optional[str]
    reply_future: Optional[asyncio.Future]


class Actor:
    def __init__(self):
        self.actor_id: str = ""
        self._system: Optional[ActorSystem] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while self._running:
            try:
                envelope = await asyncio.wait_for(
                    self._system.mailboxes[self.actor_id].get(),
                    timeout=1.0
                )
                response = await self.handle(envelope.message)
                if envelope.reply_future and not envelope.reply_future.done():
                    envelope.reply_future.set_result(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if envelope.reply_future and not envelope.reply_future.done():
                    envelope.reply_future.set_exception(e)

    async def handle(self, message: Any) -> Any:
        return None


class ActorSupervisor:
    def __init__(
        self,
        strategy: str = "restart",
        max_restarts: int = 3,
        window: float = 60.0,
    ):
        self.strategy = strategy
        self.max_restarts = max_restarts
        self.window = window
        self.restart_history: Dict[str, List[float]] = defaultdict(list)

    async def supervise(self, actor_id: str, actor_factory: Callable) -> Actor:
        while True:
            actor = await actor_factory()
            try:
                await actor.start()
                await self._monitor(actor)
            except Exception as e:
                if not self._should_restart(actor_id):
                    raise
                await self._restart(actor_id, actor_factory)

    async def _monitor(self, actor: Actor):
        while True:
            await asyncio.sleep(1)
            if actor._task and actor._task.done():
                if actor._task.cancelled():
                    break
                if actor._task.exception():
                    raise actor._task.exception()
                break

    def _should_restart(self, actor_id: str) -> bool:
        now = time.time()
        self.restart_history[actor_id] = [
            t for t in self.restart_history[actor_id] if now - t < self.window
        ]
        if len(self.restart_history[actor_id]) < self.max_restarts:
            self.restart_history[actor_id].append(now)
            return True
        return False

    async def _restart(self, actor_id: str, factory: Callable):
        await asyncio.sleep(1)


class Pipeline(Generic[T, R]):
    def __init__(
        self,
        stages: List[Callable[[T], R]],
        buffer_size: int = 100,
    ):
        self.stages = stages
        self.buffer_size = buffer_size
        self.queues: List[asyncio.Queue] = []
        self.tasks: List[asyncio.Task] = []
        self.running = False

    async def start(self):
        self.running = True
        self.queues = [asyncio.Queue(maxsize=self.buffer_size) for _ in range(len(self.stages) + 1)]

        for i, stage in enumerate(self.stages):
            task = asyncio.create_task(self._stage_worker(i, stage))
            self.tasks.append(task)

    async def stop(self):
        self.running = False
        for q in self.queues:
            await q.put(None)
        for task in self.tasks:
            await task

    async def _stage_worker(self, stage_idx: int, stage: Callable):
        in_queue = self.queues[stage_idx]
        out_queue = self.queues[stage_idx + 1] if stage_idx + 1 < len(self.queues) else None

        while self.running:
            item = await in_queue.get()
            if item is None:
                if out_queue:
                    await out_queue.put(None)
                break

            try:
                result = await stage(item) if asyncio.iscoroutinefunction(stage) else stage(item)
            except Exception as e:
                result = e

            if out_queue:
                await out_queue.put(result)

    async def push(self, item: T) -> None:
        await self.queues[0].put(item)

    async def pull(self) -> Optional[R]:
        try:
            return await asyncio.wait_for(self.queues[-1].get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


class RateLimiter:
    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: float = None) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

        if not blocking:
            return False

        start = time.time()
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if timeout and time.time() - start > timeout:
                return False
            time.sleep(0.01)

    async def acquire_async(self, tokens: int = 1, timeout: float = None) -> bool:
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if timeout:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(0.01, remaining))
            else:
                await asyncio.sleep(0.01)


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"
        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                self.failure_count = 0
                self.state = "closed"
            return result
        except self.expected_exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
            raise


class AsyncCircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"
        self.lock = asyncio.Lock()

    async def call(self, coro):
        async with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = await coro
            async with self.lock:
                self.failure_count = 0
                self.state = "closed"
            return result
        except self.expected_exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
            raise