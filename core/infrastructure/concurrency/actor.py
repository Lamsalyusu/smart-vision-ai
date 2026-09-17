from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

TMessage = TypeVar("TMessage")
TReply = TypeVar("TReply")


class ActorStatus(str, Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class MailboxFullStrategy(str, Enum):
    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class Envelope(Generic[TMessage]):
    payload: TMessage
    reply_future: asyncio.Future[Any] | None = None
    enqueued_at: float = 0.0


class Actor(Generic[TMessage]):
    """Strict asynchronous Actor with mailbox queue, supervision, and lifecycle management."""

    def __init__(
        self,
        actor_id: str,
        mailbox_capacity: int = 100,
        strategy: MailboxFullStrategy = MailboxFullStrategy.BLOCK,
    ) -> None:
        self.actor_id = actor_id
        self._mailbox_capacity = mailbox_capacity
        self._strategy = strategy
        self._mailbox: asyncio.Queue[Envelope[TMessage]] = asyncio.Queue(maxsize=mailbox_capacity)
        self._status = ActorStatus.INITIALIZED
        self._task: asyncio.Task[None] | None = None
        self._processed_count = 0
        self._dropped_count = 0
        self._last_error: Exception | None = None

    @property
    def status(self) -> ActorStatus:
        return self._status

    @property
    def mailbox_size(self) -> int:
        return self._mailbox.qsize()

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def is_alive(self) -> bool:
        return self._status == ActorStatus.RUNNING and self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the actor execution loop."""
        if self._status == ActorStatus.RUNNING and self._task and not self._task.done():
            return
        self._status = ActorStatus.RUNNING
        self._task = asyncio.create_task(self._run_loop(), name=f"actor-{self.actor_id}")

    async def stop(self) -> None:
        """Gracefully stop the actor after processing in-flight messages."""
        if self._status == ActorStatus.STOPPED:
            return
        self._status = ActorStatus.STOPPED
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Actor '{self.actor_id}' error during cancellation: {exc}")
        self._task = None

    async def tell(self, message: TMessage) -> None:
        """Fire-and-forget message delivery into actor mailbox with backpressure strategy."""
        if self._status != ActorStatus.RUNNING:
            raise RuntimeError(f"Actor '{self.actor_id}' is not running (status: {self._status})")

        envelope = Envelope(payload=message, enqueued_at=time.monotonic())

        if self._strategy == MailboxFullStrategy.REJECT:
            try:
                self._mailbox.put_nowait(envelope)
            except asyncio.QueueFull:
                self._dropped_count += 1
                raise BufferError(f"Actor '{self.actor_id}' mailbox is full")
        elif self._strategy == MailboxFullStrategy.DROP_OLDEST:
            if self._mailbox.full():
                try:
                    discarded = self._mailbox.get_nowait()
                    self._mailbox.task_done()
                    self._dropped_count += 1
                    if discarded.reply_future and not discarded.reply_future.done():
                        discarded.reply_future.set_exception(
                            BufferError("Message dropped due to DROP_OLDEST policy")
                        )
                except asyncio.QueueEmpty:
                    pass
            await self._mailbox.put(envelope)
        else:
            await self._mailbox.put(envelope)

    async def ask(self, message: TMessage, timeout_seconds: float = 5.0) -> Any:
        """Request-reply pattern with timeout and cancellation token."""
        if self._status != ActorStatus.RUNNING:
            raise RuntimeError(f"Actor '{self.actor_id}' is not running")

        loop = asyncio.get_running_loop()
        reply_future: asyncio.Future[Any] = loop.create_future()
        envelope = Envelope(
            payload=message,
            reply_future=reply_future,
            enqueued_at=time.monotonic(),
        )

        await self._mailbox.put(envelope)
        try:
            return await asyncio.wait_for(reply_future, timeout=timeout_seconds)
        except TimeoutError:
            if not reply_future.done():
                reply_future.cancel()
            raise TimeoutError(f"Actor '{self.actor_id}' timed out after {timeout_seconds}s")

    async def _run_loop(self) -> None:
        while self._status == ActorStatus.RUNNING:
            try:
                envelope = await self._mailbox.get()
                try:
                    result = await self.receive(envelope.payload)
                    self._processed_count += 1
                    if envelope.reply_future and not envelope.reply_future.done():
                        envelope.reply_future.set_result(result)
                except Exception as exc:
                    self._last_error = exc
                    self._status = ActorStatus.FAILED
                    logger.exception("Error handling message in actor '%s'", self.actor_id)
                    if envelope.reply_future and not envelope.reply_future.done():
                        envelope.reply_future.set_exception(exc)
                    break
                finally:
                    self._mailbox.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                self._last_error = exc
                self._status = ActorStatus.FAILED
                break

    async def receive(self, message: TMessage) -> Any:
        """Subclasses must implement this message handling hook."""
        raise NotImplementedError


class SupervisorStrategy(str, Enum):
    RESTART = "restart"
    RESUME = "resume"
    STOP = "stop"


class ActorSupervisor:
    """Supervises child actors, monitoring heartbeats and auto-restarting on failure."""

    def __init__(
        self,
        strategy: SupervisorStrategy = SupervisorStrategy.RESTART,
        max_restarts: int = 3,
        restart_window_seconds: float = 60.0,
    ) -> None:
        self.strategy = strategy
        self.max_restarts = max_restarts
        self.restart_window_seconds = restart_window_seconds
        self._actors: dict[str, Actor[Any]] = {}
        self._restart_history: dict[str, list[float]] = {}

    def register(self, actor: Actor[Any]) -> None:
        self._actors[actor.actor_id] = actor
        self._restart_history[actor.actor_id] = []

    async def start_all(self) -> None:
        for actor in self._actors.values():
            await actor.start()

    async def stop_all(self) -> None:
        for actor in self._actors.values():
            await actor.stop()

    async def check_and_heal(self) -> list[str]:
        """Check all supervised actors and apply supervisor strategy if any have died."""
        healed = []
        now = time.monotonic()
        for actor_id, actor in list(self._actors.items()):
            if not actor.is_alive and actor.status != ActorStatus.STOPPED:
                history = [
                    t for t in self._restart_history[actor_id]
                    if now - t <= self.restart_window_seconds
                ]
                self._restart_history[actor_id] = history

                if len(history) < self.max_restarts and self.strategy == SupervisorStrategy.RESTART:
                    self._restart_history[actor_id].append(now)
                    await actor.start()
                    healed.append(actor_id)
                elif self.strategy == SupervisorStrategy.STOP:
                    await actor.stop()
        return healed
