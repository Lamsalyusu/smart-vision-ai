from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from core.domain.ports.event_bus import DomainEvent, EventBusPort, EventHandler, TEvent

logger = logging.getLogger(__name__)


class InMemoryEventBus(EventBusPort):
    """High-throughput In-Memory Asynchronous Domain Event Bus."""

    def __init__(self) -> None:
        self._subscribers: defaultdict[type[DomainEvent], list[EventHandler[Any]]] = defaultdict(list)
        self._history: list[DomainEvent] = []
        self._max_history = 500

    def subscribe(self, event_cls: type[TEvent], handler: EventHandler[TEvent]) -> None:
        self._subscribers[event_cls].append(handler)
        logger.debug("Registered handler %s for event %s", handler, event_cls.__name__)

    async def publish(self, event: DomainEvent) -> None:
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        handlers = self._subscribers.get(type(event), [])
        if not handlers:
            return

        tasks = [
            asyncio.create_task(self._safe_invoke(handler, event))
            for handler in handlers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_invoke(self, handler: EventHandler[Any], event: DomainEvent) -> None:
        try:
            await handler(event)
        except Exception:
            logger.exception("Error in event handler %s for event %s", handler, event.event_type)

    @property
    def history(self) -> list[DomainEvent]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()


# Default singleton instance
event_bus = InMemoryEventBus()
