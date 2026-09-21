from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Type, TypeVar


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Immutable base domain event with monotonic timestamp and unique event ID."""
    event_id: str
    occurred_at: float = field(default_factory=time.monotonic)
    event_type: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", self.__class__.__name__)


TEvent = TypeVar("TEvent", bound=DomainEvent)
EventHandler = Callable[[TEvent], Coroutine[Any, Any, None]]


class EventBusPort(ABC):
    """Hexagonal Port for Domain Event Publishing and Subscribing."""

    @abstractmethod
    def subscribe(self, event_cls: Type[TEvent], handler: EventHandler[TEvent]) -> None:
        """Register a handler for a domain event type."""
        pass

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event asynchronously to all registered subscribers."""
        pass
