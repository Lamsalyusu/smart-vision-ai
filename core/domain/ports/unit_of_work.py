from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from .repositories import (
    AuditRepositoryPort,
    CropRepositoryPort,
    SessionRepositoryPort,
    TelemetryRepositoryPort,
    UserRepositoryPort,
)


class UnitOfWorkPort(ABC):
    users: UserRepositoryPort
    sessions: SessionRepositoryPort
    crops: CropRepositoryPort
    telemetry: TelemetryRepositoryPort
    audits: AuditRepositoryPort

    @abstractmethod
    def __enter__(self) -> Self:
        pass

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def rollback(self) -> None:
        pass
