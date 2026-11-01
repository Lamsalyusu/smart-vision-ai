from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from .repositories import (
    UserRepositoryPort,
    SessionRepositoryPort,
    CropRepositoryPort,
    TelemetryRepositoryPort,
    AuditRepositoryPort,
)


class UnitOfWorkPort(ABC):
    users: UserRepositoryPort
    sessions: SessionRepositoryPort
    crops: CropRepositoryPort
    telemetry: TelemetryRepositoryPort
    audits: AuditRepositoryPort

    @abstractmethod
    def __enter__(self) -> UnitOfWorkPort:
        pass

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def rollback(self) -> None:
        pass
