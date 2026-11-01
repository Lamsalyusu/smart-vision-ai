from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class UserRepositoryPort(ABC):
    @abstractmethod
    def create(self, user_data: dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_by_id(self, user_id: str) -> Any | None:
        pass

    @abstractmethod
    def get_by_identifier(self, identifier: str) -> Any | None:
        pass


class SessionRepositoryPort(ABC):
    @abstractmethod
    def create(self, user_id: str, days: int = 7) -> str:
        pass

    @abstractmethod
    def get_user_from_token(self, token: str) -> Any | None:
        pass

    @abstractmethod
    def revoke(self, token: str) -> None:
        pass


class CropRepositoryPort(ABC):
    @abstractmethod
    def save(self, crop_data: dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_latest(self, object_class: str | None = None) -> Any | None:
        pass


class TelemetryRepositoryPort(ABC):
    @abstractmethod
    def record_detection(self, entry: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class AuditRepositoryPort(ABC):
    @abstractmethod
    def log_event(self, event_type: str, user_id: str | None, details: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        pass
