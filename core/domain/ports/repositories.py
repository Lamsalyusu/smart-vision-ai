from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class UserRepositoryPort(ABC):
    @abstractmethod
    def create(self, user_data: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_by_id(self, user_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def get_by_identifier(self, identifier: str) -> Optional[Any]:
        pass


class SessionRepositoryPort(ABC):
    @abstractmethod
    def create(self, user_id: str, days: int = 7) -> str:
        pass

    @abstractmethod
    def get_user_from_token(self, token: str) -> Optional[Any]:
        pass

    @abstractmethod
    def revoke(self, token: str) -> None:
        pass


class CropRepositoryPort(ABC):
    @abstractmethod
    def save(self, crop_data: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def get_latest(self, object_class: Optional[str] = None) -> Optional[Any]:
        pass


class TelemetryRepositoryPort(ABC):
    @abstractmethod
    def record_detection(self, entry: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class AuditRepositoryPort(ABC):
    @abstractmethod
    def log_event(self, event_type: str, user_id: Optional[str], details: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        pass
