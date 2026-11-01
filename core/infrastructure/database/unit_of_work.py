from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from core.domain.ports.unit_of_work import UnitOfWorkPort
from .engine import SessionLocal
from .repositories import (
    SqlAlchemyUserRepository,
    SqlAlchemySessionRepository,
    SqlAlchemyCropRepository,
    SqlAlchemyTelemetryRepository,
    SqlAlchemyAuditRepository,
)


class SqlAlchemyUnitOfWork(UnitOfWorkPort):
    """Unit of Work managing transactions and database session lifecycle."""

    def __init__(self, session_factory=SessionLocal) -> None:
        self.session_factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self.session = self.session_factory()
        self.users = SqlAlchemyUserRepository(self.session)
        self.sessions = SqlAlchemySessionRepository(self.session)
        self.crops = SqlAlchemyCropRepository(self.session)
        self.telemetry = SqlAlchemyTelemetryRepository(self.session)
        self.audits = SqlAlchemyAuditRepository(self.session)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session is not None:
            if exc_type is not None:
                self.rollback()
            self.session.close()

    def commit(self) -> None:
        if self.session is not None:
            self.session.commit()

    def rollback(self) -> None:
        if self.session is not None:
            self.session.rollback()
