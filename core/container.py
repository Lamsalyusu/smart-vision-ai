from __future__ import annotations

from core.application.event_bus import event_bus
from core.domain.analytics.behavior import TrajectoryBehaviorAnalyzer
from core.domain.analytics.policy_engine import PolicyEngine
from core.domain.tracking.hungarian_fusion import HungarianFusionMatcher
from core.infrastructure.concurrency.worker_pool import worker_pool
from core.infrastructure.database.unit_of_work import SqlAlchemyUnitOfWork
from core.infrastructure.scalability.distributed_lock import DistributedLock
from core.infrastructure.scalability.rate_limiter import (
    rate_limiter,
)
from core.infrastructure.security.jwt_auth import jwt_service


class AppContainer:
    """Dependency Injection Container assembling Domain, Application, and Infrastructure layers."""

    def __init__(self) -> None:
        self.event_bus = event_bus
        self.worker_pool = worker_pool
        self.matcher = HungarianFusionMatcher()
        self.behavior_analyzer = TrajectoryBehaviorAnalyzer()
        self.policy_engine = PolicyEngine()
        self.rate_limiter = rate_limiter
        self.jwt_service = jwt_service

    def create_uow(self) -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork()

    def create_distributed_lock(self, resource_key: str, ttl_seconds: float = 10.0) -> DistributedLock:
        return DistributedLock(resource_key=resource_key, ttl_seconds=ttl_seconds)


container = AppContainer()
