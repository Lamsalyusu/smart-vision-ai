from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
import asyncio
import contextlib

logger = logging.getLogger(__name__)

# TODO: Add support for optimistic locking with version columns
# FIXME: Connection pool doesn't handle database failover gracefully
# XXX: No support for read-after-write consistency with replicas
# HACK: Using pickle for cache serialization - not safe for untrusted data
# FIXME: Migration manager doesn't handle concurrent migrations well
# XXX: No support for distributed transactions (2PC)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class IsolationLevel(Enum):
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class TransactionState(Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class LockMode(Enum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"
    INTENT_SHARED = "intent_shared"
    INTENT_EXCLUSIVE = "intent_exclusive"


@dataclass(frozen=True, slots=True)
class TransactionId:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Transaction:
    id: TransactionId = field(default_factory=TransactionId)
    state: TransactionState = TransactionState.ACTIVE
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    locks: Dict[str, LockMode] = field(default_factory=dict)
    savepoints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_lock(self, resource: str, mode: LockMode):
        self.locks[resource] = mode

    def has_lock(self, resource: str) -> bool:
        return resource in self.locks


@dataclass(frozen=True, slots=True)
class QueryResult:
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class QueryBuilder:
    def __init__(self, table: str):
        self.table = table
        self._select: List[str] = ["*"]
        self._where: List[Tuple[str, str, Any]] = []
        self._joins: List[Tuple[str, str, str, str]] = []
        self._order_by: List[Tuple[str, str]] = []
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._distinct = False

    def select(self, *columns: str):
        self._select = list(columns)
        return self

    def distinct(self):
        self._distinct = True
        return self

    def where(self, column: str, operator: str, value: Any):
        self._where.append((column, operator, value))
        return self

    def where_in(self, column: str, values: List[Any]):
        self._where.append((column, "IN", values))
        return self

    def where_not_in(self, column: str, values: List[Any]):
        self._where.append((column, "NOT IN", values))
        return self

    def where_between(self, column: str, start: Any, end: Any):
        self._where.append((column, "BETWEEN", (start, end)))
        return self

    def join(
        self,
        table: str,
        local_column: str,
        foreign_column: str,
        join_type: str = "INNER",
    ):
        self._joins.append((join_type, table, local_column, foreign_column))
        return self

    def left_join(self, table: str, local: str, foreign: str):
        return self.join(table, local, foreign, "LEFT")

    def right_join(self, table: str, local: str, foreign: str):
        return self.join(table, local, foreign, "RIGHT")

    def order_by(self, column: str, direction: str = "ASC"):
        self._order_by.append((column, direction.upper()))
        return self

    def group_by(self, *columns: str):
        self._group_by.extend(columns)
        return self

    def having(self, condition: str):
        self._having.append(condition)
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def offset(self, count: int):
        self._offset = count
        return self

    def build(self) -> Tuple[str, List[Any]]:
        parts = []

        select_clause = "SELECT "
        if self._distinct:
            select_clause += "DISTINCT "
        select_clause += ", ".join(self._select)
        parts.append(select_clause)

        parts.append(f"FROM {self.table}")

        for join_type, table, local, foreign in self._joins:
            parts.append(f"{join_type} JOIN {table} ON {self.table}.{local} = {table}.{foreign}")

        if self._where:
            where_parts = []
            params = []
            for column, operator, value in self._where:
                if operator == "IN" or operator == "NOT IN":
                    placeholders = ", ".join(["%s"] * len(value))
                    where_parts.append(f"{column} {operator} ({placeholders})")
                    if isinstance(value, list):
                        params.extend(value)
                elif operator == "BETWEEN":
                    where_parts.append(f"{column} BETWEEN %s AND %s")
                    params.extend(value)
                else:
                    where_parts.append(f"{column} {operator} %s")
                    params.append(value)
            parts.append("WHERE " + " AND ".join(where_parts))

        if self._group_by:
            parts.append("GROUP BY " + ", ".join(self._group_by))

        if self._having:
            parts.append("HAVING " + " AND ".join(self._having))

        if self._order_by:
            parts.append("ORDER BY " + ", ".join(f"{col} {dir}" for col, dir in self._order_by))

        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        query = " ".join(parts)
        return query, params if hasattr(self, 'params') else []


class Entity:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


class Repository(Generic[T], ABC):
    def __init__(self, entity_type: Type[T], table_name: str):
        self.entity_type = entity_type
        self.table_name = table_name

    @abstractmethod
    async def find_by_id(self, id: Any) -> Optional[T]:
        pass

    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        pass

    @abstractmethod
    async def find_by(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[T]:
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        pass

    @abstractmethod
    async def delete(self, id: Any) -> bool:
        pass

    @abstractmethod
    async def count(self, criteria: Dict[str, Any] = None) -> int:
        pass

    @abstractmethod
    async def exists(self, id: Any) -> bool:
        pass


class UnitOfWork:
    def __init__(self):
        self._new: List[Any] = []
        self._dirty: List[Any] = []
        self._deleted: List[Any] = []
        self._transaction: Optional[Transaction] = None
        self._repositories: Dict[str, Any] = {}

    def register_new(self, entity: Any):
        self._new.append(entity)

    def register_dirty(self, entity: Any):
        if entity not in self._dirty:
            self._dirty.append(entity)

    def register_deleted(self, entity: Any):
        self._deleted.append(entity)

    def register_repository(self, name: str, repo: Any):
        self._repositories[name] = repo

    def get_repository(self, name: str) -> Any:
        return self._repositories.get(name)

    def start_transaction(self, isolation: IsolationLevel = IsolationLevel.READ_COMMITTED):
        self._transaction = Transaction(isolation_level=isolation)

    async def commit(self):
        if not self._transaction:
            return
        try:
            for entity in self._new:
                await self._persist_new(entity)
            for entity in self._dirty:
                await self._persist_update(entity)
            for entity in self._deleted:
                await self._persist_delete(entity)

            self._transaction.state = TransactionState.COMMITTED
            self._transaction.end_time = datetime.utcnow()
            self._clear()
        except Exception as e:
            await self.rollback()
            raise

    async def rollback(self):
        if self._transaction:
            self._transaction.state = TransactionState.ROLLED_BACK
            self._transaction.end_time = datetime.utcnow()
            self._clear()

    async def _persist_new(self, entity: Any):
        pass

    async def _persist_update(self, entity: Any):
        pass

    async def _persist_delete(self, entity: Any):
        pass

    def _clear(self):
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()
        self._transaction = None


class ConnectionPool:
    def __init__(
        self,
        create_connection: Callable[[], Any],
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: float = 300.0,
    ):
        self.create_connection = create_connection
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._in_use: Set[Any] = set()
        self._created = 0
        self._lock = asyncio.Lock()
        self._idle_times: Dict[Any, float] = {}

    async def initialize(self):
        for _ in range(self.min_size):
            conn = await self._create_new()
            await self._pool.put(conn)

    async def _create_new(self) -> Any:
        async with self._lock:
            if self._created >= self.max_size:
                raise RuntimeError("Connection pool exhausted")
            conn = await asyncio.get_event_loop().run_in_executor(None, self.create_connection)
            self._created += 1
            return conn

    async def acquire(self) -> Any:
        try:
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            conn = await self._create_new()

        self._in_use.add(conn)
        self._idle_times[conn] = time.time()
        return conn

    async def release(self, conn: Any):
        if conn in self._in_use:
            self._in_use.remove(conn)
            self._idle_times[conn] = time.time()
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                await self._close_connection(conn)

    async def _close_connection(self, conn: Any):
        if hasattr(conn, 'close'):
            await asyncio.get_event_loop().run_in_executor(None, conn.close)
        self._created -= 1

    async def close_all(self):
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            await self._close_connection(conn)
        for conn in self._in_use:
            await self._close_connection(conn)
        self._in_use.clear()

    async def health_check(self) -> bool:
        try:
            conn = await self.acquire()
            await self.release(conn)
            return True
        except Exception:
            return False


class ReadReplicaManager:
    def __init__(self, primary: ConnectionPool, replicas: List[ConnectionPool]):
        self.primary = primary
        self.replicas = replicas
        self._current_replica = 0
        self._lock = asyncio.Lock()

    async def get_read_connection(self) -> Any:
        if not self.replicas:
            return await self.primary.acquire()

        async with self._lock:
            for _ in range(len(self.replicas)):
                replica = self.replicas[self._current_replica]
                self._current_replica = (self._current_replica + 1) % len(self.replicas)
                try:
                    conn = await replica.acquire()
                    return conn
                except Exception:
                    continue
            return await self.primary.acquire()

    async def get_write_connection(self) -> Any:
        return await self.primary.acquire()


class CacheLayer:
    def __init__(self, redis_client: Any = None, default_ttl: int = 300):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.local_cache: OrderedDict = OrderedDict()
        self.max_local_size = 1000
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.local_cache:
                value, expiry = self.local_cache[key]
                if time.time() < expiry:
                    self.local_cache.move_to_end(key)
                    return value
                else:
                    del self.local_cache[key]

        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    value = pickle.loads(data)
                    self.set(key, value)
                    return value
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            self.local_cache[key] = (value, expiry)
            self.local_cache.move_to_end(key)

            if len(self.local_cache) > self.max_local_size:
                self.local_cache.popitem(last=False)

            if self.redis:
                try:
                    self.redis.setex(key, ttl, pickle.dumps(value))
                except Exception:
                    pass

    def delete(self, key: str):
        with self._lock:
            self.local_cache.pop(key, None)
            if self.redis:
                try:
                    self.redis.delete(key)
                except Exception:
                    pass

    def clear(self):
        with self._lock:
            self.local_cache.clear()
            if self.redis:
                try:
                    self.redis.flushdb()
                except Exception:
                    pass


class Migration:
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.applied_at: Optional[datetime] = None

    @abstractmethod
    async def up(self, connection: Any):
        pass

    @abstractmethod
    async def down(self, connection: Any):
        pass


class MigrationManager:
    def __init__(self, connection_pool: ConnectionPool, migrations_table: str = "schema_migrations"):
        self.pool = connection_pool
        self.migrations_table = migrations_table
        self.migrations: List[Migration] = []
        self._lock = asyncio.Lock()

    def add_migration(self, migration: Migration):
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)

    async def initialize(self):
        conn = await self.pool.acquire()
        try:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                    version VARCHAR(255) PRIMARY KEY,
                    description TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        finally:
            await self.pool.release(conn)

    async def migrate(self, target_version: str = None) -> List[str]:
        applied = await self._get_applied()
        pending = [m for m in self.migrations if m.version not in applied]

        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        applied_versions = []
        for migration in pending:
            conn = await self.pool.acquire()
            try:
                await migration.up(conn)
                await conn.execute(
                    f"INSERT INTO {self.migrations_table} (version, description) VALUES (%s, %s)",
                    (migration.version, migration.description)
                )
                migration.applied_at = datetime.utcnow()
                applied_versions.append(migration.version)
                logger.info(f"Applied migration {migration.version}: {migration.description}")
            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                await self._rollback_migration(migration)
                raise
            finally:
                await self.pool.release(conn)
        return applied_versions

    async def rollback(self, target_version: str) -> List[str]:
        applied = await self._get_applied()
        to_rollback = [m for m in reversed(self.migrations) if m.version in applied and m.version > target_version]

        rolled_back = []
        for migration in to_rollback:
            conn = await self.pool.acquire()
            try:
                await migration.down(conn)
                await conn.execute(
                    f"DELETE FROM {self.migrations_table} WHERE version = %s",
                    (migration.version,)
                )
                migration.applied_at = None
                rolled_back.append(migration.version)
                logger.info(f"Rolled back migration {migration.version}")
            except Exception as e:
                logger.error(f"Rollback {migration.version} failed: {e}")
                raise
            finally:
                await self.pool.release(conn)
        return rolled_back

    async def _get_applied(self) -> Set[str]:
        conn = await self.pool.acquire()
        try:
            rows = await conn.fetch(f"SELECT version FROM {self.migrations_table}")
            return {row['version'] for row in rows}
        finally:
            await self.pool.release(conn)

    async def _rollback_migration(self, migration: Migration):
        pass


class QueryOptimizer:
    def __init__(self):
        self.stats: Dict[str, QueryStats] = {}
        self._lock = threading.RLock()

    def record_query(self, query: str, execution_time: float, rows_affected: int):
        with self._lock:
            key = self._normalize_query(query)
            if key not in self.stats:
                self.stats[key] = QueryStats(key)
            self.stats[key].record(execution_time, rows_affected)

    def _normalize_query(self, query: str) -> str:
        return " ".join(query.split())

    def get_slow_queries(self, threshold_ms: float = 1000.0, limit: int = 10) -> List[QueryStats]:
        with self._lock:
            return sorted(
                [s for s in self.stats.values() if s.avg_time_ms > threshold_ms],
                key=lambda s: s.avg_time_ms,
                reverse=True,
            )[:limit]

    def get_most_frequent(self, limit: int = 10) -> List[QueryStats]:
        with self._lock:
            return sorted(
                self.stats.values(),
                key=lambda s: s.count,
                reverse=True,
            )[:limit]


@dataclass
class QueryStats:
    query: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_rows: int = 0

    def record(self, execution_time: float, rows: int):
        self.count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.total_rows += rows

    @property
    def avg_time_ms(self) -> float:
        return (self.total_time / self.count) * 1000 if self.count > 0 else 0


class DatabaseCluster:
    def __init__(self):
        self.shards: Dict[str, ConnectionPool] = {}
        self.shard_key_func: Callable[[Any], str] = lambda x: "default"
        self.replication_factor = 1

    def add_shard(self, shard_id: str, pool: ConnectionPool):
        self.shards[shard_id] = pool

    def set_shard_key(self, func: Callable[[Any], str]):
        self.shard_key_func = func

    async def get_connection(self, key: Any) -> Any:
        shard_id = self.shard_key_func(key)
        pool = self.shards.get(shard_id)
        if not pool:
            raise ValueError(f"Shard {shard_id} not found")
        return await pool.acquire()


class SagaOrchestrator:
    def __init__(self):
        self.steps: List[SagaStep] = []
        self.compensation_handlers: Dict[str, Callable] = {}

    def add_step(
        self,
        name: str,
        execute: Callable,
        compensate: Callable = None,
    ):
        self.steps.append(SagaStep(name, execute, compensate))

    def add_compensation(self, step_name: str, handler: Callable):
        self.compensation_handlers[step_name] = handler

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        executed = []
        try:
            for step in self.steps:
                result = await step.execute(context)
                context[step.name] = result
                executed.append(step.name)
            return context
        except Exception as e:
            await self._compensate(executed, context)
            raise

    async def _compensate(self, executed: List[str], context: Dict[str, Any]):
        for step_name in reversed(executed):
            if step_name in self.compensation_handlers:
                try:
                    await self.compensation_handlers[step_name](context)
                except Exception as e:
                    logger.error(f"Compensation failed for {step_name}: {e}")


@dataclass
class SagaStep:
    name: str
    execute: Callable
    compensate: Callable = None


class EventSourcingRepository:
    def __init__(self, connection_pool: ConnectionPool, event_store_table: str = "events"):
        self.pool = connection_pool
        self.event_store_table = event_store_table
        self.snapshots: Dict[str, Any] = {}

    async def append_event(
        self,
        aggregate_id: str,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Dict[str, Any] = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "aggregate_id": aggregate_id,
            "event_type": event_type,
            "payload": json.dumps(payload),
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.utcnow(),
        }

        conn = await self._get_connection()
        try:
            await conn.execute(
                f"INSERT INTO {self._event_store_table} (event_id, aggregate_id, event_type, payload, metadata, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (event["event_id"], event["aggregate_id"], event["event_type"], event["payload"], event["metadata"], event["created_at"])
            )
            return event_id
        finally:
            await self._release_connection(conn)

    def _get_connection(self):
        pass

    def _release_connection(self, conn):
        pass


class CQRSRepository:
    def __init__(
        self,
        command_repo: Repository,
        query_repo: Repository,
    ):
        self.command_repo = command_repo
        self.query_repo = query_repo

    async def create(self, entity: T) -> T:
        return await self.command_repo.create(entity)

    async def update(self, entity: T) -> T:
        return await self.command_repo.update(entity)

    async def delete(self, id: Any) -> bool:
        return await self.command_repo.delete(id)

    async def find_by_id(self, id: Any) -> Optional[T]:
        return await self.query_repo.find_by_id(id)

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        return await self.query_repo.find_all(limit, offset)

    async def find_by(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[T]:
        return await self.query_repo.find_by(criteria, limit, offset)


def create_database_stack(
    connection_string: str,
    min_pool_size: int = 5,
    max_pool_size: int = 20,
    replicas: List[str] = None,
    cache_client: Any = None,
) -> Tuple[ConnectionPool, Optional[ReadReplicaManager], CacheLayer, MigrationManager]:
    pool = ConnectionPool(
        create_connection=lambda: None,
        min_size=min_pool_size,
        max_size=max_pool_size,
    )

    replica_manager = None
    if replicas:
        replica_pools = [
            ConnectionPool(lambda: None, min_size=2, max_size=10)
            for _ in replicas
        ]
        replica_manager = ReadReplicaManager(
            primary=pool,
            replicas=replica_pools,
        )

    cache = CacheLayer(default_ttl=300)

    migration_manager = MigrationManager(None)

    return pool, replica_manager, cache, migration_manager