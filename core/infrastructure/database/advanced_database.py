from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import threading
import sqlite3
import json
import pickle
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    DYNAMODB = "dynamodb"
    COUCHDB = "couchdb"


class TransactionIsolation(Enum):
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    db_type: DatabaseType = DatabaseType.SQLITE
    connection_string: str = ""
    max_connections: int = 10
    timeout: float = 30.0
    pool_size: int = 5
    enable_cache: bool = True
    cache_size: int = 1000
    auto_commit: bool = True
    isolation_level: TransactionIsolation = TransactionIsolation.READ_COMMITTED


@dataclass(frozen=True, slots=True)
class QueryResult:
    rows: List[Dict[str, Any]]
    affected_rows: int
    last_insert_id: Optional[int]
    execution_time: float
    columns: List[str]


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    name: str
    up_sql: str
    down_sql: str
    applied_at: Optional[float] = None


class ConnectionPool:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: List[Any] = []
        self.in_use: Set[Any] = set()
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.config.pool_size):
            conn = self._create_connection()
            if conn:
                self.pool.append(conn)

    def _create_connection(self) -> Any:
        if self.config.db_type == DatabaseType.SQLITE:
            conn = sqlite3.connect(self.config.connection_string or ":memory:", check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        return None

    def acquire(self) -> Any:
        with self.lock:
            if self.pool:
                conn = self.pool.pop()
                self.in_use.add(conn)
                return conn
            elif len(self.in_use) < self.config.max_connections:
                conn = self._create_connection()
                if conn:
                    self.in_use.add(conn)
                return conn
        return None

    def release(self, conn: Any):
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                if len(self.pool) < self.config.pool_size:
                    self.pool.append(conn)
                else:
                    self._close_connection(conn)

    def _close_connection(self, conn: Any):
        if self.config.db_type == DatabaseType.SQLITE:
            conn.close()

    def close_all(self):
        with self.lock:
            for conn in self.pool:
                self._close_connection(conn)
            for conn in self.in_use:
                self._close_connection(conn)
            self.pool.clear()
            self.in_use.clear()


class BaseRepository(ABC):
    def __init__(self, table_name: str, primary_key: str = "id"):
        self.table_name = table_name
        self.primary_key = primary_key

    @abstractmethod
    def create(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def find_by_id(self, id: Any) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def find_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def update(self, id: Any, entity: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        pass


class SQLiteRepository(BaseRepository):
    def __init__(self, connection: sqlite3.Connection, table_name: str, primary_key: str = "id"):
        super().__init__(table_name, primary_key)
        self.connection = connection

    def create(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        columns = ", ".join(entity.keys())
        placeholders = ", ".join(["?" for _ in entity])
        sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"

        cursor = self.connection.cursor()
        cursor.execute(sql, list(entity.values()))
        self.connection.commit()

        entity[self.primary_key] = cursor.lastrowid
        return entity

    def find_by_id(self, id: Any) -> Optional[Dict[str, Any]]:
        sql = f"SELECT * FROM {self.table_name} WHERE {self.primary_key} = ?"
        cursor = self.connection.cursor()
        cursor.execute(sql, (id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def find_all(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {self.table_name} LIMIT ? OFFSET ?"
        cursor = self.connection.cursor()
        cursor.execute(sql, (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    def update(self, id: Any, entity: Dict[str, Any]) -> bool:
        set_clause = ", ".join([f"{k} = ?" for k in entity.keys()])
        sql = f"UPDATE {self.table_name} SET {set_clause} WHERE {self.primary_key} = ?"

        cursor = self.connection.cursor()
        cursor.execute(sql, list(entity.values()) + [id])
        self.connection.commit()
        return cursor.rowcount > 0

    def delete(self, id: Any) -> bool:
        sql = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = ?"
        cursor = self.connection.cursor()
        cursor.execute(sql, (id,))
        self.connection.commit()
        return cursor.rowcount > 0

    def query(self, where: str, params: Tuple = (), limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {self.table_name} WHERE {where} LIMIT ? OFFSET ?"
        cursor = self.connection.cursor()
        cursor.execute(sql, params + (limit, offset))
        return [dict(row) for row in cursor.fetchall()]

    def count(self, where: str = "", params: Tuple = ()) -> int:
        sql = f"SELECT COUNT(*) FROM {self.table_name}"
        if where:
            sql += f" WHERE {where}"
        cursor = self.connection.cursor()
        cursor.execute(sql, params)
        return cursor.fetchone()[0]


class UnitOfWork:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.new_objects: List[Tuple[str, Dict]] = []
        self.dirty_objects: List[Tuple[str, Any, Dict]] = []
        self.removed_objects: List[Tuple[str, Any]] = []
        self.committed = False

    def register_new(self, table: str, entity: Dict):
        self.new_objects.append((table, entity))

    def register_dirty(self, table: str, id: Any, entity: Dict):
        self.dirty_objects.append((table, id, entity))

    def register_removed(self, table: str, id: Any):
        self.removed_objects.append((table, id))

    def commit(self):
        cursor = self.connection.cursor()

        for table, entity in self.new_objects:
            columns = ", ".join(entity.keys())
            placeholders = ", ".join(["?" for _ in entity])
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, list(entity.values()))

        for table, id, entity in self.dirty_objects:
            set_clause = ", ".join([f"{k} = ?" for k in entity.keys()])
            sql = f"UPDATE {table} SET {set_clause} WHERE id = ?"
            cursor.execute(sql, list(entity.values()) + [id])

        for table, id in self.removed_objects:
            sql = f"DELETE FROM {table} WHERE id = ?"
            cursor.execute(sql, (id,))

        self.connection.commit()
        self.committed = True

    def rollback(self):
        self.new_objects.clear()
        self.dirty_objects.clear()
        self.removed_objects.clear()


class MigrationManager:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self._ensure_migration_table()

    def _ensure_migration_table(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at REAL NOT NULL
            )
        """)
        self.connection.commit()

    def get_applied_migrations(self) -> List[str]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT version FROM migrations ORDER BY version")
        return [row[0] for row in cursor.fetchall()]

    def apply_migration(self, migration: Migration):
        if migration.version in self.get_applied_migrations():
            return False

        cursor = self.connection.cursor()
        cursor.executescript(migration.up_sql)
        cursor.execute(
            "INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (migration.version, migration.name, time.time())
        )
        self.connection.commit()
        return True

    def rollback_migration(self, version: str):
        cursor = self.connection.cursor()
        cursor.execute("SELECT down_sql FROM migrations WHERE version = ?", (version,))
        row = cursor.fetchone()
        if row:
            cursor.executescript(row[0])
            cursor.execute("DELETE FROM migrations WHERE version = ?", (version,))
            self.connection.commit()
            return True
        return False


class DataManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self.repositories: Dict[str, BaseRepository] = {}
        self.migration_manager: Optional[MigrationManager] = None

    def get_connection(self) -> Any:
        return self.pool.acquire()

    def release_connection(self, conn: Any):
        self.pool.release(conn)

    @contextmanager
    def transaction(self):
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    def get_repository(self, table_name: str) -> BaseRepository:
        if table_name not in self.repositories:
            conn = self.get_connection()
            self.repositories[table_name] = SQLiteRepository(conn, table_name)
        return self.repositories[table_name]

    def execute(self, sql: str, params: Tuple = ()) -> QueryResult:
        start = time.time()
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []

            return QueryResult(
                rows=rows,
                affected_rows=cursor.rowcount,
                last_insert_id=cursor.lastrowid,
                execution_time=time.time() - start,
                columns=columns
            )
        finally:
            self.release_connection(conn)

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            conn.commit()
            return cursor.rowcount
        finally:
            self.release_connection(conn)

    def batch_insert(self, table: str, entities: List[Dict]) -> int:
        if not entities:
            return 0

        columns = list(entities[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

        params = [tuple(e[col] for col in columns) for e in entities]
        return self.execute_many(sql, params)

    def shutdown(self):
        self.pool.close_all()


class QueryBuilder:
    def __init__(self, table: str):
        self.table = table
        self.select_columns = ["*"]
        self.where_conditions = []
        self.where_params = []
        self.join_clauses = []
        self.order_by = ""
        self.limit_val = 0
        self.offset_val = 0

    def select(self, *columns: str) -> 'QueryBuilder':
        self.select_columns = list(columns) if columns else ["*"]
        return self

    def where(self, condition: str, *params) -> 'QueryBuilder':
        self.where_conditions.append(condition)
        self.where_params.extend(params)
        return self

    def join(self, table: str, condition: str, join_type: str = "INNER") -> 'QueryBuilder':
        self.join_clauses.append(f"{join_type} JOIN {table} ON {condition}")
        return self

    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        self.order_by = f"ORDER BY {column} {'DESC' if desc else 'ASC'}"
        return self

    def limit(self, limit: int) -> 'QueryBuilder':
        self.limit_val = limit
        return self

    def offset(self, offset: int) -> 'QueryBuilder':
        self.offset_val = offset
        return self

    def build_select(self) -> Tuple[str, Tuple]:
        sql = f"SELECT {', '.join(self.select_columns)} FROM {self.table}"

        for join in self.join_clauses:
            sql += f" {join}"

        if self.where_conditions:
            sql += " WHERE " + " AND ".join(self.where_conditions)

        if self.order_by:
            sql += f" {self.order_by}"

        if self.limit_val > 0:
            sql += f" LIMIT {self.limit_val}"

        if self.offset_val > 0:
            sql += f" OFFSET {self.offset_val}"

        return sql, tuple(self.where_params)

    def build_count(self) -> Tuple[str, Tuple]:
        sql = f"SELECT COUNT(*) FROM {self.table}"

        for join in self.join_clauses:
            sql += f" {join}"

        if self.where_conditions:
            sql += " WHERE " + " AND ".join(self.where_conditions)

        return sql, tuple(self.where_params)


class CacheLayer:
    def __init__(self, max_size: int = 10000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest = min(self.cache.items(), key=lambda x: x[1][1])
                del self.cache[oldest[0]]
            self.cache[key] = (value, time.time())

    def invalidate(self, key: str):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        with self.lock:
            self.cache.clear()


class DatabaseMetrics:
    def __init__(self):
        self.query_count = 0
        self.total_time = 0.0
        self.slow_queries: List[Tuple[str, float]] = []
        self.errors = 0
        self.lock = threading.Lock()

    def record_query(self, sql: str, execution_time: float):
        with self.lock:
            self.query_count += 1
            self.total_time += execution_time
            if execution_time > 1.0:
                self.slow_queries.append((sql, execution_time))
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]

    def record_error(self):
        with self.lock:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'query_count': self.query_count,
                'avg_time': self.total_time / max(self.query_count, 1),
                'slow_queries': len(self.slow_queries),
                'errors': self.errors
            }


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet