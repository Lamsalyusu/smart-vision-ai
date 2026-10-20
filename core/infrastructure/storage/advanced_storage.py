from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
import json
import sqlite3
import threading
import hashlib
import pickle
from collections import defaultdict
from contextlib import contextmanager


class StorageBackend(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    MONGODB = "mongodb"
    CASSANDRA = "cassandra"
    ELASTICSEARCH = "elasticsearch"
    S3 = "s3"
    FILESYSTEM = "filesystem"


class CompressionType(Enum):
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    SNAPPY = "snappy"
    BROTLI = "brotli"


class SerializationFormat(Enum):
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    PARQUET = "parquet"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    backend: StorageBackend = StorageBackend.MEMORY
    connection_string: str = ""
    compression: CompressionType = CompressionType.NONE
    serialization: SerializationFormat = SerializationFormat.JSON
    max_connections: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3
    batch_size: int = 100
    enable_cache: bool = True
    cache_size: int = 10000


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    version: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class QueryResult:
    objects: List[StoredObject]
    total_count: int
    page: int
    page_size: int
    query_time: float


class BaseStorage(ABC):
    def __init__(self, config: StorageConfig):
        self.config = config
        self._cache: Dict[str, StoredObject] = {}
        self._cache_lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'reads': 0,
            'deletes': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()

    @abstractmethod
    def _connect(self):
        pass

    @abstractmethod
    def _disconnect(self):
        pass

    @abstractmethod
    def _put_raw(self, key: str, data: bytes, metadata: Dict) -> bool:
        pass

    @abstractmethod
    def _get_raw(self, key: str) -> Optional[Tuple[bytes, Dict]]:
        pass

    @abstractmethod
    def _delete_raw(self, key: str) -> bool:
        pass

    @abstractmethod
    def _list_keys(self, prefix: str = "", limit: int = 1000) -> List[str]:
        pass

    @abstractmethod
    def _query_raw(self, query: Dict, limit: int, offset: int) -> Tuple[List[Tuple[bytes, Dict]], int]:
        pass

    def _serialize(self, obj: Any) -> bytes:
        if self.config.serialization == SerializationFormat.JSON:
            return json.dumps(obj, default=str).encode('utf-8')
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(obj)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.packb(obj, use_bin_type=True)
            except ImportError:
                return json.dumps(obj, default=str).encode('utf-8')
        return json.dumps(obj, default=str).encode('utf-8')

    def _deserialize(self, data: bytes) -> Any:
        if self.config.serialization == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.unpackb(data, raw=False)
            except ImportError:
                return json.loads(data.decode('utf-8'))
        return json.loads(data.decode('utf-8'))

    def _compress(self, data: bytes) -> bytes:
        if self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        elif self.config.compression == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                return data
        elif self.config.compression == CompressionType.ZSTD:
            try:
                import zstd
                return zstd.compress(data)
            except ImportError:
                return data
        return data

    def _decompress(self, data: bytes) -> bytes:
        if self.config.compression == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                return data
        elif self.config.compression == CompressionType.ZSTD:
            try:
                import zstd
                return zstd.decompress(data)
            except ImportError:
                return data
        return data

    def _get_from_cache(self, key: str) -> Optional[StoredObject]:
        with self._cache_lock:
            obj = self._cache.get(key)
            if obj and (obj.expires_at is None or obj.expires_at > time.time()):
                with self._stats_lock:
                    self._stats['hits'] += 1
                return obj
            elif obj:
                del self._cache[key]
            with self._stats_lock:
                self._stats['misses'] += 1
            return None

    def _put_to_cache(self, key: str, obj: StoredObject):
        if not self.config.enable_cache:
            return

        with self._cache_lock:
            if len(self._cache) >= self.config.cache_size:
                oldest = min(self._cache.items(), key=lambda x: x[1].created_at)
                del self._cache[oldest[0]]
            self._cache[key] = obj

    def _invalidate_cache(self, key: str):
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]

    def put(self, key: str, value: Any, metadata: Dict = None, ttl: float = None, tags: List[str] = None) -> bool:
        try:
            serialized = self._serialize(value)
            compressed = self._compress(serialized)

            now = time.time()
            stored_obj = StoredObject(
                key=key,
                value=value,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
                expires_at=now + ttl if ttl else None,
                tags=tags or []
            )

            meta = {
                'created_at': stored_obj.created_at,
                'updated_at': stored_obj.updated_at,
                'expires_at': stored_obj.expires_at,
                'version': stored_obj.version,
                'tags': stored_obj.tags,
                'metadata': stored_obj.metadata,
                'compressed': self.config.compression != CompressionType.NONE,
                'serialized': self.config.serialization.value
            }

            success = self._put_raw(key, compressed, meta)

            if success:
                self._put_to_cache(key, stored_obj)
                with self._stats_lock:
                    self._stats['writes'] += 1

            return success

        except Exception:
            with self._stats_lock:
                self._stats['errors'] += 1
            return False

    def get(self, key: str, default: Any = None) -> Any:
        cached = self._get_from_cache(key)
        if cached:
            with self._stats_lock:
                self._stats['reads'] += 1
            return cached.value

        try:
            result = self._get_raw(key)
            if result is None:
                return default

            data, meta = result
            if meta.get('compressed', False):
                data = self._decompress(data)

            value = self._deserialize(data)

            stored_obj = StoredObject(
                key=key,
                value=value,
                metadata=meta.get('metadata', {}),
                created_at=meta.get('created_at', time.time()),
                updated_at=meta.get('updated_at', time.time()),
                expires_at=meta.get('expires_at'),
                version=meta.get('version', 1),
                tags=meta.get('tags', [])
            )

            self._put_to_cache(key, stored_obj)
            with self._stats_lock:
                self._stats['reads'] += 1

            return value

        except Exception:
            with self._stats_lock:
                self._stats['errors'] += 1
            return default

    def delete(self, key: str) -> bool:
        try:
            self._invalidate_cache(key)
            success = self._delete_raw(key)
            if success:
                with self._stats_lock:
                    self._stats['deletes'] += 1
            return success
        except Exception:
            with self._stats_lock:
                self._stats['errors'] += 1
            return False

    def exists(self, key: str) -> bool:
        cached = self._get_from_cache(key)
        if cached:
            return True
        return self._get_raw(key) is not None

    def list_keys(self, prefix: str = "", limit: int = 1000) -> List[str]:
        return self._list_keys(prefix, limit)

    def query(self, query: Dict, limit: int = 100, offset: int = 0) -> QueryResult:
        start = time.time()
        try:
            results, total = self._query_raw(query, limit, offset)
            objects = []

            for data, meta in results:
                if meta.get('compressed', False):
                    data = self._decompress(data)
                value = self._deserialize(data)

                objects.append(StoredObject(
                    key=meta.get('key', ''),
                    value=value,
                    metadata=meta.get('metadata', {}),
                    created_at=meta.get('created_at', time.time()),
                    updated_at=meta.get('updated_at', time.time()),
                    expires_at=meta.get('expires_at'),
                    version=meta.get('version', 1),
                    tags=meta.get('tags', [])
                ))

            return QueryResult(
                objects=objects,
                total_count=total,
                page=offset // limit + 1 if limit > 0 else 1,
                page_size=limit,
                query_time=time.time() - start
            )
        except Exception:
            return QueryResult([], 0, 1, limit, time.time() - start)

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = self._stats.copy()
            stats['cache_size'] = len(self._cache)
            stats['hit_rate'] = stats['hits'] / max(stats['hits'] + stats['misses'], 1)
        return stats


class MemoryStorage(BaseStorage):
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._data: Dict[str, Tuple[bytes, Dict]] = {}
        self._lock = threading.RLock()

    def _connect(self):
        pass

    def _disconnect(self):
        pass

    def _put_raw(self, key: str, data: bytes, metadata: Dict) -> bool:
        with self._lock:
            metadata['key'] = key
            self._data[key] = (data, metadata)
        return True

    def _get_raw(self, key: str) -> Optional[Tuple[bytes, Dict]]:
        with self._lock:
            return self._data.get(key)

    def _delete_raw(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
        return False

    def _list_keys(self, prefix: str = "", limit: int = 1000) -> List[str]:
        with self._lock:
            keys = [k for k in self._data.keys() if k.startswith(prefix)]
            return keys[:limit]

    def _query_raw(self, query: Dict, limit: int, offset: int) -> Tuple[List[Tuple[bytes, Dict]], int]:
        with self._lock:
            matching = []
            for key, (data, meta) in self._data.items():
                match = True
                for qk, qv in query.items():
                    if qk == 'tags':
                        if not any(tag in meta.get('tags', []) for tag in qv):
                            match = False
                            break
                    elif qk in meta and meta[qk] != qv:
                        match = False
                        break
                if match:
                    matching.append((data, meta))

            total = len(matching)
            return matching[offset:offset+limit], total


class SQLiteStorage(BaseStorage):
    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.connection_string = config.connection_string or ":memory:"
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.connection_string, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self):
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL,
                version INTEGER DEFAULT 1,
                tags TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON objects(expires_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tags ON objects(tags)")
        conn.commit()

    def _connect(self):
        self._get_connection()

    def _disconnect(self):
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None

    def _put_raw(self, key: str, data: bytes, metadata: Dict) -> bool:
        try:
            conn = self._get_connection()
            tags_json = json.dumps(metadata.get('tags', []))
            meta_json = json.dumps({k: v for k, v in metadata.items() if k != 'tags'})

            conn.execute("""
                INSERT OR REPLACE INTO objects (key, data, metadata, created_at, updated_at, expires_at, version, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key, data, meta_json,
                metadata.get('created_at', time.time()),
                metadata.get('updated_at', time.time()),
                metadata.get('expires_at'),
                metadata.get('version', 1),
                tags_json
            ))
            conn.commit()
            return True
        except Exception:
            return False

    def _get_raw(self, key: str) -> Optional[Tuple[bytes, Dict]]:
        try:
            conn = self._get_connection()
            row = conn.execute("SELECT data, metadata, created_at, updated_at, expires_at, version, tags FROM objects WHERE key = ?", (key,)).fetchone()
            if row:
                meta = json.loads(row['metadata'])
                meta['created_at'] = row['created_at']
                meta['updated_at'] = row['updated_at']
                meta['expires_at'] = row['expires_at']
                meta['version'] = row['version']
                meta['tags'] = json.loads(row['tags'])
                return row['data'], meta
        except Exception:
            pass
        return None

    def _delete_raw(self, key: str) -> bool:
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM objects WHERE key = ?", (key,))
            conn.commit()
            return True
        except Exception:
            return False

    def _list_keys(self, prefix: str = "", limit: int = 1000) -> List[str]:
        try:
            conn = self._get_connection()
            if prefix:
                rows = conn.execute("SELECT key FROM objects WHERE key LIKE ? LIMIT ?", (f"{prefix}%", limit)).fetchall()
            else:
                rows = conn.execute("SELECT key FROM objects LIMIT ?", (limit,)).fetchall()
            return [row['key'] for row in rows]
        except Exception:
            return []

    def _query_raw(self, query: Dict, limit: int, offset: int) -> Tuple[List[Tuple[bytes, Dict]], int]:
        try:
            conn = self._get_connection()

            where_clauses = []
            params = []

            for qk, qv in query.items():
                if qk == 'tags':
                    for tag in qv:
                        where_clauses.append("tags LIKE ?")
                        params.append(f"%{tag}%")
                elif qk in ['created_at', 'updated_at', 'expires_at', 'version']:
                    where_clauses.append(f"{qk} = ?")
                    params.append(qv)

            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            count_row = conn.execute(f"SELECT COUNT(*) as cnt FROM objects{where_sql}", params).fetchone()
            total = count_row['cnt'] if count_row else 0

            rows = conn.execute(
                f"SELECT data, metadata, created_at, updated_at, expires_at, version, tags FROM objects{where_sql} LIMIT ? OFFSET ?",
                params + [limit, offset]
            ).fetchall()

            results = []
            for row in rows:
                meta = json.loads(row['metadata'])
                meta['created_at'] = row['created_at']
                meta['updated_at'] = row['updated_at']
                meta['expires_at'] = row['expires_at']
                meta['version'] = row['version']
                meta['tags'] = json.loads(row['tags'])
                results.append((row['data'], meta))

            return results, total
        except Exception:
            return [], 0


class DistributedStorage:
    def __init__(self, nodes: List[BaseStorage], replication_factor: int = 2):
        self.nodes = nodes
        self.replication_factor = replication_factor
        self._hash_ring = self._build_hash_ring()

    def _build_hash_ring(self) -> List[Tuple[int, BaseStorage]]:
        ring = []
        for i, node in enumerate(self.nodes):
            for vnode in range(100):
                hash_val = int(hashlib.md5(f"{i}:{vnode}".encode()).hexdigest(), 16)
                ring.append((hash_val, node))
        ring.sort()
        return ring

    def _get_nodes_for_key(self, key: str) -> List[BaseStorage]:
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        nodes = []

        for i, (ring_hash, node) in enumerate(self._hash_ring):
            if ring_hash >= hash_val:
                for j in range(self.replication_factor):
                    idx = (i + j) % len(self._hash_ring)
                    nodes.append(self._hash_ring[idx][1])
                break
        else:
            for j in range(self.replication_factor):
                nodes.append(self._hash_ring[j][1])

        return nodes[:self.replication_factor]

    def put(self, key: str, value: Any, **kwargs) -> bool:
        nodes = self._get_nodes_for_key(key)
        results = [node.put(key, value, **kwargs) for node in nodes]
        return any(results)

    def get(self, key: str, default: Any = None) -> Any:
        nodes = self._get_nodes_for_key(key)
        for node in nodes:
            value = node.get(key)
            if value is not None:
                return value
        return default

    def delete(self, key: str) -> bool:
        nodes = self._get_nodes_for_key(key)
        results = [node.delete(key) for node in nodes]
        return any(results)


class StorageManager:
    def __init__(self, default_config: StorageConfig = None):
        self.default_config = default_config or StorageConfig()
        self.storages: Dict[str, BaseStorage] = {}
        self.default_storage: Optional[BaseStorage] = None

    def add_storage(self, name: str, config: StorageConfig, set_default: bool = False) -> BaseStorage:
        if config.backend == StorageBackend.MEMORY:
            storage = MemoryStorage(config)
        elif config.backend == StorageBackend.SQLITE:
            storage = SQLiteStorage(config)
        else:
            storage = MemoryStorage(config)

        storage._connect()
        self.storages[name] = storage

        if set_default or self.default_storage is None:
            self.default_storage = storage

        return storage

    def get_storage(self, name: str) -> Optional[BaseStorage]:
        return self.storages.get(name)

    def put(self, key: str, value: Any, storage: str = None, **kwargs) -> bool:
        store = self.storages.get(storage) if storage else self.default_storage
        if store:
            return store.put(key, value, **kwargs)
        return False

    def get(self, key: str, default: Any = None, storage: str = None) -> Any:
        store = self.storages.get(storage) if storage else self.default_storage
        if store:
            return store.get(key, default)
        return default

    def delete(self, key: str, storage: str = None) -> bool:
        store = self.storages.get(storage) if storage else self.default_storage
        if store:
            return store.delete(key)
        return False

    def shutdown(self):
        for storage in self.storages.values():
            storage._disconnect()


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
