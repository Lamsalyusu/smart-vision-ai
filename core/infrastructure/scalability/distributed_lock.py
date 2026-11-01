from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, ClassVar, Self

logger = logging.getLogger(__name__)

class DistributedLock:
    """Enterprise Distributed Lock supporting Redis backend with safe in-memory fallback."""

    # Class-level shared in-memory lock state: resource_key -> (Lock, expiry_time, owner_token)
    _shared_memory_locks: ClassVar[dict[str, tuple[asyncio.Lock, float, str]]] = {}

    def __init__(self, resource_key: str, ttl_seconds: float = 10.0, redis_client: Any | None = None) -> None:
        self.resource_key = f"lock:{resource_key}"
        self.ttl = ttl_seconds
        self.lock_token = str(uuid.uuid4())
        self._redis = redis_client
        self._is_acquired = False

    async def acquire(self, timeout_seconds: float = 5.0) -> bool:
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_seconds:
            # Try Redis if available
            if self._redis is not None:
                try:
                    res = await self._redis.set(self.resource_key, self.lock_token, px=int(self.ttl * 1000), nx=True)
                    if res:
                        self._is_acquired = True
                        return True
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Redis lock acquisition failed; using memory fallback: %s", exc)

            # In-memory fallback
            now = time.monotonic()
            lock_entry = self._shared_memory_locks.get(self.resource_key)

            if lock_entry is None or now > lock_entry[1]:
                lock = lock_entry[0] if lock_entry else asyncio.Lock()
                if not lock.locked():
                    await lock.acquire()
                    self._shared_memory_locks[self.resource_key] = (lock, now + self.ttl, self.lock_token)
                    self._is_acquired = True
                    return True

            await asyncio.sleep(0.02)

        return False

    async def release(self) -> bool:
        if not self._is_acquired:
            return False

        if self._redis is not None:
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                await self._redis.eval(lua_script, 1, self.resource_key, self.lock_token)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Redis lock release failed; clearing local lock state: %s", exc)

        lock_entry = self._shared_memory_locks.get(self.resource_key)
        if lock_entry and lock_entry[2] == self.lock_token:
            lock, _, _ = lock_entry
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass
            self._shared_memory_locks.pop(self.resource_key, None)

        self._is_acquired = False
        return True

    async def __aenter__(self) -> Self:
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(f"Could not acquire distributed lock for resource '{self.resource_key}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release()
