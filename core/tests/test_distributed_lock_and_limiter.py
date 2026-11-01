import unittest

from core.infrastructure.scalability.distributed_lock import DistributedLock
from core.infrastructure.scalability.rate_limiter import SlidingWindowRateLimiter


class DistributedLockAndLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_distributed_lock_mutual_exclusion(self):
        lock1 = DistributedLock(resource_key="camera_stream_101", ttl_seconds=2.0)
        lock2 = DistributedLock(resource_key="camera_stream_101", ttl_seconds=2.0)

        # First lock acquires immediately
        self.assertTrue(await lock1.acquire(timeout_seconds=0.1))

        # Second lock fails to acquire while lock1 is active
        self.assertFalse(await lock2.acquire(timeout_seconds=0.1))

        # Release first lock
        await lock1.release()

        # Second lock can now acquire
        self.assertTrue(await lock2.acquire(timeout_seconds=0.1))
        await lock2.release()

    def test_sliding_window_rate_limiter(self):
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10.0)
        client = "client_ip_192.168.1.50"

        # 3 requests allowed
        allowed1, rem1, _ = limiter.is_allowed(client, now=100.0)
        self.assertTrue(allowed1)
        self.assertEqual(rem1, 2)

        allowed2, rem2, _ = limiter.is_allowed(client, now=101.0)
        self.assertTrue(allowed2)
        self.assertEqual(rem2, 1)

        allowed3, rem3, _ = limiter.is_allowed(client, now=102.0)
        self.assertTrue(allowed3)
        self.assertEqual(rem3, 0)

        # 4th request blocked
        allowed4, rem4, _ = limiter.is_allowed(client, now=103.0)
        self.assertFalse(allowed4)
        self.assertEqual(rem4, 0)

        # After window slides (now=111.0 > 100.0 + 10.0), capacity freed
        allowed5, _rem5, _ = limiter.is_allowed(client, now=111.0)
        self.assertTrue(allowed5)


if __name__ == "__main__":
    unittest.main()
