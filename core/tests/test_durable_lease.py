import unittest

from core.infrastructure.scalability import DurableLeaseStore, create_memory_lease_store


class DurableLeaseStoreTests(unittest.TestCase):
    def test_second_worker_cannot_acquire_an_active_lease(self) -> None:
        store = create_memory_lease_store()
        first = store.acquire("camera:front", owner="worker-a", ttl_seconds=10, now_seconds=100)
        self.assertIsNotNone(first)
        assert first is not None

        blocked = store.acquire("camera:front", owner="worker-b", ttl_seconds=10, now_seconds=101)
        self.assertIsNone(blocked)

        takeover = store.acquire("camera:front", owner="worker-b", ttl_seconds=10, now_seconds=111)
        self.assertIsNotNone(takeover)
        assert takeover is not None
        self.assertEqual(takeover.fencing_token, first.fencing_token + 1)

    def test_leases_survive_a_new_store_instance_and_renewal(self) -> None:
        store = create_memory_lease_store()
        lease = store.acquire("scene-graph:write", owner="worker-a", ttl_seconds=5, now_seconds=10)
        assert lease is not None

        reopened = DurableLeaseStore(store.engine)
        persisted = reopened.current("scene-graph:write")
        self.assertEqual(persisted, lease)

        self.assertTrue(reopened.renew(lease, ttl_seconds=10, now_seconds=12))
        renewed = reopened.current("scene-graph:write")
        assert renewed is not None
        self.assertEqual(renewed.expires_at_seconds, 22)

        other = reopened.acquire(
            "scene-graph:write", owner="worker-b", ttl_seconds=5, now_seconds=13
        )
        self.assertIsNone(other)

    def test_wrong_owner_cannot_renew_or_release(self) -> None:
        store = create_memory_lease_store()
        lease = store.acquire("inference:gpu-0", owner="worker-a", ttl_seconds=30, now_seconds=1)
        assert lease is not None

        forged = type(lease)(
            lease.resource, "worker-b", lease.fencing_token, lease.expires_at_seconds
        )
        self.assertFalse(store.renew(forged, now_seconds=2))
        self.assertFalse(store.release(forged))
        self.assertTrue(store.release(lease))
        self.assertIsNotNone(
            store.acquire("inference:gpu-0", owner="worker-c", ttl_seconds=5, now_seconds=3)
        )

    def test_stale_release_cannot_steal_a_newer_lease(self) -> None:
        store = create_memory_lease_store()
        first = store.acquire("tracker:global", owner="worker-a", ttl_seconds=1, now_seconds=0)
        assert first is not None
        second = store.acquire("tracker:global", owner="worker-b", ttl_seconds=10, now_seconds=5)
        assert second is not None

        self.assertFalse(store.release(first))
        active = store.current("tracker:global")
        assert active is not None
        self.assertEqual(active.owner, "worker-b")

    def test_validates_inputs(self) -> None:
        store = create_memory_lease_store()
        with self.assertRaises(ValueError):
            store.acquire("")
        with self.assertRaises(ValueError):
            store.acquire("resource", ttl_seconds=0)
        with self.assertRaises(ValueError):
            store.acquire("resource", now_seconds=float("nan"))


if __name__ == "__main__":
    unittest.main()
