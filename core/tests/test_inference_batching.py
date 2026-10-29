import unittest

from core.application.inference import InferenceRequest, PriorityBatcher


def request(request_id: str, model: str, priority: int, timestamp: float) -> InferenceRequest[str]:
    return InferenceRequest(request_id, model, request_id, priority, timestamp)


class PriorityBatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batcher: PriorityBatcher[str] = PriorityBatcher(
            max_batch_size=2,
            max_queue_delay_seconds=1.0,
            max_queue_size=8,
        )

    def test_waits_until_batch_full_or_queue_deadline(self) -> None:
        self.batcher.enqueue(request("a", "detector", 1, 0))

        self.assertIsNone(self.batcher.take_ready_batch(0.5))
        batch = self.batcher.take_ready_batch(1.0)

        self.assertEqual([item.request_id for item in batch], ["a"])

    def test_schedules_highest_priority_ready_model_first(self) -> None:
        self.batcher.enqueue(request("low", "model-a", 1, 0))
        self.batcher.enqueue(request("high", "model-b", 8, 0))

        batch = self.batcher.take_ready_batch(1, force=True)

        self.assertEqual([item.request_id for item in batch], ["high"])
        self.assertEqual(self.batcher.queue_length, 1)

    def test_batches_do_not_mix_models_and_order_by_priority(self) -> None:
        self.batcher.enqueue(request("first", "detector", 1, 0))
        self.batcher.enqueue(request("high", "detector", 10, 0.1))
        self.batcher.enqueue(request("other", "depth", 0, 0))

        batch = self.batcher.take_ready_batch(2)

        self.assertEqual([item.request_id for item in batch], ["high", "first"])
        self.assertTrue(all(item.model_name == "detector" for item in batch))
        self.assertEqual(self.batcher.queue_length, 1)

    def test_rejects_duplicate_ids_and_queue_overflow(self) -> None:
        small: PriorityBatcher[str] = PriorityBatcher(1, 2, 1)
        small.enqueue(request("a", "detector", 1, 0))
        with self.assertRaisesRegex(ValueError, "already queued"):
            small.enqueue(request("a", "depth", 1, 0))
        with self.assertRaises(OverflowError):
            small.enqueue(request("b", "detector", 1, 0))

    def test_drains_multiple_batches_without_losing_requests(self) -> None:
        for request_id in ("a", "b", "c", "d", "e"):
            self.batcher.enqueue(request(request_id, "detector", 1, 0))

        first = self.batcher.take_ready_batch(0)
        second = self.batcher.take_ready_batch(0)
        final = self.batcher.take_ready_batch(0, force=True)

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual(len(final), 1)
        self.assertEqual(self.batcher.queue_length, 0)

    def test_rejects_invalid_current_time(self) -> None:
        with self.assertRaises(ValueError):
            self.batcher.take_ready_batch(float("nan"))


if __name__ == "__main__":
    unittest.main()