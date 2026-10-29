import threading
import unittest

from core.infrastructure.resilience import CircuitBreaker, CircuitOpenError, CircuitState


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_seconds=2,
            max_recovery_timeout_seconds=8,
            jitter_ratio=0.25,
            clock=self.clock,
            jitter_source=lambda low, high: high,
        )

    def test_opens_after_failure_threshold_and_uses_jittered_delay(self) -> None:
        operation = lambda: (_ for _ in ()).throw(ValueError("down"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        with self.assertRaises(CircuitOpenError) as raised:
            self.breaker.call(lambda: "unreachable")
        self.assertEqual(raised.exception.retry_after_seconds, 2.5)

    def test_open_circuit_can_use_fallback(self) -> None:
        operation = lambda: (_ for _ in ()).throw(ValueError("down"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)

        result = self.breaker.call(lambda: "unreachable", lambda error: error.retry_after_seconds)

        self.assertEqual(result, 2.5)

    def test_half_open_probe_success_closes_and_resets_backoff(self) -> None:
        operation = lambda: (_ for _ in ()).throw(ValueError("down"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)
        self.clock.now = 2.5

        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)
        self.assertEqual(self.breaker.call(lambda: "ok"), "ok")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)
        with self.assertRaises(CircuitOpenError) as raised:
            self.breaker.call(lambda: None)
        self.assertEqual(raised.exception.retry_after_seconds, 2.5)

    def test_failed_half_open_probe_doubles_recovery_delay(self) -> None:
        operation = lambda: (_ for _ in ()).throw(ValueError("down"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)
        self.clock.now = 2.5
        with self.assertRaises(ValueError):
            self.breaker.call(operation)

        with self.assertRaises(CircuitOpenError) as raised:
            self.breaker.call(lambda: None)
        self.assertEqual(raised.exception.retry_after_seconds, 5.0)

    def test_allows_only_one_half_open_probe(self) -> None:
        operation = lambda: (_ for _ in ()).throw(ValueError("down"))
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker.call(operation)
        self.clock.now = 2.5
        entered = threading.Event()
        release = threading.Event()

        def probe() -> None:
            entered.set()
            release.wait(timeout=2)
            return None

        worker = threading.Thread(target=lambda: self.breaker.call(probe))
        worker.start()
        self.assertTrue(entered.wait(timeout=2))
        with self.assertRaises(CircuitOpenError):
            self.breaker.call(lambda: None)
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaises(ValueError):
            CircuitBreaker(jitter_ratio=1.1)


if __name__ == "__main__":
    unittest.main()