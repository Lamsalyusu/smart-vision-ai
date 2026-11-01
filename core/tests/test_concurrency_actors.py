import asyncio
import unittest

from core.infrastructure.concurrency.actor import (
    Actor,
    ActorStatus,
    ActorSupervisor,
    MailboxFullStrategy,
    SupervisorStrategy,
)


class MathActor(Actor[int]):
    def __init__(self, actor_id: str, mailbox_capacity: int = 10, strategy: MailboxFullStrategy = MailboxFullStrategy.BLOCK) -> None:
        super().__init__(actor_id, mailbox_capacity=mailbox_capacity, strategy=strategy)
        self.state = 0

    async def receive(self, message: int) -> int:
        if message < 0:
            raise ValueError("Negative numbers not allowed")
        self.state += message
        return self.state


class ActorConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_actor_lifecycle_and_tell_ask(self) -> None:
        actor = MathActor("math-1")
        self.assertEqual(actor.status, ActorStatus.INITIALIZED)

        await actor.start()
        self.assertEqual(actor.status, ActorStatus.RUNNING)
        self.assertTrue(actor.is_alive)

        # Tell
        await actor.tell(5)
        # Ask
        result = await actor.ask(10, timeout_seconds=2.0)
        self.assertEqual(result, 15)
        self.assertEqual(actor.processed_count, 2)

        await actor.stop()
        self.assertEqual(actor.status, ActorStatus.STOPPED)
        self.assertFalse(actor.is_alive)

    async def test_actor_ask_timeout(self) -> None:
        class SlowActor(Actor[str]):
            async def receive(self, message: str) -> str:
                await asyncio.sleep(0.5)
                return message

        actor = SlowActor("slow-1")
        await actor.start()

        with self.assertRaises(TimeoutError):
            await actor.ask("hello", timeout_seconds=0.05)

        await actor.stop()

    async def test_actor_drop_oldest_backpressure(self) -> None:
        actor = MathActor("drop-actor", mailbox_capacity=2, strategy=MailboxFullStrategy.DROP_OLDEST)
        # Fill mailbox before starting so messages queue up
        await actor.start()
        # Pause actor loop by making it sleep on first message
        class BlockableActor(Actor[int]):
            def __init__(self) -> None:
                super().__init__("blockable", mailbox_capacity=2, strategy=MailboxFullStrategy.DROP_OLDEST)
                self.processed = []

            async def receive(self, message: int) -> None:
                await asyncio.sleep(0.05)
                self.processed.append(message)

        b_actor = BlockableActor()
        await b_actor.start()

        # Enqueue 5 items rapidly with mailbox capacity 2
        for i in range(5):
            await b_actor.tell(i)

        await asyncio.sleep(0.2)
        await b_actor.stop()

        self.assertGreater(b_actor.dropped_count, 0)

    async def test_supervisor_restarts_failed_actor(self) -> None:
        supervisor = ActorSupervisor(strategy=SupervisorStrategy.RESTART, max_restarts=3)
        actor = MathActor("supervised-1")
        supervisor.register(actor)

        await supervisor.start_all()
        self.assertTrue(actor.is_alive)

        # Send fatal message
        with self.assertRaises(ValueError):
            await actor.ask(-1)

        # Actor encountered error; supervisor heals
        healed = await supervisor.check_and_heal()
        self.assertIn("supervised-1", healed)
        self.assertTrue(actor.is_alive)
        await supervisor.stop_all()


if __name__ == "__main__":
    unittest.main()
