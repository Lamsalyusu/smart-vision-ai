from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.application.perception.pipeline_orchestrator import (
    PerceptionPipelineOrchestrator,
    PipelineExecutionSnapshot,
)
from core.infrastructure.concurrency.actor import Actor, MailboxFullStrategy


@dataclass(frozen=True, slots=True)
class ProcessFrameMessage:
    frame_sequence: int
    frame_data: Any
    timestamp_seconds: float
    simulated_embeddings: Sequence[Sequence[float]] | None = None
    estimated_depth_m: float = 3.0


class PerceptionPipelineActor(Actor[ProcessFrameMessage]):
    """Actor managing a camera's perception pipeline instance asynchronously."""

    def __init__(
        self,
        actor_id: str,
        orchestrator: PerceptionPipelineOrchestrator,
        mailbox_capacity: int = 50,
        strategy: MailboxFullStrategy = MailboxFullStrategy.DROP_OLDEST,
    ) -> None:
        super().__init__(actor_id=actor_id, mailbox_capacity=mailbox_capacity, strategy=strategy)
        self.orchestrator = orchestrator

    async def receive(self, message: ProcessFrameMessage) -> PipelineExecutionSnapshot:
        """Process incoming frame asynchronously within actor context."""
        return self.orchestrator.process_frame(
            frame_sequence=message.frame_sequence,
            frame_data=message.frame_data,
            timestamp_seconds=message.timestamp_seconds,
            simulated_embeddings=message.simulated_embeddings,
            estimated_depth_m=message.estimated_depth_m,
        )
