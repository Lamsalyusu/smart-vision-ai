import time
import unittest

import numpy as np

from core.container import container
from core.domain.analytics.geofence import GeofenceZone, ZoneType
from core.domain.ports.event_bus import DomainEvent
from core.domain.tracking.hungarian_fusion import DetectionCandidate, TrackTarget


class PipelineFinishedEvent(DomainEvent):
    def __init__(self, event_id: str, tracks_count: int):
        super().__init__(event_id=event_id)
        object.__setattr__(self, "tracks_count", tracks_count)


class EndToEndPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_autonomous_vision_pipeline(self):
        # 1. Setup Domain Services via DI Container
        matcher = container.matcher
        event_bus = container.event_bus
        uow = container.create_uow()

        # Event Bus subscription
        received_events = []

        async def on_pipeline_finished(event: PipelineFinishedEvent):
            received_events.append(event)

        event_bus.subscribe(PipelineFinishedEvent, on_pipeline_finished)

        # 2. Simulated Ingested Detections
        det_face = DetectionCandidate(
            detection_id="det_01",
            class_name="human_face",
            bbox=(100.0, 100.0, 200.0, 200.0),
            feature_embedding=np.array([0.9, 0.1, 0.0]),
        )
        det_knife = DetectionCandidate(
            detection_id="det_02",
            class_name="knife",
            bbox=(220.0, 180.0, 260.0, 240.0),
            feature_embedding=np.array([0.0, 0.9, 0.1]),
        )

        # 3. Active UKF Tracks
        track1 = TrackTarget(
            track_id=1,
            class_name="human_face",
            state_mean=np.array([150.0, 150.0, 10000.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            state_covariance=np.eye(8) * 5.0,
            feature_gallery=[np.array([0.9, 0.1, 0.0])],
        )

        # 4. Multi-Modal Hungarian Fusion Matching
        matched, _unmatched_t, unmatched_d = matcher.match([track1], [det_face, det_knife])
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0], (0, 0))  # Track #1 matched to det_face
        self.assertIn(1, unmatched_d)         # det_knife is a new unconfirmed track candidate

        # 5. Geofence & Exclusion Zone Analytics
        vault_zone = GeofenceZone(
            zone_id="zone_vault",
            name="Bank Vault",
            polygon_vertices=[(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)],
            zone_type=ZoneType.CRITICAL,
        )
        breach = vault_zone.evaluate_track(track_id=1, point=(150.0, 150.0), now=time.monotonic())
        self.assertIsNotNone(breach)
        self.assertEqual(breach.zone_id, "zone_vault")

        # 6. Database Transactional Persistence via Unit of Work
        with uow:
            uow.telemetry.record_detection({
                "class": "human_face",
                "confidence": 0.95,
                "track_id": 1,
                "zone_breach": breach.zone_id,
            })
            uow.audits.log_event(
                event_type="ZONE_SECURITY_BREACH",
                user_id="system_tracker",
                details={"zone": breach.zone_id, "severity": breach.severity.value},
            )
            uow.commit()

        # 7. Asynchronous Domain Event Broadcast
        await event_bus.publish(PipelineFinishedEvent(event_id="evt_pipe_001", tracks_count=1))
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].tracks_count, 1)


if __name__ == "__main__":
    unittest.main()
