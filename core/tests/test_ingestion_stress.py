import unittest

from core.application.simulation import run_ingestion_stress


class IngestionStressTests(unittest.TestCase):
    def test_runs_one_hundred_streams_and_reports_latency(self) -> None:
        report = run_ingestion_stress(
            stream_count=100,
            frames_per_stream=3,
            worker_count=16,
        )

        self.assertEqual(report.submitted_frames, 300)
        self.assertEqual(report.accepted_frames, 300)
        self.assertEqual(report.dropped_frames, 0)
        self.assertGreater(report.throughput_fps, 0)
        self.assertGreaterEqual(report.p99_latency_ms, report.p50_latency_ms)

    def test_reports_ring_backpressure_drops(self) -> None:
        report = run_ingestion_stress(
            stream_count=4,
            frames_per_stream=3,
            worker_count=4,
            ring_capacity=1,
            admission_capacity_per_stream=1,
        )

        self.assertEqual(report.submitted_frames, 12)
        self.assertEqual(report.accepted_frames + report.dropped_frames, 12)
        self.assertGreater(report.dropped_frames, 0)

    def test_rejects_invalid_worker_count(self) -> None:
        with self.assertRaises(ValueError):
            run_ingestion_stress(stream_count=1, frames_per_stream=1, worker_count=0)


if __name__ == "__main__":
    unittest.main()