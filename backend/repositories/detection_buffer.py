import threading
import time
import logging
from backend.repositories.data_manager import data_manager

logger = logging.getLogger(__name__)


class DetectionBatchBuffer:
    """Thread-safe write-behind telemetry buffer for high-frequency camera inference."""

    def __init__(self, flush_interval_sec=3.0, max_batch_size=20):
        self.flush_interval_sec = flush_interval_sec
        self.max_batch_size = max_batch_size
        self._buffer = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def add(self, user_id, detection):
        """Queue detection record and flush if batch criteria met."""
        if not user_id:
            return

        record = {
            "user_id": str(user_id),
            "object_class": detection.get("class", "unknown"),
            "confidence": round(float(detection.get("confidence", 0.0)), 4),
            "bbox": detection.get("bbox"),
        }

        with self._lock:
            self._buffer.append(record)
            should_flush = (
                len(self._buffer) >= self.max_batch_size
                or (time.time() - self._last_flush) >= self.flush_interval_sec
            )

        if should_flush:
            self.flush()

    def flush(self):
        """Asynchronously commit buffered telemetry."""
        with self._lock:
            if not self._buffer:
                return
            records_to_send = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

        def _persist_records():
            try:
                for rec in records_to_send:
                    data_manager.save_detection_record(
                        user_id=rec["user_id"],
                        object_class=rec["object_class"],
                        confidence=rec["confidence"],
                        bbox=rec.get("bbox"),
                    )
                logger.debug("Successfully flushed %d detection telemetry records", len(records_to_send))
            except Exception:
                logger.debug("Telemetry buffer persistence encountered transient error")

        threading.Thread(target=_persist_records, daemon=True).start()


detection_buffer = DetectionBatchBuffer()
