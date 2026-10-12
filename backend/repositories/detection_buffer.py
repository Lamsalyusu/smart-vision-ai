import threading
import time
import logging
from backend.supabase_client import supabase_admin

logger = logging.getLogger(__name__)


class DetectionBatchBuffer:
    """Thread-safe batching buffer for high-frequency detection telemetry."""

    def __init__(self, flush_interval_sec=4.0, max_batch_size=25):
        self.flush_interval_sec = flush_interval_sec
        self.max_batch_size = max_batch_size
        self._buffer = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

    def add(self, user_id, detection):
        """Append detection record to buffer and flush if batch size threshold is reached."""
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
        """Flush accumulated records to Supabase in a single batch request."""
        with self._lock:
            if not self._buffer:
                return
            records_to_send = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

        def _async_send():
            if not supabase_admin:
                return
            try:
                supabase_admin.table("detection_history").insert(records_to_send).execute()
                logger.debug("Flushed %d detection records to database", len(records_to_send))
            except Exception:
                logger.debug("Failed to flush detection batch buffer")

        threading.Thread(target=_async_send, daemon=True).start()


detection_buffer = DetectionBatchBuffer()
