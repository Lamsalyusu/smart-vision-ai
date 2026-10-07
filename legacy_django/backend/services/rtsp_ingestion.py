from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable
from urllib.parse import urlsplit

import cv2

logger = logging.getLogger(__name__)

FrameProcessor = Callable[[str, int, bytes], None]
CaptureFactory = Callable[[str], Any]
FrameEncoder = Callable[[Any], bytes]
_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class RTSPStreamMetrics:
    camera_id: str
    connected: bool
    frames_received: int
    frames_processed: int
    frames_dropped: int
    reconnects: int
    processing_errors: int
    last_frame_at: float | None


class _RTSPStreamWorker:
    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        process_frame: FrameProcessor,
        capture_factory: CaptureFactory,
        encode_frame: FrameEncoder,
        reconnect_delay_seconds: float,
    ) -> None:
        self.camera_id = camera_id
        self.stream_url = stream_url
        self._process_frame = process_frame
        self._capture_factory = capture_factory
        self._encode_frame = encode_frame
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._stop = threading.Event()
        self._frames: Queue[tuple[int, Any]] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._capture: Any | None = None
        self._connected = False
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._reconnects = 0
        self._processing_errors = 0
        self._last_frame_at: float | None = None
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"rtsp-capture-{camera_id}",
            daemon=True,
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name=f"rtsp-process-{camera_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._capture_thread.start()
        self._processing_thread.start()

    def stop(self, join_timeout_seconds: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        for thread in (self._capture_thread, self._processing_thread):
            if thread.is_alive():
                thread.join(join_timeout_seconds)

    def metrics(self) -> RTSPStreamMetrics:
        with self._lock:
            return RTSPStreamMetrics(
                self.camera_id,
                self._connected,
                self._frames_received,
                self._frames_processed,
                self._frames_dropped,
                self._reconnects,
                self._processing_errors,
                self._last_frame_at,
            )

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._capture_factory(self.stream_url)
                with self._lock:
                    self._capture = capture
                    self._connected = bool(capture.isOpened())

                while self._connected and not self._stop.is_set():
                    success, frame = capture.read()
                    if not success or frame is None:
                        break
                    with self._lock:
                        self._frames_received += 1
                        sequence = self._frames_received
                        self._last_frame_at = time.time()
                    try:
                        self._frames.put_nowait((sequence, frame))
                    except Full:
                        try:
                            self._frames.get_nowait()
                        except Empty:
                            pass
                        with self._lock:
                            self._frames_dropped += 1
                        try:
                            self._frames.put_nowait((sequence, frame))
                        except Full:
                            with self._lock:
                                self._frames_dropped += 1
            except Exception as exc:
                logger.warning("RTSP capture failed camera=%s error=%s", self.camera_id, type(exc).__name__)
            finally:
                if capture is not None:
                    capture.release()
                with self._lock:
                    self._capture = None
                    self._connected = False

            if self._stop.is_set():
                break
            with self._lock:
                self._reconnects += 1
            self._stop.wait(self._reconnect_delay_seconds)

    def _processing_loop(self) -> None:
        while not self._stop.is_set() or not self._frames.empty():
            try:
                sequence, frame = self._frames.get(timeout=0.1)
            except Empty:
                continue
            try:
                encoded = self._encode_frame(frame)
                self._process_frame(self.camera_id, sequence, encoded)
                with self._lock:
                    self._frames_processed += 1
            except Exception as exc:
                logger.warning(
                    "RTSP frame processing failed camera=%s error=%s",
                    self.camera_id,
                    type(exc).__name__,
                )
                with self._lock:
                    self._processing_errors += 1


class RTSPIngestionManager:
    """Bounded RTSP readers with allowlisted endpoints and latest-frame backpressure."""

    def __init__(
        self,
        process_frame: FrameProcessor,
        *,
        allowed_hosts: set[str] | frozenset[str],
        maximum_streams: int = 50,
        reconnect_delay_seconds: float = 1.0,
        capture_factory: CaptureFactory | None = None,
        encode_frame: FrameEncoder | None = None,
    ) -> None:
        if maximum_streams < 1 or reconnect_delay_seconds <= 0:
            raise ValueError("stream limit and reconnect delay must be positive")
        self._process_frame = process_frame
        self._allowed_hosts = frozenset(host.strip().lower() for host in allowed_hosts if host.strip())
        self._maximum_streams = maximum_streams
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._capture_factory = capture_factory or (
            lambda url: cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        )
        self._encode_frame = encode_frame or self._encode_jpeg
        self._streams: dict[str, _RTSPStreamWorker] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _encode_jpeg(frame: Any) -> bytes:
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("camera frame could not be encoded")
        return encoded.tobytes()

    def register(self, camera_id: str, stream_url: str) -> RTSPStreamMetrics:
        if not _CAMERA_ID.fullmatch(camera_id):
            raise ValueError("camera id must use 1-64 letters, digits, dot, underscore, or hyphen")
        parsed = urlsplit(stream_url)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("stream URL must be a valid RTSP or RTSPS URL")
        host = parsed.netloc.rsplit("@", 1)[-1].lower()
        if not self._allowed_hosts or host not in self._allowed_hosts:
            raise PermissionError("RTSP endpoint is not in the configured host allowlist")

        with self._lock:
            if camera_id in self._streams:
                raise ValueError("camera stream is already registered")
            if len(self._streams) >= self._maximum_streams:
                raise OverflowError("maximum concurrent RTSP stream count reached")
            worker = _RTSPStreamWorker(
                camera_id,
                stream_url,
                self._process_frame,
                self._capture_factory,
                self._encode_frame,
                self._reconnect_delay_seconds,
            )
            self._streams[camera_id] = worker
            worker.start()
        return worker.metrics()

    def unregister(self, camera_id: str) -> bool:
        with self._lock:
            worker = self._streams.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        return True

    def metrics(self) -> tuple[RTSPStreamMetrics, ...]:
        with self._lock:
            streams = tuple(self._streams.values())
        return tuple(sorted((stream.metrics() for stream in streams), key=lambda item: item.camera_id))

    def close(self) -> None:
        with self._lock:
            camera_ids = tuple(self._streams)
        for camera_id in camera_ids:
            self.unregister(camera_id)
