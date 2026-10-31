import math
import threading
from dataclasses import dataclass
from enum import Enum


class VideoCodec(str, Enum):
    VP8 = "vp8"
    H264 = "h264"


@dataclass(frozen=True, slots=True)
class CameraStream:
    camera_id: str
    codec: VideoCodec
    width: int
    height: int
    target_bitrate_bps: int


@dataclass(frozen=True, slots=True)
class BitrateDecision:
    previous_bitrate_bps: int
    target_bitrate_bps: int
    congested: bool
    reason: str


class AdaptiveBitrateController:
    """AIMD target bitrate controller driven by receiver-side congestion signals."""

    def __init__(
        self,
        initial_bitrate_bps: int,
        *,
        minimum_bitrate_bps: int = 150_000,
        maximum_bitrate_bps: int = 8_000_000,
        additive_step_bps: int = 100_000,
        multiplicative_decrease: float = 0.7,
        target_latency_ms: float = 120.0,
        maximum_queue_depth: int = 3,
        maximum_packet_loss_ratio: float = 0.03,
    ) -> None:
        if not 0 < minimum_bitrate_bps <= initial_bitrate_bps <= maximum_bitrate_bps:
            raise ValueError("bitrate must satisfy minimum <= initial <= maximum")
        if additive_step_bps < 1 or maximum_queue_depth < 0:
            raise ValueError("additive step must be positive and queue limit non-negative")
        if not 0 < multiplicative_decrease < 1:
            raise ValueError("multiplicative decrease must be between zero and one")
        if not math.isfinite(target_latency_ms) or target_latency_ms <= 0:
            raise ValueError("target latency must be finite and positive")
        if not 0 <= maximum_packet_loss_ratio <= 1:
            raise ValueError("packet-loss threshold must be between zero and one")

        self.minimum_bitrate_bps = minimum_bitrate_bps
        self.maximum_bitrate_bps = maximum_bitrate_bps
        self.additive_step_bps = additive_step_bps
        self.multiplicative_decrease = multiplicative_decrease
        self.target_latency_ms = target_latency_ms
        self.maximum_queue_depth = maximum_queue_depth
        self.maximum_packet_loss_ratio = maximum_packet_loss_ratio
        self._bitrate_bps = initial_bitrate_bps

    @property
    def target_bitrate_bps(self) -> int:
        return self._bitrate_bps

    def observe(
        self,
        *,
        latency_ms: float,
        queue_depth: int,
        packet_loss_ratio: float,
    ) -> BitrateDecision:
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency must be finite and non-negative")
        if queue_depth < 0:
            raise ValueError("queue depth must be non-negative")
        if not math.isfinite(packet_loss_ratio) or not 0 <= packet_loss_ratio <= 1:
            raise ValueError("packet loss ratio must be between zero and one")

        previous = self._bitrate_bps
        reasons = []
        if latency_ms > self.target_latency_ms:
            reasons.append("latency")
        if queue_depth > self.maximum_queue_depth:
            reasons.append("queue_depth")
        if packet_loss_ratio > self.maximum_packet_loss_ratio:
            reasons.append("packet_loss")

        if reasons:
            self._bitrate_bps = max(
                self.minimum_bitrate_bps,
                int(self._bitrate_bps * self.multiplicative_decrease),
            )
            return BitrateDecision(previous, self._bitrate_bps, True, "+".join(reasons))

        self._bitrate_bps = min(
            self.maximum_bitrate_bps,
            self._bitrate_bps + self.additive_step_bps,
        )
        return BitrateDecision(previous, self._bitrate_bps, False, "healthy")


class CameraStreamRegistry:
    """Thread-safe metadata registry for independent camera media channels."""

    def __init__(self, maximum_streams: int = 50) -> None:
        if maximum_streams < 1:
            raise ValueError("maximum stream count must be positive")
        self.maximum_streams = maximum_streams
        self._streams: dict[str, CameraStream] = {}
        self._lock = threading.RLock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._streams)

    def register(self, stream: CameraStream) -> None:
        if not stream.camera_id or stream.camera_id.strip() != stream.camera_id:
            raise ValueError("camera id must be non-empty and have no surrounding whitespace")
        if stream.width < 1 or stream.height < 1 or stream.target_bitrate_bps < 1:
            raise ValueError("stream dimensions and target bitrate must be positive")
        with self._lock:
            if stream.camera_id in self._streams:
                raise ValueError("camera stream is already registered")
            if len(self._streams) >= self.maximum_streams:
                raise OverflowError("maximum concurrent camera stream count reached")
            self._streams[stream.camera_id] = stream

    def get(self, camera_id: str) -> CameraStream | None:
        with self._lock:
            return self._streams.get(camera_id)

    def update_bitrate(self, camera_id: str, target_bitrate_bps: int) -> CameraStream:
        if target_bitrate_bps < 1:
            raise ValueError("target bitrate must be positive")
        with self._lock:
            stream = self._streams.get(camera_id)
            if stream is None:
                raise KeyError(camera_id)
            updated = CameraStream(
                stream.camera_id,
                stream.codec,
                stream.width,
                stream.height,
                target_bitrate_bps,
            )
            self._streams[camera_id] = updated
            return updated

    def unregister(self, camera_id: str) -> CameraStream | None:
        with self._lock:
            return self._streams.pop(camera_id, None)

    def snapshot(self) -> tuple[CameraStream, ...]:
        with self._lock:
            return tuple(self._streams.values())