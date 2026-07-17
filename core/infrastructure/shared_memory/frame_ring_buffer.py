from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import struct


class FramePixelFormat(IntEnum):
    NV12 = 1
    RGB24 = 2
    BGR24 = 3


@dataclass(frozen=True, slots=True)
class FrameDescriptor:
    camera_id: str
    sequence_number: int
    width: int
    height: int
    pixel_format: FramePixelFormat
    captured_at_ns: int


class FrameLease:
    def __init__(
        self,
        descriptor: FrameDescriptor,
        view: memoryview,
        lock: multiprocessing.synchronize.Lock,
        state_view: memoryview,
    ) -> None:
        self.descriptor = descriptor
        self._view = view.toreadonly()
        self._lock = lock
        self._state_view = state_view
        self._released = False

    @property
    def data(self) -> memoryview:
        if self._released:
            raise RuntimeError("frame lease has been released")
        return self._view

    def release(self) -> None:
        if self._released:
            return
        self._view.release()
        with self._lock:
            self._state_view[0] = SharedFrameRingBuffer.EMPTY
        self._state_view.release()
        self._released = True

    def __enter__(self) -> "FrameLease":
        if self._released:
            raise RuntimeError("frame lease has been released")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class SharedFrameRingBuffer:
    """Fixed-capacity POSIX shared-memory ring with leased zero-copy reads.

    Pass the same multiprocessing lock to every process attaching to a named
    ring. Writers replace the oldest unclaimed frame when full, never a leased
    slot; writes return False when every slot is currently leased.
    """

    EMPTY = 0
    READY = 1
    READING = 2
    _HEADER = struct.Struct("!BQQIIB64s")
    _FORMATS = {
        FramePixelFormat.NV12: 1,
        FramePixelFormat.RGB24: 2,
        FramePixelFormat.BGR24: 3,
    }
    _FORMATS_BY_CODE = {value: key for key, value in _FORMATS.items()}

    def __init__(
        self,
        capacity: int,
        max_frame_bytes: int,
        *,
        name: str | None = None,
        create: bool = True,
        lock: multiprocessing.synchronize.Lock | None = None,
    ) -> None:
        if capacity < 1 or max_frame_bytes < 1:
            raise ValueError("ring capacity and maximum frame size must be positive")
        if not create and lock is None:
            raise ValueError("attaching processes must share the creator's lock")

        self.capacity = capacity
        self.max_frame_bytes = max_frame_bytes
        self._slot_size = self._HEADER.size + max_frame_bytes
        self._lock = lock or multiprocessing.Lock()
        self._owner = create
        self._memory = SharedMemory(
            name=name,
            create=create,
            size=capacity * self._slot_size if create else 0,
        )
        if not create and self._memory.size != capacity * self._slot_size:
            self._memory.close()
            raise ValueError("shared-memory region size does not match ring configuration")
        self._buffer = self._memory.buf
        if create:
            for index in range(capacity):
                self._slot_state(index)[0] = self.EMPTY

    @property
    def name(self) -> str:
        return self._memory.name

    def _slot_offset(self, index: int) -> int:
        return index * self._slot_size

    def _slot_state(self, index: int) -> memoryview:
        return self._buffer[self._slot_offset(index) : self._slot_offset(index) + 1]

    def write_frame(
        self,
        *,
        camera_id: str,
        sequence_number: int,
        width: int,
        height: int,
        pixel_format: FramePixelFormat,
        captured_at_ns: int,
        data: bytes | bytearray | memoryview,
    ) -> bool:
        self._validate_frame(camera_id, sequence_number, width, height, captured_at_ns, data)
        try:
            format_code = self._FORMATS[pixel_format]
        except KeyError as exc:
            raise ValueError("unsupported frame pixel format") from exc

        frame_view = memoryview(data)
        if frame_view.nbytes > self.max_frame_bytes:
            raise ValueError("frame exceeds configured slot capacity")
        expected_bytes = self._expected_size(width, height, pixel_format)
        if pixel_format is FramePixelFormat.NV12 and (width % 2 or height % 2):
            raise ValueError("NV12 frame dimensions must be even")
        if frame_view.nbytes != expected_bytes:
            raise ValueError(f"frame data size must be {expected_bytes} bytes for this format")

        encoded_camera_id = camera_id.encode("utf-8")
        if len(encoded_camera_id) > 64:
            raise ValueError("camera id must fit in 64 UTF-8 bytes")

        with self._lock:
            empty_slots = [
                index for index in range(self.capacity)
                if self._slot_state(index)[0] == self.EMPTY
            ]
            ready_slots = [
                index for index in range(self.capacity)
                if self._slot_state(index)[0] == self.READY
            ]
            if empty_slots:
                slot_index = empty_slots[0]
            elif ready_slots:
                slot_index = min(ready_slots, key=self._read_sequence)
            else:
                return False

            offset = self._slot_offset(slot_index)
            self._slot_state(slot_index)[0] = self.EMPTY
            self._HEADER.pack_into(
                self._buffer,
                offset,
                self.EMPTY,
                sequence_number,
                captured_at_ns,
                width,
                height,
                format_code,
                encoded_camera_id.ljust(64, b"\0"),
            )
            payload_start = offset + self._HEADER.size
            self._buffer[payload_start : payload_start + frame_view.nbytes] = frame_view
            self._slot_state(slot_index)[0] = self.READY
            return True

    def read_frame(self) -> FrameLease | None:
        with self._lock:
            ready_slots = [
                index for index in range(self.capacity)
                if self._slot_state(index)[0] == self.READY
            ]
            if not ready_slots:
                return None

            slot_index = min(ready_slots, key=self._read_sequence)
            offset = self._slot_offset(slot_index)
            _, sequence_number, captured_at_ns, width, height, format_code, raw_camera_id = (
                self._HEADER.unpack_from(self._buffer, offset)
            )
            try:
                pixel_format = self._FORMATS_BY_CODE[format_code]
            except KeyError as exc:
                raise RuntimeError("shared-memory slot contains an unknown pixel format") from exc
            camera_id = raw_camera_id.rstrip(b"\0").decode("utf-8")
            descriptor = FrameDescriptor(
                camera_id=camera_id,
                sequence_number=sequence_number,
                width=width,
                height=height,
                pixel_format=pixel_format,
                captured_at_ns=captured_at_ns,
            )
            self._slot_state(slot_index)[0] = self.READING
            payload_start = offset + self._HEADER.size
            payload_size = self._expected_size(width, height, pixel_format)
            payload = self._buffer[payload_start : payload_start + payload_size]
            return FrameLease(descriptor, payload, self._lock, self._slot_state(slot_index))

    def _read_sequence(self, index: int) -> int:
        return self._HEADER.unpack_from(self._buffer, self._slot_offset(index))[1]

    @staticmethod
    def _expected_size(width: int, height: int, pixel_format: FramePixelFormat) -> int:
        if pixel_format is FramePixelFormat.NV12:
            return width * height * 3 // 2
        return width * height * 3

    @staticmethod
    def _validate_frame(
        camera_id: str,
        sequence_number: int,
        width: int,
        height: int,
        captured_at_ns: int,
        data: bytes | bytearray | memoryview,
    ) -> None:
        if not camera_id or camera_id.strip() != camera_id:
            raise ValueError("camera id must be non-empty and have no surrounding whitespace")
        if (
            isinstance(sequence_number, bool)
            or not isinstance(sequence_number, int)
            or isinstance(captured_at_ns, bool)
            or not isinstance(captured_at_ns, int)
            or min(sequence_number, captured_at_ns) < 0
        ):
            raise ValueError("sequence and capture timestamp must be non-negative")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width < 1
            or height < 1
        ):
            raise ValueError("frame dimensions must be positive")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("frame data must expose a byte buffer")
    def close(self) -> None:
        self._buffer.release()
        self._memory.close()

    def unlink(self) -> None:
        if self._owner:
            self._memory.unlink()