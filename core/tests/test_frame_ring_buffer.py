import unittest

from core.infrastructure.shared_memory import FramePixelFormat, SharedFrameRingBuffer


class SharedFrameRingBufferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = SharedFrameRingBuffer(capacity=2, max_frame_bytes=64)

    def tearDown(self) -> None:
        self.ring.close()
        self.ring.unlink()

    def write_rgb(self, sequence_number: int, value: int = 0) -> bool:
        return self.ring.write_frame(
            camera_id="camera-1",
            sequence_number=sequence_number,
            width=2,
            height=2,
            pixel_format=FramePixelFormat.RGB24,
            captured_at_ns=sequence_number * 100,
            data=bytes([value]) * 12,
        )

    def test_reads_zero_copy_frame_and_descriptor(self) -> None:
        self.assertTrue(self.write_rgb(7, 9))

        with self.ring.read_frame() as lease:
            self.assertEqual(bytes(lease.data), bytes([9]) * 12)
            self.assertTrue(lease.data.readonly)
            self.assertEqual(lease.descriptor.sequence_number, 7)
            self.assertEqual(lease.descriptor.camera_id, "camera-1")
            self.assertEqual(lease.descriptor.pixel_format, FramePixelFormat.RGB24)

        self.assertIsNone(self.ring.read_frame())

    def test_overwrites_oldest_unclaimed_frame_when_full(self) -> None:
        self.write_rgb(1, 1)
        self.write_rgb(2, 2)
        self.write_rgb(3, 3)

        with self.ring.read_frame() as lease:
            self.assertEqual(lease.descriptor.sequence_number, 2)
            self.assertEqual(bytes(lease.data), bytes([2]) * 12)

    def test_never_overwrites_a_leased_frame(self) -> None:
        self.write_rgb(1, 1)
        first = self.ring.read_frame()
        self.write_rgb(2, 2)
        second = self.ring.read_frame()

        self.assertFalse(self.write_rgb(3, 3))
        self.assertEqual(bytes(first.data), bytes([1]) * 12)
        first.release()
        second.release()

    def test_nv12_frame_size_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame data size"):
            self.ring.write_frame(
                camera_id="camera-1",
                sequence_number=1,
                width=4,
                height=2,
                pixel_format=FramePixelFormat.NV12,
                captured_at_ns=1,
                data=bytes(8),
            )

    def test_nv12_requires_even_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions must be even"):
            self.ring.write_frame(
                camera_id="camera-1",
                sequence_number=1,
                width=3,
                height=2,
                pixel_format=FramePixelFormat.NV12,
                captured_at_ns=1,
                data=bytes(9),
            )

    def test_rejects_mismatched_attach_configuration(self) -> None:
        with self.assertRaises(ValueError):
            SharedFrameRingBuffer(
                capacity=3,
                max_frame_bytes=64,
                name=self.ring.name,
                create=False,
                lock=self.ring._lock,
            )

    def test_attached_reader_uses_shared_memory_and_lock(self) -> None:
        attached = SharedFrameRingBuffer(
            capacity=2,
            max_frame_bytes=64,
            name=self.ring.name,
            create=False,
            lock=self.ring._lock,
        )
        try:
            self.write_rgb(8, 4)
            with attached.read_frame() as lease:
                self.assertEqual(bytes(lease.data), bytes([4]) * 12)
        finally:
            attached.close()

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            self.ring.write_frame(
                camera_id="camera-1",
                sequence_number=1,
                width=0,
                height=2,
                pixel_format=FramePixelFormat.RGB24,
                captured_at_ns=1,
                data=b"",
            )


if __name__ == "__main__":
    unittest.main()