#!/usr/bin/env python3
"""Operational synthetic camera load and chaos measurement.

Examples:
    python scripts/load_test_synthetic.py --cameras 64 --frames 300
    python scripts/load_test_synthetic.py --cameras 64 --frames 120 \
        --drop-every 7 --disconnect-every 23 --error-every 31
    python scripts/load_test_synthetic.py --cameras 8 --frames 30 \
        --detector-url http://127.0.0.1:8001/api/detect/
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.application.simulation import (
    CameraStreamPlan,
    ChaosPolicy,
    StreamingLoadHarness,
    SyntheticCameraFarm,
)


def build_processor(detector_url: str | None):
    if detector_url:

        def post_frame(frame) -> None:
            request = urllib.request.Request(
                detector_url,
                data=frame.jpeg_bytes,
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Tracking-Session": f"synthetic-{frame.camera_id}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10.0) as response:
                if response.status >= 400:
                    raise RuntimeError(f"detector returned {response.status}")

        return post_frame

    def decode_frame(frame) -> None:
        decoded = cv2.imdecode(
            np.frombuffer(frame.jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise ValueError("synthetic frame could not be decoded")

    return decode_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=int, default=64)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--detector-url", default=None)
    parser.add_argument("--drop-every", type=int, default=0)
    parser.add_argument("--disconnect-every", type=int, default=0)
    parser.add_argument("--error-every", type=int, default=0)
    parser.add_argument("--latency-spike-every", type=int, default=0)
    parser.add_argument("--latency-spike-seconds", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cameras < 1:
        print("--cameras must be positive", file=sys.stderr)
        return 2

    camera_ids = [f"cam-{index:03d}" for index in range(args.cameras)]
    farm = SyntheticCameraFarm(
        camera_ids, width=args.width, height=args.height
    )
    plans = [
        CameraStreamPlan(camera_id, frames=args.frames, target_fps=args.fps)
        for camera_id in camera_ids
    ]
    chaos = ChaosPolicy(
        drop_every=args.drop_every,
        disconnect_every=args.disconnect_every,
        error_every=args.error_every,
        latency_spike_every=args.latency_spike_every,
        latency_spike_seconds=args.latency_spike_seconds,
    )

    report = StreamingLoadHarness(
        build_processor(args.detector_url),
        farm=farm,
        realtime=args.realtime,
    ).run(plans, chaos=chaos)

    payload = report.to_dict()
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
