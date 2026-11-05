"""Measure real inference latency of the vision pipeline on this machine.

Runs the trained detector (and the SORT tracker) in-process on camera-sized
frames and reports latency percentiles and throughput. Depth (MiDaS) is
measured too, but only when its weights are already cached locally; this
script never downloads model code.

Frame content barely affects a CNN's runtime, so synthetic frames give
representative timings. Pass --image to time a real photo instead.

    cd legacy_django
    DJANGO_DEBUG=True python model_evaluation/benchmark_inference.py --frames 50

Results are printed and written to model_evaluation/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendcore.settings")
os.environ.setdefault("OFFLINE_MODE", "true")

import cv2  # noqa: E402
import django  # noqa: E402
import numpy as np  # noqa: E402

django.setup()


def make_frame(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (31, 31), 0)  # camera-like smooth content, not pure noise
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return encoded.tobytes()


def summarise(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95 = float(np.percentile(ordered, 95))
    return {
        "frames": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": round(float(np.percentile(ordered, 50)), 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(ordered[-1], 1),
        "fps_at_mean": round(1000 / statistics.fmean(ordered), 2),
    }


def time_calls(fn, frames: list[bytes], warmup: int) -> list[float]:
    for frame in frames[:warmup]:
        fn(frame)
    samples = []
    for frame in frames[warmup:]:
        started = time.perf_counter()
        fn(frame)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def depth_cached() -> bool:
    try:
        import torch
    except ImportError:
        return False
    hub = Path(torch.hub.get_dir())
    return any(hub.glob("intel-isl_MiDaS*")) and any(hub.glob("checkpoints/midas_v21_small*"))


def machine() -> dict[str, object]:
    import tensorflow as tf

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "gpu_used": bool(tf.config.list_physical_devices("GPU")),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "os": f"{platform.system()} {platform.release()}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=int, default=50, help="measured frames per scenario")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--image", type=Path, help="time this image instead of synthetic frames")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark_results.json"))
    args = parser.parse_args()

    from backend.ml_model.inference import detector
    from backend.services.vision_service import VisionPipelineService

    total = args.frames + args.warmup
    if args.image:
        data = args.image.read_bytes()
        frames = [data] * total
        source = str(args.image)
    else:
        frames = [make_frame(args.width, args.height, seed) for seed in range(total)]
        source = f"synthetic {args.width}x{args.height} JPEG"

    results: dict[str, object] = {"input": source, "machine": machine(), "scenarios": {}}
    scenarios = results["scenarios"]
    assert isinstance(scenarios, dict)

    scenarios["detector_only"] = summarise(time_calls(detector.detect_from_bytes, frames, args.warmup))

    pipeline = VisionPipelineService()
    scenarios["detect_and_track"] = summarise(time_calls(
        lambda f: pipeline.process_frame(f, session_key="benchmark"), frames, args.warmup
    ))

    if depth_cached():
        # Depth only runs when there are detections, so time the depth model directly.
        from backend.ml_model.depth_estimation import depth_estimator

        scenarios["depth_midas_small"] = summarise(time_calls(depth_estimator.estimate_depth, frames, args.warmup))
    else:
        scenarios["depth_midas_small"] = "not measured: MiDaS weights are not cached on this machine"

    print(json.dumps(results, indent=2))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
