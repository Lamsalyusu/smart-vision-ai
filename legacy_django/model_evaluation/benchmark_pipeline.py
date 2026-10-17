"""
Pipeline performance and latency profiling benchmark.
Measures latency percentiles (P50, P95, P99) and FPS for:
1. Vectorized YOLO Post-Processing
2. SORT Kalman Multi-Object Tracking
3. 2D-to-3D Metric Spatial Fusion
"""

import os
import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ml_model.tracker import SortTracker
from backend.ml_model.spatial_fusion import Spatial3DFusion


def benchmark_postprocessing(num_iterations=200):
    grid_sizes = (80, 40, 20)
    anchors_list = (
        np.array([[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]], dtype=np.float32),
        np.array([[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]], dtype=np.float32),
        np.array([[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]], dtype=np.float32),
    )

    latencies = []

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -60.0, 60.0)))

    for _ in range(num_iterations):
        t0 = time.perf_counter()

        for scale_idx, grid_size in enumerate(grid_sizes):
            anchors = anchors_list[scale_idx]
            pred = np.random.randn(grid_size, grid_size, 3, 15).astype(np.float32)

            obj = sigmoid(pred[..., 4])
            col_grid, row_grid = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
            col_grid = np.expand_dims(col_grid, -1)
            row_grid = np.expand_dims(row_grid, -1)

            bx = (col_grid + sigmoid(pred[..., 0])) / grid_size
            by = (row_grid + sigmoid(pred[..., 1])) / grid_size

            anchor_w = anchors[:, 0].reshape(1, 1, 3)
            anchor_h = anchors[:, 1].reshape(1, 1, 3)

            bw = anchor_w * np.exp(np.clip(pred[..., 2], -5.0, 5.0))
            bh = anchor_h * np.exp(np.clip(pred[..., 3], -5.0, 5.0))

            c_probs = sigmoid(pred[..., 5:])
            best_score = np.max(c_probs, axis=-1)
            conf = obj * best_score

            mask = (obj >= 0.10) & (conf >= 0.25) & (bw >= 0.01) & (bh >= 0.01)
            _ = np.stack([bx[mask], by[mask]], axis=-1)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def benchmark_tracking(num_frames=500):
    tracker = SortTracker(max_age=15, min_hits=2, iou_threshold=0.3)
    latencies = []

    for f in range(num_frames):
        # Generate 4 moving objects with slight Gaussian jitter
        center_x = 100 + (f * 1.5) % 400
        center_y = 150 + (f * 0.8) % 300
        dets = [
            {
                "bbox": [center_x, center_y, center_x + 80, center_y + 120],
                "class": "water_bottle",
                "class_id": 2,
                "confidence": 0.88 + np.random.uniform(-0.05, 0.05),
            },
            {
                "bbox": [300 + np.random.uniform(-3, 3), 200, 380, 320],
                "class": "chair",
                "class_id": 4,
                "confidence": 0.92,
            },
        ]

        t0 = time.perf_counter()
        _ = tracker.update(dets)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def benchmark_spatial_fusion(num_iterations=200):
    fuser = Spatial3DFusion()
    depth_map = np.random.uniform(0.2, 0.9, (384, 384)).astype(np.float32)
    dets = [
        {"bbox": [100.0, 100.0, 250.0, 300.0], "class": "water_bottle", "confidence": 0.9},
        {"bbox": [320.0, 150.0, 480.0, 380.0], "class": "chair", "confidence": 0.85},
    ]

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = fuser.estimate_spatial_metrics(dets, depth_map, (480, 640))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def print_stats(name, latencies):
    arr = np.array(latencies)
    mean = np.mean(arr)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)
    fps = 1000.0 / mean if mean > 0 else 0

    print(f"\n--- {name} ---")
    print(f"Mean Latency: {mean:.2f} ms")
    print(f"P50 Latency:  {p50:.2f} ms")
    print(f"P95 Latency:  {p95:.2f} ms")
    print(f"P99 Latency:  {p99:.2f} ms")
    print(f"Throughput:   {fps:.1f} FPS")


if __name__ == "__main__":
    print("Running SmartVision Engineering Benchmarks...")
    vec_latencies = benchmark_postprocessing()
    print_stats("Vectorized YOLO Post-Processing (3 Scales)", vec_latencies)

    track_latencies = benchmark_tracking()
    print_stats("SORT Multi-Object Tracker (Kalman + IoU)", track_latencies)

    spatial_latencies = benchmark_spatial_fusion()
    print_stats("2D-to-3D Metric Spatial Fusion", spatial_latencies)
