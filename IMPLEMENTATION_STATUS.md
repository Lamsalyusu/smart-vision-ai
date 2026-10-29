# Implementation Status

This document tracks the requested six-phase roadmap. Counts such as LOC and
commit totals are reported as measurements, not used as substitutes for tested
capabilities.

## Phase 1: Contracts and Core Architecture

- **Implemented:** Versioned protobuf schemas for frames, detections, 3D tracks,
  scene graphs, and camera telemetry.
- **Implemented:** Reproducible Python stub generation with `make proto-python`,
  plus a tested domain-to-protobuf detection-batch codec.
- **Implemented:** Initial perception, tracking, and spatial domain packages;
  immutable validated entities, 3D state, Mahalanobis cost, quaternions, and
  frame-aware transforms.
- **Partial:** A perception application mapper bridges the legacy detector's
  dictionaries to the domain model.
- **Remaining:** TypeScript, Go, and Rust generated clients and CI checks. This
  checkout currently has no Node/npm, Go, or Rust toolchains.
- **Remaining:** Full migration of `lib/` and `legacy_django/backend/` behind
  application ports; production Redis, Triton, Kafka, and WebRTC adapters.

## Phase 2: Distributed Video Ingestion

- **Implemented:** Per-camera token-bucket admission with keyframe policy and
  tracking-preservation signal when a frame is skipped.
- **Implemented:** POSIX shared-memory NV12/RGB ring with bounded slots, leases,
  process attachment using a shared lock, and overwrite/drop policy.
- **Implemented:** Camera stream metadata registry, AIMD bitrate policy, and an
  application ingress coordinator connecting registration, admission, and the
  shared-memory ring.
- **Remaining:** WebRTC SFU, RTSP bridge, VP8/H.264 packetization, and measured
  50-camera adaptive-bitrate media integration. The current registry and
  bitrate controller do not negotiate or transmit media.
- **Remaining:** GPU/CUDA IPC path and live connection between media ingress,
  ring buffers, inference workers, and tracker state.

## Phase 3: Spatial Perception and Tracking

- **Implemented:** Eight-state UKF primitive, normalized homography and
  fundamental-matrix estimation, Zhang intrinsic recovery, and RANSAC
  ground-plane fitting.
- **Implemented:** Temporal graph relations for approaching, occluding, near,
  and hazard-zone membership with bounded timestamped history.
- **Implemented:** Legacy 2D tracker's greedy assignment replaced by SciPy's
  Jonker-Volgenant solver.
- **Remaining:** Integrate the UKF with detections/depth and multi-camera
  identity association; add appearance embeddings and calibrated extrinsics to
  live tracking.
- **Remaining:** MiDaS/Depth-Anything inference adapter, metric frustum
  back-projection, and surface-normal estimation.
- **Remaining:** Cypher-compatible query language and a measured 60 Hz graph
  update path.

## Phase 4: Reliability and MLOps

- **Implemented:** Priority/model-isolated batching policy, circuit breaker,
  KS/Wasserstein drift monitor, bounded annotation queue, and transport-neutral
  Raft election/log state machine.
- **Implemented:** In-process concurrent ingress harness for up to 100 camera
  producers, reporting submission throughput and P50/P99 latency. It does not
  measure network-to-inference latency.
- **Remaining:** Triton client/server deployment, TensorRT compilation,
  multi-GPU scheduling, and inference fallback adapters.
- **Remaining:** Durable Raft storage, network transport, membership changes,
  and multi-process fault/recovery tests.
- **Remaining:** Chaos worker and integrated frame-drop/network/GPU-failure
  recovery tests.
- **Remaining:** Persistent annotation workflow and camera-condition diagnosis;
  current drift logic reports distribution change only.

## Phase 5: Spatial Web Client

- **Implemented:** Browser-local replay of up to 240 camera detection snapshots,
  with timeline controls, trajectory canvas, and per-frame detection inspection.
- **Remaining:** The replay view is not backed by a server history API and stores
  no video frames.
- **Remaining:** WebGPU compute/render pipelines, million-point-cloud
  performance measurements, and fallback rendering.
- **Remaining:** Rust/WASM geometry module, build pipeline, and client-side
  correctness/performance tests. Rust tooling is unavailable in this checkout.

## Phase 6: Verification and Simulation

- **Implemented:** Seeded synthetic box-sequence generator with noise, misses,
  occlusions, and false positives; MOTA/MOTP/IDF1 evaluation.
- **Implemented:** 117 core unit tests and Ruff checks for `core/` at the last
  validated checkpoint. A strict MyPy CI gate covers the dependency-light
  perception and tracking primitives.
- **Implemented:** GitHub Actions runs protobuf generation, core Ruff, the
  narrow strict MyPy gate, and core tests.
- **Remaining:** Synthetic image rendering and tracker benchmark integration.
- **Remaining:** Mutation testing, repository-wide strict MyPy/Sonar gates, and
  zero-warning checks. Strict MyPy was assessed and reported 106 issues across
  generated protobufs, NumPy/SciPy code, tests, and legacy code, so CI does not
  claim to gate those areas yet.
- **Remaining:** Distributed 100-stream WebRTC stress harness and measured
  end-to-end P99 latency.

## Environment and Repository Notes

- `origin/main` was at 173 commits when this status was updated; the requested
  200 commits have not been fabricated.
- The meaningful codebase remains far below 100,000 lines; no generated or
  duplicated filler is counted toward that target.
- Python protobuf stubs are generated locally and ignored by Git. The root
  `.venv` points at an externally managed interpreter and cannot be modified;
  local test runs have used an isolated temporary environment.
- An existing untracked `backendcore/backend/urls.py` is user-owned and has
  intentionally not been modified or staged.