# SmartVision AI

Real-time spatial perception and object detection web application built with **Next.js, React, TypeScript, Tailwind CSS, and Three.js**.

This project connects computer vision object detection across 10 custom classes with real-time multi-object tracking, 2D-to-3D spatial estimation using camera intrinsics, and interactive 3D model visualization. It also provides a comprehensive Python core for computer vision algorithms, spatial geometry, robotics, SLAM, and analytics.

---

## What it does

The application provides an interactive camera feed and dashboard for detecting and tracking objects in real time:

- **10 Detection Classes**: Detects `sunglasses`, `knife`, `water_bottle`, `pen`, `chair`, `human_face`, `mobile_phone`, `helmet`, `fire`, and `can`.
- **Object Tracking**: Uses a Kalman filter box tracker (SORT approach) to keep stable track IDs across frames and estimate velocities.
- **2D-to-3D Spatial Estimation**: Uses pinhole camera geometry to calculate estimated $(X, Y, Z)$ coordinates and distance from the camera based on bounding box scales and depth.
- **Spatial Scene Relations**: Identifies basic physical relationships between objects (e.g., whether someone is holding a phone, wearing sunglasses, or if an item is sitting near another).
- **Interactive 3D Visualizer**: A Three.js viewer with orbit controls that lets you view 3D point clouds and load 3D GLB meshes corresponding to detected classes.
- **Analytics & History**: Dashboard with detection history, class breakdown, and recent activity logs.

---

## Tech Stack

- **Multi-Class Detection Engine**: High-throughput multi-scale detection across 10 object classes (`sunglasses`, `knife`, `water_bottle`, `pen`, `chair`, `human_face`, `mobile_phone`, `helmet`, `fire`, `can`).
- **SORT Multi-Object Tracking**: Continuous 7-state Kalman filter box tracking (`[u, v, s, r, u_dot, v_dot, s_dot]`) with velocity estimation and bipartite IoU matching (`lib/tracker.ts`).
- **RGB-D Metric 3D Fusion**: Pinhole camera intrinsic back-projection fusing 2D boxes with monocular depth maps to compute metric 3D positions $(X, Y, Z)$, Euclidean distances, and ground elevations (`lib/spatial-fusion.ts`).
- **Spatial Scene Graph**: Semantic relation graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ computing spatial interactions (`adjacent_to`, `interacting_with`, `wearing`, `supporting`, `left_of`, `above`) for contextual AI reasoning (`lib/scene-graph.ts`).
- **Multimodal AI Scene Assistant**: Real-time natural language reasoning over spatial scene graphs and perceived objects via `/api/chat`.
- **Interactive 3D Visualizer**: Dual-mode Three.js WebGL visualizer with real-time OrbitControls, Point Cloud generation with depth/size sliders, and GLB mesh loader (`app/model-viewer/page.tsx`).
- **Telemetry Console**: Database-backed detection totals, class distributions, active sessions, and recent activity (`app/dashboard/page.tsx`).

**Frontend**: Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Three.js & OrbitControls, Lucide React
**Backend / Core**: Python, PyTorch, TensorFlow, NumPy, SciPy, OpenCV
**3D Visualization**: Three.js, WebGPU (experimental)
**Infrastructure**: Docker, Docker Compose, FastAPI, PostgreSQL (via Django ORM)

---

## Getting Started

### Requirements
- Node.js (v18 or v20+)
- npm or yarn
- Python 3.11+ (for core algorithms)
- Docker (optional, for inference service)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Lamsalyusu/smart-vision-ai.git
   cd smart-vision-ai
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

Camera inference runs in the separate vision service (`legacy_django/`, port 8001), which loads the author-trained YOLO weights from `legacy_django/backend/ml_model/weights/best_weights.weights.h5`. For local Next.js development, start that service with Docker Compose:

```bash
docker compose up --build vision
```

`next dev` defaults to `http://localhost:8001/api/detect/`; the Compose web service uses `http://vision:8001/api/detect/`. Configure `VISION_API_URL` to a reachable inference service URL in production. Detection requests return `503` when no inference service is configured rather than displaying generated detections.

### Model weights & dataset licence

The trained detection weights (`best_weights.weights.h5`) were trained by the author on public Roboflow Universe datasets licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Full attribution (creator credit, licence URL, modification statement, per-project source URLs): see [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md) and [`legacy_django/backend/ml_model/weights/DATASET_LICENSE.md`](legacy_django/backend/ml_model/weights/DATASET_LICENSE.md).

The weights file itself is licensed by the project author under the MIT License (see [`LICENSE`](LICENSE)); the training data remains CC BY 4.0 and is **not** covered by the project MIT licence.

Generate a private internal service token and place the same value in `INTERNAL_API_TOKEN` for the web and auth services. Replay metadata is scoped to the opaque browser session, capped at 240 frames per session, and retained for 30 days; video images are not stored.

```bash
openssl rand -hex 32
```

### RTSP camera ingestion

The vision service can ingest RTSP/RTSPS streams whose endpoint appears in `RTSP_ALLOWED_HOSTS` (comma-separated `host` or `host:port` values). Manage streams through the token-protected endpoint, sending the same `INTERNAL_API_TOKEN` value in the `X-Internal-Token` header:

- `GET /api/streams/rtsp/` returns connection and frame-count metrics for each stream.
- `POST /api/streams/rtsp/` with `{"camera_id": "...", "stream_url": "rtsp://..."}` starts ingestion and returns `202`.
- `DELETE /api/streams/rtsp/?camera_id=...` stops and removes a stream.

`RTSP_MAX_STREAMS` caps concurrent streams, and `RTSP_RECONNECT_DELAY_SECONDS` controls the wait before reconnecting a dropped camera.

---

## Project Structure

```
smart-vision-ai/
├── app/                  # Next.js App Router pages and API routes
│   ├── camera/           # Live camera feed with detection overlay & HUD
│   ├── dashboard/        # Telemetry, analytics, and class statistics
│   ├── model-viewer/     # Three.js 3D point cloud & GLB mesh viewer
│   ├── gpu-viewer/       # WebGPU million-point viewer with measured FPS
│   ├── replay/           # Session-scoped SQL-backed spatial trajectory replay
│   └── api/              # API endpoints for detection, chat & 3D models
├── lib/                  # Core algorithms in TypeScript
│   ├── detector.ts       # Object detection logic & class thresholds
│   ├── tracker.ts        # Kalman box tracking (SORT)
│   ├── spatial-fusion.ts # Camera intrinsics & 3D back-projection
│   ├── scene-graph.ts    # Spatial relation builder
│   ├── webgpu-pointcloud.ts # WebGPU point-cloud renderer
│   ├── geometry-accel.ts # JS geometry with optional WASM acceleration
│   └── store.ts          # Client state store
├── core/                 # Computer vision & spatial geometry algorithms in Python
│   ├── application/      # Application layer (ingestion, perception, simulation, etc.)
│   ├── domain/           # Domain models (tracking, vision, SLAM, robotics, etc.)
│   ├── infrastructure/   # Infrastructure (database, API, security, concurrency, etc.)
│   └── tests/            # Unit tests
├── rust/geometry/        # Optional Rust/WASM point-cloud geometry kernels
├── scripts/              # Operational load and chaos measurement commands
├── public/models/        # 3D GLB assets for detected object classes
├── configs/              # Sensor profiles and dataset configurations
├── docs/                 # Documentation and event schemas
├── legacy_django/        # Original Django backend and model training notebook
├── sdk/                  # Python and TypeScript SDK clients
└── tests/                # Frontend tests
```

---

## Development Scripts

```bash
npm run dev      # Start Next.js development server
npm run build    # Compile optimized production bundle (23/23 routes)
npm run start    # Start production server
npm run lint     # Run ESLint validation
npx tsc --noEmit # Type-check TypeScript codebase
```

Operational and quality commands:

```bash
make load-test     # 64 synthetic camera streams, throughput + p50/p95/p99
make chaos-test    # Same load with drops, disconnects, and pipeline errors
make mutation-test # Cosmic Ray mutation testing, writes reports/mutation.json
```

To build the optional WASM geometry kernels:

```bash
rustup target add wasm32-unknown-unknown
cd rust/geometry && cargo build --release --target wasm32-unknown-unknown
mkdir -p ../../public/wasm
cp target/wasm32-unknown-unknown/release/smartvision_geometry.wasm ../../public/wasm/geometry.wasm
```

When `/wasm/geometry.wasm` is absent the browser uses the JavaScript geometry
implementation automatically, so WebGPU discovery does not depend on Rust.

---

## Performance Evidence

No production inference benchmark is published yet: accuracy and end-to-end
camera-to-model latency require representative cameras, GPU hardware, and a
running Triton/TensorRT deployment. The dashboard labels those values as
unavailable until they are measured rather than displaying fabricated numbers.

The synthetic load harness has been measured locally on this workstation's CPU
for **64 concurrent camera streams x 120 frames** (7,680 frames of deterministic
320x240 JPEG encode + decode, no model inference):

| Scenario | Throughput | p50 | p95 | p99 | Max | Faults |
| --- | --- | --- | --- | --- | --- | --- |
| Clean streams | 4,461 fps | 3.35 ms | 12.16 ms | 21.39 ms | 53.83 ms | 0 |
| Chaos injection | 4,011 fps | 3.46 ms | 12.57 ms | 19.81 ms | 48.71 ms | 1,600 |

Chaos injection configured drops every 7 frames, disconnects every 23, errors
every 31, and a 5 ms latency spike every 47. Results include machine and
JIT/decoder warm-up variance; reproduce with `make load-test` and
`make chaos-test`. These figures measure the synthetic frame pipeline only, not
detector accuracy or production inference throughput.

Mutation testing is wired through `make mutation-test`, which initializes a
Cosmic Ray session, executes it with the local distributor, and writes
`reports/mutation.json` with killed/survived counts and the mutation score. A
smoke session over `core/application/ingestion/frame_admission.py` produced
**111 killed / 27 survived out of 138 mutations (80.43%)**. The surviving
mutations cluster around idle-timeout eviction and rate-limit boundary
assertions; they are reported rather than hidden, and the full `core` target is
available for a complete score.

The TensorFlow detector has a CPU smoke test through its HTTP endpoint. The
Triton client, TensorRT artifact checks, and multi-GPU scheduler are unit tested
with injected transports; no GPU cluster was available in this environment, so
multi-GPU/Triton throughput is not claimed.

---

## Model Training & Background

For details about how the dataset was collected, cleaned, and trained on Google Colab, see [`ProjectJourney.md`](ProjectJourney.md) and the training notebook in `legacy_django/144_2432214_YuyutsuLamsal_ObjectDetectionModel.ipynb`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# CI test
