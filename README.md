# SmartVision AI

Real-time object detection and spatial perception in the browser, built with **Next.js, React, TypeScript, Tailwind CSS and Three.js**, backed by a custom-trained TensorFlow detector.

You point a webcam at a scene. The app detects 10 object classes with a YOLO-style model I trained myself, keeps stable track IDs across frames, estimates rough distance when a depth model is available, and lets you ask questions about the scene.

---

## What it does

- **10 detection classes**: `sunglasses`, `knife`, `water_bottle`, `pen`, `chair`, `human_face`, `mobile_phone`, `helmet`, `fire`, `can`.
- **Custom detector**: CSP-style backbone, SPPF and an FPN neck with three detection scales (about 7.9M parameters), trained on public Roboflow datasets. See [`model.py`](legacy_django/backend/ml_model/model.py) and the training notebook.
- **Object tracking**: a SORT tracker in the vision service (Kalman filter on `[u, v, s, r, u̇, v̇, ṡ]` with IoU matching) keeps track IDs stable between frames.
- **Approximate distance (optional)**: when PyTorch and the MiDaS weights are installed, each box gets a 3D position and distance from monocular depth. MiDaS depth is relative, so distances are mapped into a fixed 0.35–4.5 m range and are rough estimates. Without the depth model, detections are returned without 3D values; the app never guesses.
- **Scene relations and chat**: a scene graph of spatial relations (`left_of`, `above`, `wearing`, …) feeds a chat assistant. It uses Groq when `GROQ_API_KEY` is set and a rule-based fallback otherwise.
- **3D viewer**: a Three.js viewer that loads a GLB mesh per class. A mesh can also be reconstructed by the FastAPI service (TSDF + marching cubes). Point-cloud mode turns your last saved detection crop into a point cloud with the MiDaS depth model; before any crop is saved it shows a synthetic demo surface, labelled as such.
- **Dashboard and replay**: detection totals, class breakdown and recent activity, plus measured inference latency, round-trip latency and throughput from every processed frame. Replay shows a session's recent detections.
- **Alerts**: optional signed webhooks when watched classes (default `knife`, `fire`) are detected.
- **Accounts**: sign-up with optional email verification, password reset by email, roles, and an admin page for users, roles and the audit log. Login, sign-up and email endpoints are rate-limited per client.

---

## Architecture

```
Browser ── Next.js (port 3000) ──┬── vision service: Django + TensorFlow (port 8001)
            middleware.ts         │     detection, tracking, optional depth
            requires a session    │
                                  └── auth/persistence service: FastAPI (port 8000)
                                        users, sessions, telemetry, replay, mesh
                                        reconstruction ── PostgreSQL
```

- **Next.js** serves the UI and thin API routes. `middleware.ts` sends anonymous visitors to `/login` and returns 401 from data and inference APIs.
- **FastAPI** (`core/infrastructure/api`) is the single source of truth for users and sessions. Session tokens are 256-bit random values; only their SHA-256 hashes are stored.
- **Django** (`legacy_django/`) runs the trained model. It was the original full app; today the Next.js frontend uses it only for inference.
- The services authenticate each other with a shared `INTERNAL_API_TOKEN`. FastAPI refuses every non-health request without it, and the Django detect endpoint requires it or a signed-in Django user.

---

## Getting started

### Requirements

- Docker with Docker Compose, **or** Node.js 20+ and Python 3.12+
- Git LFS (the trained weights are stored with LFS): `git lfs install`

### Run everything with Docker

```bash
git clone https://github.com/Lamsalyusu/smart-vision-ai.git
cd smart-vision-ai
git lfs pull

cp .env.example .env
# Set at least these two, each to the output of: openssl rand -hex 32
#   INTERNAL_API_TOKEN=...
#   SECRET_KEY=...

docker compose up --build
```

Open http://localhost:3000, create an account and open the camera page. Compose refuses to start without `INTERNAL_API_TOKEN`.

New accounts get the `operator` role. Clearing the shared detection history needs `admin`. Promote an account with:

```bash
docker compose exec auth python -m core.infrastructure.manage set-role your-username admin
# outside Docker: python -m core.infrastructure.manage set-role your-username admin
```

`list-users` shows every account and its role. Role changes are recorded in the audit log. Admins also get an **Admin** button on the dashboard, which opens `/admin`: users, roles, email-verification status and the audit log.

### Run for development

```bash
npm install
pip install -r requirements-dev.txt                        # FastAPI service + test tools
pip install -r legacy_django/requirements-vision.txt       # vision service

export INTERNAL_API_TOKEN=$(openssl rand -hex 32)          # same value for all three processes

uvicorn core.infrastructure.api.main:app --port 8000       # uses SQLite if PostgreSQL is unreachable
(cd legacy_django && DJANGO_DEBUG=True python manage.py runserver 8001)
npm run dev
```

`next dev` defaults to `http://localhost:8000` for FastAPI (`AUTH_API_URL`) and `http://localhost:8001/api/detect/` for inference (`VISION_API_URL`). In production, set both explicitly.

### Optional features

| Feature | Enable with |
| --- | --- |
| Distance estimates | Included in the Docker vision image (CPU PyTorch; MiDaS weights are fetched at build time). Build with `--build-arg WITH_DEPTH=false` to leave it out. Outside Docker: `pip install torch torchvision timm`; the weights download from TorchHub on first use. |
| Password reset and verification emails | `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` (SMTP with STARTTLS; `EMAIL_HOST`/`EMAIL_PORT` default to Gmail) and `APP_BASE_URL`. Without them, the reset page reports that email is not configured and no verification emails are sent. |
| Require verified email to sign in | `REQUIRE_EMAIL_VERIFICATION=true`. Only enforced when email is configured, so it cannot lock everyone out. Admins can mark an address verified on `/admin`. |
| Detection alerts | `ALERT_WEBHOOK_URL` (must be a public address). `ALERT_CLASSES` (default `knife,fire`), `ALERT_MIN_CONFIDENCE` (default 0.5) and `ALERT_COOLDOWN_SECONDS` (default 60 per class) tune it. Payloads are signed with `WEBHOOK_SECRET` in `X-SmartVision-Signature` (HMAC-SHA256). |
| Live telemetry stream | WebSocket `ws://<auth-host>:8000/ws/telemetry?token=<INTERNAL_API_TOKEN>` receives one `{"type": "detections", ...}` event per processed frame. It is for server-side integrations; browsers should not hold the internal token. |
| Rate limits behind a proxy | Limits are per client address. Set `TRUST_PROXY_HEADERS=true` only when a reverse proxy you control sets `X-Forwarded-For`; otherwise all clients share one bucket. `RATE_LIMITS_ENABLED=false` turns limits off for load testing. Counters live in memory in the FastAPI process. |
| LLM chat answers | `GROQ_API_KEY` |
| RTSP cameras | `RTSP_ALLOWED_HOSTS`, then `POST /api/streams/rtsp/` on the vision service with the `X-Internal-Token` header |
| Supabase sync, OAuth, email | Only used by the Django UI; see `.env.example` |

Replay metadata is scoped to an opaque browser session, capped at 240 frames per session and kept for 30 days; images are not stored.

---

## Development commands

```bash
make test         # Python (pytest) and TypeScript test suites
make lint         # ruff and ESLint
make type-check   # mypy and tsc
npm run build     # production build of the Next.js app

# Vision service tests (runs the real model; needs the LFS weights)
cd legacy_django && DJANGO_DEBUG=True python manage.py test backend
```

CI (`.github/workflows/ci.yml`) runs the core Python checks, the vision service tests with the real weights, the Next.js checks and build, and a Docker Compose smoke test.

The test suites never touch live services: Django tests force `OFFLINE_MODE`, and the core tests use a temporary SQLite database and blank every email, alert, Groq and Supabase credential, even when `.env` contains real ones.

### Database migrations

The FastAPI service applies its Alembic migrations (`core/infrastructure/database/migrations`) when it starts. Databases created before migrations existed are upgraded in place; existing tables are kept. After changing a model, create a migration with:

```bash
python -m core.infrastructure.database.migrate revision "describe the change"
```

---

## Project structure

```
smart-vision-ai/
├── app/                  # Next.js pages and API routes
│   ├── camera/           # live camera feed with detection overlay
│   ├── dashboard/        # telemetry and class statistics
│   ├── model-viewer/     # Three.js GLB / mesh viewer
│   ├── gpu-viewer/       # WebGPU point-cloud renderer demo
│   ├── replay/           # per-session detection replay
│   ├── admin/            # users, roles and audit log (admins only)
│   ├── verify-email/     # email verification and resend
│   └── api/              # auth, detect, chat, replay, crops, mesh
├── middleware.ts         # session check for pages and APIs
├── lib/                  # TypeScript: API client, session helpers, scene graph, WebGPU renderer
├── core/                 # Python: the FastAPI auth/persistence service
│   ├── domain/           #   mesh reconstruction (TSDF + marching cubes)
│   ├── infrastructure/   #   FastAPI app, SQLAlchemy models, JWT/RBAC, webhooks
│   └── tests/
├── legacy_django/        # vision service, trained weights (LFS) and training notebook
├── rust/geometry/        # optional WASM point-cloud kernels for the GPU viewer
├── sdk/                  # small Python and TypeScript clients for the FastAPI service
├── public/models/        # one GLB mesh per class
└── tests/                # TypeScript tests
```

### Optional WASM geometry kernels

```bash
rustup target add wasm32-unknown-unknown
cd rust/geometry && cargo build --release --target wasm32-unknown-unknown
mkdir -p ../../public/wasm
cp target/wasm32-unknown-unknown/release/smartvision_geometry.wasm ../../public/wasm/geometry.wasm
```

Without `public/wasm/geometry.wasm` the GPU viewer uses the JavaScript implementation.

---

## Model evaluation

Per-class precision, recall and F1 on the test set (confidence threshold 0.25) are in [`legacy_django/model_evaluation/metrics_summary.txt`](legacy_django/model_evaluation/metrics_summary.txt) (mean F1 0.56). The model is strongest on `can`, `helmet` and `chair` and weakest on `fire` and small objects.

### Inference speed

Measured with [`benchmark_inference.py`](legacy_django/model_evaluation/benchmark_inference.py) on a laptop CPU (Intel Core i5-1335U, 12 threads, no GPU, TensorFlow 2.21), 50 frames of 640×480 after 5 warm-up frames:

| Stage | Median | 95th percentile | Frames/s |
| --- | --- | --- | --- |
| Detector only | 281 ms | 292 ms | 3.6 |
| Detection + tracking | 301 ms | 310 ms | 3.4 |

The camera page sends one frame per second, so a CPU like this keeps up. Raw results are in [`benchmark_results.json`](legacy_django/model_evaluation/benchmark_results.json).

Inside the Docker vision container on the same laptop (Docker Desktop VM with 1.8 GB of memory, 30 frames), detection + tracking took 392 ms median / 413 ms p95, and the MiDaS depth model 90 ms median but 1.66 s p95. Depth times vary widely in that memory-limited VM; give Docker more memory for steadier results. Run it on your own hardware with:

```bash
cd legacy_django && DJANGO_DEBUG=True python model_evaluation/benchmark_inference.py --frames 50
```

While the app runs, the dashboard shows the same measurements for real traffic.

## Model weights and dataset licence

The weights (`best_weights.weights.h5`, 32 MB, stored with Git LFS) were trained by the author on public Roboflow Universe datasets licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Attribution and per-dataset sources are in [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md) and [`legacy_django/backend/ml_model/weights/DATASET_LICENSE.md`](legacy_django/backend/ml_model/weights/DATASET_LICENSE.md).

The weights are released under the project's MIT License; the training data remains CC BY 4.0 and is not covered by the MIT licence.

## Model training and background

How the dataset was collected, cleaned and trained on Google Colab is described in [`ProjectJourney.md`](ProjectJourney.md) and in the notebook [`legacy_django/YuyutsuLamsal_ObjectDetectionModel.ipynb`](legacy_django/YuyutsuLamsal_ObjectDetectionModel.ipynb).

---

## License

MIT; see [LICENSE](LICENSE). Third-party components are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
