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
- **3D viewer**: a Three.js viewer that loads a GLB mesh per class. A mesh can also be reconstructed by the FastAPI service (TSDF + marching cubes). The point-cloud mode shows a synthetic demo surface and is labelled as such.
- **Dashboard and replay**: detection totals, class breakdown and recent activity from the database, plus a per-session replay of recent detections.

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

`list-users` shows every account and its role. Role changes are recorded in the audit log.

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
| Password reset emails | `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD` (SMTP with STARTTLS; `EMAIL_HOST`/`EMAIL_PORT` default to Gmail) and `APP_BASE_URL`. Without them, the reset page reports that email is not configured. |
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

The test suites never touch live services: Django tests force `OFFLINE_MODE`, and the core tests use a temporary SQLite database even when `.env` contains real credentials.

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

Per-class precision, recall and F1 on the validation split are in [`legacy_django/model_evaluation/metrics_summary.txt`](legacy_django/model_evaluation/metrics_summary.txt) (mean F1 0.56). The model is strongest on `can`, `helmet` and `chair` and weakest on `fire` and small objects. No end-to-end latency or throughput benchmark is published; the dashboard shows those values as "Not measured".

## Model weights and dataset licence

The weights (`best_weights.weights.h5`, 32 MB, stored with Git LFS) were trained by the author on public Roboflow Universe datasets licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. Attribution and per-dataset sources are in [`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md) and [`legacy_django/backend/ml_model/weights/DATASET_LICENSE.md`](legacy_django/backend/ml_model/weights/DATASET_LICENSE.md).

The weights are released under the project's MIT License; the training data remains CC BY 4.0 and is not covered by the MIT licence.

## Model training and background

How the dataset was collected, cleaned and trained on Google Colab is described in [`ProjectJourney.md`](ProjectJourney.md) and in the notebook [`legacy_django/YuyutsuLamsal_ObjectDetectionModel.ipynb`](legacy_django/YuyutsuLamsal_ObjectDetectionModel.ipynb).

---

## License

MIT; see [LICENSE](LICENSE). Third-party components are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
