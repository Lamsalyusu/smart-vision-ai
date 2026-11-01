# SmartVision AI

Real-time spatial perception and object detection web application built with **Next.js (App Router), TypeScript, Tailwind CSS, React, and Three.js**.

The system performs image-based object detection across 10 classes, temporal multi-object tracking via SORT Kalman filtering, 2D-to-3D metric spatial fusion when depth is available, and multimodal scene reasoning.

---

## Technology Stack

- **Framework**: [Next.js](https://nextjs.org/) (App Router, Version 15)
- **Language**: [TypeScript](https://www.typescriptlang.org/) (Strict mode, 100% type safety)
- **UI Library**: [React 19](https://react.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) with Cyber-Dark Glassmorphic design system
- **3D Graphics**: [Three.js](https://threejs.org/) WebGL with OrbitControls (Point Cloud & GLB Meshes)
- **Icons**: [Lucide React](https://lucide.dev/)

---

## Key Features

- **Multi-Class Detection Engine**: High-throughput multi-scale detection across 10 object classes (`sunglasses`, `knife`, `water_bottle`, `pen`, `chair`, `human_face`, `mobile_phone`, `helmet`, `fire`, `can`).
- **SORT Multi-Object Tracking**: Continuous 7-state Kalman filter box tracking (`[u, v, s, r, u_dot, v_dot, s_dot]`) with velocity estimation and bipartite IoU matching (`lib/tracker.ts`).
- **RGB-D Metric 3D Fusion**: Pinhole camera intrinsic back-projection fusing 2D boxes with monocular depth maps to compute metric 3D positions $(X, Y, Z)$, Euclidean distances, and ground elevations (`lib/spatial-fusion.ts`).
- **Spatial Scene Graph**: Semantic relation graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ computing spatial interactions (`adjacent_to`, `interacting_with`, `wearing`, `supporting`, `left_of`, `above`) for contextual AI reasoning (`lib/scene-graph.ts`).
- **Multimodal AI Scene Assistant**: Real-time natural language reasoning over spatial scene graphs and perceived objects via `/api/chat`.
- **Interactive 3D Visualizer**: Dual-mode Three.js WebGL visualizer with real-time OrbitControls, Point Cloud generation with depth/size sliders, and GLB mesh loader (`app/model-viewer/page.tsx`).
- **Telemetry Console**: Database-backed detection totals, class distributions, active sessions, and recent activity (`app/dashboard/page.tsx`). Unmeasured latency, tracking, and depth-accuracy values are shown as unavailable.

---

## Quickstart

### Prerequisites

- **Node.js**: v20.9.0 or higher (v22.x recommended)
- **npm**: v10.x or higher

### Installation & Local Run

```bash
# 1. Clone repository
git clone https://github.com/Lamsalyusu/smart-vision-ai.git
cd smart-vision-ai

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
```

Open `http://localhost:3000` (or `http://localhost:3001` if port 3000 is occupied) in your browser.

Camera inference uses the CPU TensorFlow service and checked-in trained weights. For local Next.js development, start that service with Docker Compose:

```bash
docker compose up --build vision
```

`next dev` defaults to `http://localhost:8001/api/detect/`; the Compose web service uses `http://vision:8001/api/detect/`. Configure `VISION_API_URL` to a reachable inference service URL in production. Detection requests return `503` when no inference service is configured rather than displaying generated detections.

---

## Developer Commands

```bash
npm run dev      # Start Next.js development server
npm run build    # Compile optimized production bundle (23/23 routes)
npm run start    # Start production server
npm run lint     # Run ESLint validation
npx tsc --noEmit # Type-check TypeScript codebase
```

---

## Application Routes & APIs

### Pages (React & Tailwind CSS)
| Route | Description |
| :--- | :--- |
| `/` | Modern landing page with feature architecture and benchmarks |
| `/camera` | Real-time camera studio with canvas bounding boxes, tracking HUD, and AI chat |
| `/dashboard` | Telemetry console with detection KPIs and class distribution |
| `/model-viewer` | Interactive Three.js 3D point cloud and mesh inspection |
| `/login` | Authentication sign-in with quick demo credentials autofill |
| `/signup` | User account registration |
| `/forgot-password` | Password recovery request |

### API Handlers (Next.js Route Handlers)
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/detect` | `POST` | Detection, SORT Kalman tracking, and 3D metric fusion |
| `/api/chat` | `POST` | Spatial scene graph reasoning & multimodal chat |
| `/api/generate-3d` | `POST` | Synthesizes 3D monocular depth point cloud arrays |
| `/api/generate-3d-mesh` | `POST` | Generates GLB 3D mesh model metadata |
| `/api/3d-model/[className]` | `GET` | Retrieves GLB 3D model asset |
| `/api/user/stats` | `GET` | User detection telemetry and class distributions |
| `/api/save-crop` | `POST` | Stores cropped detection targets |
| `/api/get-crop` | `GET` | Retrieves latest cropped detection targets |
| `/api/clear-history` | `POST` | Clears telemetry logs and crop cache |
| `/api/auth/login` | `POST` | Authenticates user session |
| `/api/auth/signup` | `POST` | Creates user account |
| `/api/auth/logout` | `POST` | Invalidates session cookie |
| `/api/auth/session` | `GET` | Validates active session token |

---

## Performance Evidence

No production benchmark figures are currently published. The dashboard reports measured detection counts and labels performance or accuracy metrics as unavailable until they are collected from representative workloads. The TensorFlow detector has a CPU smoke test; it is not a throughput or accuracy benchmark.

---

## License

- **Project License**: MIT License (see [LICENSE](LICENSE))
