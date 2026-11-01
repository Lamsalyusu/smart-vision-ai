# SmartVision AI

Real-time spatial perception and object detection web application built with **Next.js, React, TypeScript, Tailwind CSS, and Three.js**.

This project connects image-based object detection across 10 custom classes with real-time multi-object tracking, 2D-to-3D spatial estimation when depth is available, and interactive 3D visualization.

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
- **Telemetry Console**: Database-backed detection totals, class distributions, active sessions, and recent activity (`app/dashboard/page.tsx`). Unmeasured latency, tracking, and depth-accuracy values are shown as unavailable.

---

## Getting Started

### Requirements
- Node.js (v18 or v20+)
- npm or yarn

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

Camera inference uses the CPU TensorFlow service and checked-in trained weights. For local Next.js development, start that service with Docker Compose:

```bash
docker compose up --build vision
```

`next dev` defaults to `http://localhost:8001/api/detect/`; the Compose web service uses `http://vision:8001/api/detect/`. Configure `VISION_API_URL` to a reachable inference service URL in production. Detection requests return `503` when no inference service is configured rather than displaying generated detections.

---

## Project Structure

```
├── app/                  # Next.js App Router pages and API routes
│   ├── camera/           # Live camera feed with detection overlay & HUD
│   ├── dashboard/        # Telemetry, analytics, and class statistics
│   ├── model-viewer/     # Three.js 3D point cloud & GLB mesh viewer
│   ├── replay/           # Spatial trajectory replay
│   └── api/              # API endpoints for detection, chat & 3D models
├── lib/                  # Core algorithms in TypeScript
│   ├── detector.ts       # Object detection logic & class thresholds
│   ├── tracker.ts        # Kalman box tracking (SORT)
│   ├── spatial-fusion.ts # Camera intrinsics & 3D back-projection
│   ├── scene-graph.ts    # Spatial relation builder
│   └── store.ts          # Client state store
├── core/                 # Computer vision & spatial geometry algorithms in Python
├── public/models/        # 3D GLB assets for detected object classes
└── legacy_django/        # Original Django backend and model training notebook
```

---

## Development Scripts

```bash
npm run dev      # Start development server
npm run build    # Build production bundle
npm run start    # Run production server
npm run lint     # Lint check with ESLint
```

---

## Performance Evidence

No production benchmark figures are currently published. The dashboard reports measured detection counts and labels performance or accuracy metrics as unavailable until they are collected from representative workloads. The TensorFlow detector has a CPU smoke test; it is not a throughput or accuracy benchmark.

## Model Training & Background

For details about how the dataset was collected, cleaned, and trained on Google Colab, see [`ProjectJourney.md`](ProjectJourney.md) and the training notebook in `legacy_django/144_2432214_YuyutsuLamsal_ObjectDetectionModel.ipynb`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
