# SmartVision AI

Real-time spatial perception and object detection web application built with **Next.js, React, TypeScript, Tailwind CSS, and Three.js**.

This project connects computer vision object detection across 10 custom classes with real-time multi-object tracking, 2D-to-3D spatial estimation using camera intrinsics, and interactive 3D model visualization.

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

- **Frontend**: Next.js 15 (App Router), React 19, TypeScript
- **Styling**: Tailwind CSS
- **3D Visualization**: Three.js & OrbitControls
- **Icons**: Lucide React
- **Backend / Training**: Python, PyTorch / YOLO model trained on Roboflow custom dataset (see `legacy_django/` and model training notebook)

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

---

## Project Structure

```
smart-vision-ai/
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

## Model Training & Background

For details about how the dataset was collected, cleaned, and trained on Google Colab, see [`ProjectJourney.md`](ProjectJourney.md) and the training notebook in `legacy_django/144_2432214_YuyutsuLamsal_ObjectDetectionModel.ipynb`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
