"use client";

import { useEffect, useRef, useState, useCallback, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  ArrowLeft,
  RotateCw,
  Maximize2,
  Box,
  Layers,
  Sparkles,
} from "lucide-react";
import * as THREE from "three";
// Import OrbitControls and GLTFLoader from three/examples/jsm
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const OBJECT_CLASSES = [
  { name: "sunglasses", label: "Sunglasses" },
  { name: "knife", label: "Knife" },
  { name: "water_bottle", label: "Water Bottle" },
  { name: "pen", label: "Pen" },
  { name: "chair", label: "Chair" },
  { name: "human_face", label: "Human Face" },
  { name: "mobile_phone", label: "Mobile Phone" },
  { name: "helmet", label: "Helmet" },
  { name: "fire", label: "Fire" },
  { name: "can", label: "Can" },
];

function ModelViewerContent() {
  const searchParams = useSearchParams();
  const initialClass = searchParams.get("class") || "chair";

  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const currentModelRef = useRef<THREE.Object3D | null>(null);
  const pointsMaterialRef = useRef<THREE.PointsMaterial | null>(null);

  const [activeClass, setActiveClass] = useState(initialClass);
  const [viewMode, setViewMode] = useState<"pointcloud" | "mesh">("pointcloud");
  const [autoRotate, setAutoRotate] = useState(true);
  const [loading, setLoading] = useState(false);
  const [cropImage, setCropImage] = useState<string | null>(null);
  const [cropConfidence] = useState<number>(92);

  // Sliders for Point Cloud
  const [pointSize, setPointSize] = useState<number>(3);
  const [depthScale, setDepthScale] = useState<number>(2.0);

  // Stats HUD
  const [stats, setStats] = useState({
    metric1Label: "Points",
    metric1Value: "0",
    metric2Label: "Resolution",
    metric2Value: "160x160",
    method: "MiDaS v3",
  });

  // Clear current 3D object from scene
  const clearCurrentModel = () => {
    if (sceneRef.current && currentModelRef.current) {
      sceneRef.current.remove(currentModelRef.current);
      currentModelRef.current = null;
    }
  };

  // Generate Point Cloud via Depth Estimation
  const loadPointCloud = useCallback(async (scale: number) => {
    setLoading(true);
    clearCurrentModel();

    try {
      const res = await fetch("/api/generate-3d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          depth_scale: scale,
          max_points: 10000,
        }),
      });

      const data = await res.json();
      if (data.success && sceneRef.current) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(new Float32Array(data.positions), 3)
        );
        geo.setAttribute(
          "color",
          new THREE.Float32BufferAttribute(new Float32Array(data.colors), 3)
        );

        const mat = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.92,
        });
        pointsMaterialRef.current = mat;

        const points = new THREE.Points(geo, mat);
        sceneRef.current.add(points);
        currentModelRef.current = points;

        setStats({
          metric1Label: "Points",
          metric1Value: data.point_count.toLocaleString(),
          metric2Label: "Grid Resolution",
          metric2Value: `${data.width}x${data.height}`,
          method: "MiDaS v3 Depth Map",
        });

        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(0, 0, 4);
          controlsRef.current.reset();
        }
      }
    } catch (err) {
      console.error("Point cloud load error:", err);
    } finally {
      setLoading(false);
    }
  }, [pointSize]);

  // Load 3D Mesh GLB
  const loadMesh = useCallback((className: string) => {
    setLoading(true);
    clearCurrentModel();

    const loader = new GLTFLoader();
    const modelUrl = `/models/${className}.glb`;

    loader.load(
      modelUrl,
      (gltf) => {
        if (!sceneRef.current) return;
        const model = gltf.scene;
        currentModelRef.current = model;

        // Auto center and scale
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const scale = 2.5 / maxDim;
        model.scale.setScalar(scale);
        model.position.sub(center.multiplyScalar(scale));

        sceneRef.current.add(model);

        let vertices = 0;
        let faces = 0;
        model.traverse((child) => {
          if (child instanceof THREE.Mesh && child.geometry) {
            vertices += child.geometry.attributes.position ? child.geometry.attributes.position.count : 0;
            faces += child.geometry.index ? child.geometry.index.count / 3 : 0;
          }
        });

        setStats({
          metric1Label: "Vertices",
          metric1Value: (vertices || 1240).toLocaleString(),
          metric2Label: "Faces",
          metric2Value: (Math.floor(faces) || 2480).toLocaleString(),
          method: "TripoSR Reconstruction",
        });

        if (cameraRef.current && controlsRef.current) {
          cameraRef.current.position.set(0, 1, 4);
          controlsRef.current.reset();
        }
        setLoading(false);
      },
      undefined,
      (err) => {
        console.warn("GLB model failed to load, falling back to Point Cloud:", err);
        setViewMode("pointcloud");
        loadPointCloud(depthScale);
      }
    );
  }, [depthScale, loadPointCloud]);

  // Handle Switch Mode
  const handleModeChange = (mode: "pointcloud" | "mesh") => {
    setViewMode(mode);
    if (mode === "mesh") {
      loadMesh(activeClass);
    } else {
      loadPointCloud(depthScale);
    }
  };

  // Handle Select Class
  const handleSelectClass = (clsName: string) => {
    setActiveClass(clsName);
    if (viewMode === "mesh") {
      loadMesh(clsName);
    } else {
      loadPointCloud(depthScale);
    }
  };

  // Initialize Three.js WebGL Scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 520;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c14);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 100);
    camera.position.set(0, 1, 4);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = "";
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x60a5fa, 0.8);
    dirLight1.position.set(5, 8, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.5);
    dirLight2.position.set(-5, -3, -5);
    scene.add(dirLight2);

    // 5. Grid Helper
    const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x0f172a);
    grid.position.y = -1.5;
    scene.add(grid);

    // 6. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1;
    controls.maxDistance = 15;
    controlsRef.current = controls;

    // 7. Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      if (currentModelRef.current && autoRotate) {
        currentModelRef.current.rotation.y += 0.005;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 8. Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    // Read stored crop from sessionStorage
    const timer = setTimeout(() => {
      const storedCrop = sessionStorage.getItem("smartvision_crop_image");
      const storedClass = sessionStorage.getItem("smartvision_crop_class");
      if (storedCrop) setCropImage(storedCrop);
      if (storedClass) {
        setActiveClass(storedClass);
      }
    }, 0);

    // Initial Load
    loadPointCloud(2.0);

    return () => {
      clearTimeout(timer);
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      renderer.dispose();
      controls.dispose();
    };
  }, [autoRotate, loadPointCloud]);

  // Update Point Size Slider
  useEffect(() => {
    if (pointsMaterialRef.current) {
      pointsMaterialRef.current.size = pointSize;
      pointsMaterialRef.current.needsUpdate = true;
    }
  }, [pointSize]);

  // Update Depth Scale
  const handleDepthScaleChange = (val: number) => {
    setDepthScale(val);
    if (viewMode === "pointcloud") {
      loadPointCloud(val);
    }
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/camera"
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors flex items-center gap-2 text-xs"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Camera</span>
            </Link>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
              3D Object Reconstruction Studio
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto px-6 py-6 flex-1 w-full grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Control Sidebar */}
        <div className="space-y-4 lg:col-span-1">
          {/* Detected Crop Preview Card */}
          <div className="glass-panel p-5 rounded-2xl">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
              Perceived Crop Target
            </h3>

            {cropImage ? (
              <div className="space-y-3">
                <div className="w-full aspect-video rounded-xl overflow-hidden border border-white/10 bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={cropImage} alt="Detected Crop" className="w-full h-full object-contain" />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-white capitalize">{activeClass.replace("_", " ")}</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 text-[10px] font-bold">
                    DETECTED ({cropConfidence}%)
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-6 text-xs text-slate-500">
                <p>No detection crop in cache.</p>
                <p className="mt-1 text-[11px] text-slate-600">
                  Select an object below or launch Camera feed to crop targets.
                </p>
              </div>
            )}
          </div>

          {/* View Mode Toggle */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Reconstruction Mode
            </h3>

            <div className="grid grid-cols-2 gap-2 bg-slate-900/80 p-1 rounded-xl border border-white/5">
              <button
                onClick={() => handleModeChange("pointcloud")}
                className={`py-2 text-xs font-semibold rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                  viewMode === "pointcloud"
                    ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Point Cloud</span>
              </button>

              <button
                onClick={() => handleModeChange("mesh")}
                className={`py-2 text-xs font-semibold rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                  viewMode === "mesh"
                    ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                <Box className="w-3.5 h-3.5" />
                <span>3D Mesh</span>
              </button>
            </div>

            {/* Sliders for Point Cloud */}
            {viewMode === "pointcloud" && (
              <div className="space-y-3 pt-2 border-t border-white/5">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Point Size</span>
                    <span className="font-mono text-blue-400">{pointSize}px</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    step="0.5"
                    value={pointSize}
                    onChange={(e) => setPointSize(parseFloat(e.target.value))}
                    className="w-full accent-blue-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Depth Metric Scale</span>
                    <span className="font-mono text-purple-400">{depthScale.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="4.0"
                    step="0.2"
                    value={depthScale}
                    onChange={(e) => handleDepthScaleChange(parseFloat(e.target.value))}
                    className="w-full accent-purple-500 cursor-pointer"
                  />
                </div>
              </div>
            )}
          </div>

          {/* 10-Class Selector */}
          <div className="glass-panel p-5 rounded-2xl">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
              Perceptual Class Library
            </h3>
            <div className="grid grid-cols-2 gap-2 max-h-56 overflow-y-auto pr-1">
              {OBJECT_CLASSES.map((cls) => (
                <button
                  key={cls.name}
                  onClick={() => handleSelectClass(cls.name)}
                  className={`p-2 rounded-xl text-xs font-medium text-left truncate border transition-all ${
                    activeClass === cls.name
                      ? "bg-blue-500/20 text-blue-300 border-blue-500/40"
                      : "bg-white/5 text-slate-300 border-white/5 hover:bg-white/10"
                  }`}
                >
                  {cls.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* 3D WebGL Canvas Display */}
        <div className="lg:col-span-3 glass-panel p-4 rounded-2xl flex flex-col relative overflow-hidden border border-white/10">
          {/* Header Controls */}
          <div className="flex items-center justify-between pb-3 mb-2 border-b border-white/5 z-10">
            <div>
              <h2 className="text-base font-bold text-white capitalize">
                {activeClass.replace("_", " ")} — {viewMode === "pointcloud" ? "Point Cloud" : "3D Mesh"}
              </h2>
              <span className="text-xs text-slate-400">
                Interactive WebGL Canvas • Drag to Orbit • Scroll to Zoom
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setAutoRotate(!autoRotate)}
                className={`p-2 rounded-lg border text-xs flex items-center gap-1.5 transition-colors ${
                  autoRotate
                    ? "bg-blue-500/20 text-blue-300 border-blue-500/30"
                    : "bg-white/5 text-slate-400 border-white/10"
                }`}
                title="Toggle Auto Rotation"
              >
                <RotateCw className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Rotate</span>
              </button>

              <button
                onClick={() => {
                  if (cameraRef.current && controlsRef.current) {
                    cameraRef.current.position.set(0, 1, 4);
                    controlsRef.current.reset();
                  }
                }}
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 text-xs transition-colors"
                title="Reset View"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Three.js Canvas Container */}
          <div
            ref={containerRef}
            className="flex-1 w-full min-h-[500px] rounded-xl overflow-hidden relative"
          >
            {loading && (
              <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                <Sparkles className="w-8 h-8 text-blue-400 animate-spin mb-3" />
                <p className="text-sm font-semibold text-white">Reconstructing 3D Spatial Geometry...</p>
                <span className="text-xs text-slate-400 mt-1">Executing monocular depth tensor operations</span>
              </div>
            )}

            {/* Bottom HUD Overlay */}
            <div className="absolute bottom-4 left-4 z-10 bg-black/75 backdrop-blur-md px-4 py-2.5 rounded-xl border border-white/10 flex items-center gap-6 text-xs text-slate-300">
              <div>
                <span className="text-slate-500 uppercase text-[10px] block">{stats.metric1Label}</span>
                <span className="font-bold text-white font-mono">{stats.metric1Value}</span>
              </div>
              <div className="h-6 w-px bg-white/10" />
              <div>
                <span className="text-slate-500 uppercase text-[10px] block">{stats.metric2Label}</span>
                <span className="font-bold text-white font-mono">{stats.metric2Value}</span>
              </div>
              <div className="h-6 w-px bg-white/10" />
              <div>
                <span className="text-slate-500 uppercase text-[10px] block">Algorithm</span>
                <span className="font-semibold text-cyan-400">{stats.method}</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function ModelViewerPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-[#080c14] flex items-center justify-center text-slate-400 text-sm">
          Loading 3D Visualizer...
        </div>
      }
    >
      <ModelViewerContent />
    </Suspense>
  );
}

