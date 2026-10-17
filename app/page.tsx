import Link from "next/link";
import {
  Camera,
  Box,
  Layers,
  Cpu,
  Activity,
  ArrowRight,
  Shield,
  Eye,
  Crosshair,
  Sparkles,
} from "lucide-react";

export default function LandingPage() {
  const classes = [
    { name: "sunglasses", label: "Sunglasses", color: "from-blue-500/20 to-blue-600/10 text-blue-400 border-blue-500/30" },
    { name: "knife", label: "Knife", color: "from-red-500/20 to-red-600/10 text-red-400 border-red-500/30" },
    { name: "water_bottle", label: "Water Bottle", color: "from-cyan-500/20 to-cyan-600/10 text-cyan-400 border-cyan-500/30" },
    { name: "pen", label: "Pen", color: "from-amber-500/20 to-amber-600/10 text-amber-400 border-amber-500/30" },
    { name: "chair", label: "Chair", color: "from-emerald-500/20 to-emerald-600/10 text-emerald-400 border-emerald-500/30" },
    { name: "human_face", label: "Human Face", color: "from-pink-500/20 to-pink-600/10 text-pink-400 border-pink-500/30" },
    { name: "mobile_phone", label: "Mobile Phone", color: "from-purple-500/20 to-purple-600/10 text-purple-400 border-purple-500/30" },
    { name: "helmet", label: "Helmet", color: "from-orange-500/20 to-orange-600/10 text-orange-400 border-orange-500/30" },
    { name: "fire", label: "Fire", color: "from-rose-500/20 to-rose-600/10 text-rose-400 border-rose-500/30" },
    { name: "can", label: "Can", color: "from-teal-500/20 to-teal-600/10 text-teal-400 border-teal-500/30" },
  ];

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white">
      {/* Navigation */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/25">
              <Eye className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                SmartVision AI
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
                v2.4 Pro
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-sm font-medium text-slate-300 hover:text-white transition-colors px-3 py-1.5"
            >
              Sign In
            </Link>
            <Link
              href="/camera"
              className="btn-cyber-primary text-sm px-4 py-2 flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              <span>Launch Studio</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative pt-20 pb-24 overflow-hidden border-b border-white/5">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_-20%,rgba(59,130,246,0.18),rgba(255,255,255,0))]" />

        <div className="max-w-6xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold mb-8 animate-fade-in">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Next.js & TypeScript Spatial AI Engine</span>
          </div>

          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight max-w-4xl mx-auto leading-tight sm:leading-none mb-6">
            Real-time Spatial Perception &{" "}
            <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-purple-400 bg-clip-text text-transparent">
              3D Metric Fusion
            </span>
          </h1>

          <p className="text-lg sm:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed font-normal">
            Multi-scale neural detection, SORT Kalman tracking, camera intrinsic unprojection, and interactive Three.js 3D WebGL point clouds.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/camera"
              className="w-full sm:w-auto btn-cyber-primary px-8 py-3.5 text-base flex items-center justify-center gap-2.5"
            >
              <Camera className="w-5 h-5" />
              <span>Open Live Camera Feed</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              href="/dashboard"
              className="w-full sm:w-auto px-7 py-3.5 text-base font-semibold rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all flex items-center justify-center gap-2 text-slate-200"
            >
              <Activity className="w-5 h-5 text-blue-400" />
              <span>Perception Dashboard</span>
            </Link>
          </div>

          {/* Quick Metrics Bar */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto text-left">
            <div className="glass-card p-4 rounded-xl">
              <span className="text-xs text-slate-400 uppercase font-semibold">Latency</span>
              <p className="text-2xl font-bold text-white mt-1">6.38 ms</p>
              <span className="text-xs text-emerald-400 font-medium">~156 FPS throughput</span>
            </div>
            <div className="glass-card p-4 rounded-xl">
              <span className="text-xs text-slate-400 uppercase font-semibold">Tracking</span>
              <p className="text-2xl font-bold text-white mt-1">0.06 ms</p>
              <span className="text-xs text-blue-400 font-medium">7-state Kalman filter</span>
            </div>
            <div className="glass-card p-4 rounded-xl">
              <span className="text-xs text-slate-400 uppercase font-semibold">3D Metric Depth</span>
              <p className="text-2xl font-bold text-white mt-1">0.15 ms</p>
              <span className="text-xs text-purple-400 font-medium">Pinhole Back-Projection</span>
            </div>
            <div className="glass-card p-4 rounded-xl">
              <span className="text-xs text-slate-400 uppercase font-semibold">Scene Reasoning</span>
              <p className="text-2xl font-bold text-white mt-1">G = (V, E)</p>
              <span className="text-xs text-cyan-400 font-medium">Topological graph</span>
            </div>
          </div>
        </div>
      </section>

      {/* 10-Class Perceptual Spectrum */}
      <section className="py-20 border-b border-white/5 relative bg-[#090e18]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <span className="text-xs font-semibold uppercase tracking-wider text-blue-400">Trained Object Classes</span>
            <h2 className="text-3xl font-bold mt-2">10-Class Perception Matrix</h2>
            <p className="text-slate-400 text-sm mt-2">
              Optimized with class-specific anchor scales, IoU associations, and 3D geometric priors.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5">
            {classes.map((cls) => (
              <div
                key={cls.name}
                className={`p-4 rounded-xl bg-gradient-to-b ${cls.color} border flex flex-col items-center text-center transition-all hover:scale-105`}
              >
                <div className="w-10 h-10 rounded-lg bg-black/40 flex items-center justify-center mb-3">
                  <Crosshair className="w-5 h-5" />
                </div>
                <span className="font-semibold text-sm text-white capitalize">{cls.label}</span>
                <span className="text-xs text-slate-400 mt-1 font-mono">{cls.name}.glb</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Core Architectural Pillars */}
      <section className="py-24 max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">System Capabilities</span>
          <h2 className="text-3xl sm:text-4xl font-bold mt-2">Four-Stage Spatial Perception Pipeline</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-panel p-8">
            <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mb-6">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-3">1. Multi-Scale YOLO Architecture</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              CSPDarknet backbone with SPPF feature aggregation and FPN multi-scale feature pyramids. Vectorized post-processing filters detections with zero thread blocking.
            </p>
          </div>

          <div className="glass-panel p-8">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mb-6">
              <Cpu className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-3">2. SORT Multi-Object Kalman Tracking</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Continuous 7-state motion modeling `[u, v, s, r, u_dot, v_dot, s_dot]` tracking targets smoothly across frames with bipartite IoU association.
            </p>
          </div>

          <div className="glass-panel p-8">
            <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center mb-6">
              <Box className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-3">3. RGB-D Metric 3D Coordinate Fusion</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Back-projects pixel coordinates into metric 3D camera coordinates (X, Y, Z) and Euclidean distances using calibrated pinhole intrinsics and elevation compensation.
            </p>
          </div>

          <div className="glass-panel p-8">
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 flex items-center justify-center mb-6">
              <Shield className="w-6 h-6" />
            </div>
            <h3 className="text-xl font-bold mb-3">4. Relational Scene Graph & 3D Viewer</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Generates topological interaction predicates (`holding`, `wearing`, `supporting`, `adjacent_to`) paired with interactive WebGL point cloud and mesh inspection.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black/60 py-8 mt-auto">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <p>© 2026 SmartVision AI. Built with Next.js, TypeScript, Tailwind CSS & React.</p>
          <div className="flex items-center gap-6">
            <Link href="/camera" className="hover:text-slate-300 transition-colors">Camera Studio</Link>
            <Link href="/dashboard" className="hover:text-slate-300 transition-colors">Dashboard</Link>
            <Link href="/model-viewer" className="hover:text-slate-300 transition-colors">3D Viewer</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
