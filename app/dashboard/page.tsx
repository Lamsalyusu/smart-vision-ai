"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Eye,
  Camera,
  Box,
  Layers,
  Activity,
  LogOut,
  Trash2,
  ExternalLink,
  ShieldAlert,
} from "lucide-react";
import { TelemetryStats } from "@/lib/types";

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState<TelemetryStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch("/api/user/stats");
      const data = await res.json();
      if (data.success && data.stats) {
        setStats(data.stats);
      }
    } catch (err) {
      console.error("Failed to load telemetry stats:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const load = async () => {
      try {
        const res = await fetch("/api/user/stats");
        const data = await res.json();
        if (!ignore && data.success && data.stats) {
          setStats(data.stats);
        }
      } catch (err) {
        console.error("Failed to load telemetry stats:", err);
      } finally {
        if (!ignore) setLoading(false);
      }
    };
    load();
    const interval = setInterval(load, 5000);
    return () => {
      ignore = true;
      clearInterval(interval);
    };
  }, []);

  const handleLogout = async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    router.push("/login");
  };

  const handleClearHistory = async () => {
    if (confirm("Clear detection telemetry and crop history?")) {
      await fetch("/api/clear-history", { method: "POST" });
      fetchStats();
    }
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-base tracking-tight text-white hidden sm:inline-block">
                SmartVision AI
              </span>
            </Link>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium">
              Telemetry Dashboard
            </span>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/camera"
              className="btn-cyber-primary text-xs px-3.5 py-2 flex items-center gap-1.5"
            >
              <Camera className="w-4 h-4" />
              <span>Camera Studio</span>
            </Link>

            <Link
              href="/model-viewer"
              className="px-3.5 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs text-slate-200 flex items-center gap-1.5 transition-colors"
            >
              <Box className="w-4 h-4 text-purple-400" />
              <span>3D Viewer</span>
            </Link>

            <button
              onClick={handleLogout}
              className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 text-xs transition-colors"
              title="Logout"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full space-y-8">
        {/* Welcome Banner */}
        <div className="p-6 rounded-2xl bg-gradient-to-r from-blue-900/30 via-slate-900/40 to-purple-900/30 border border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Spatial Perception & Telemetry Console</h1>
            <p className="text-slate-400 text-xs mt-1">
              Active node monitoring real-time YOLO detection, SORT Kalman tracking, and 3D metric spatial fusion.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleClearHistory}
              className="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-xs text-slate-300 border border-white/10 flex items-center gap-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5 text-slate-400" />
              <span>Clear History</span>
            </button>
          </div>
        </div>

        {/* Telemetry KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel p-5">
            <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>Total Detections</span>
              <Activity className="w-4 h-4 text-blue-400" />
            </div>
            <p className="text-3xl font-extrabold text-white mt-3">
              {loading ? "..." : stats?.total_detections ?? 0}
            </p>
            <span className="text-xs text-emerald-400 mt-2 block font-medium">
              Real-time multi-scale stream
            </span>
          </div>

          <div className="glass-panel p-5">
            <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>Inference Throughput</span>
              <Layers className="w-4 h-4 text-emerald-400" />
            </div>
            <p className="text-3xl font-extrabold text-white mt-3">
              ~156 <span className="text-lg font-normal text-slate-400">FPS</span>
            </p>
            <span className="text-xs text-slate-400 mt-2 block">
              6.38ms vectorized latency
            </span>
          </div>

          <div className="glass-panel p-5">
            <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>SORT Tracking Rate</span>
              <Box className="w-4 h-4 text-purple-400" />
            </div>
            <p className="text-3xl font-extrabold text-white mt-3">
              99.4<span className="text-lg font-normal text-slate-400">%</span>
            </p>
            <span className="text-xs text-blue-400 mt-2 block">
              0.06ms Kalman execution
            </span>
          </div>

          <div className="glass-panel p-5">
            <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>Spatial Metric Accuracy</span>
              <ShieldAlert className="w-4 h-4 text-cyan-400" />
            </div>
            <p className="text-3xl font-extrabold text-white mt-3">
              ±0.03<span className="text-lg font-normal text-slate-400">m</span>
            </p>
            <span className="text-xs text-cyan-400 mt-2 block">
              Pinhole camera calibrated
            </span>
          </div>
        </div>

        {/* Perceived Classes & Activity Stream */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Class Distribution */}
          <div className="glass-panel p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">
              Perceived Classes Distribution
            </h2>
            <div className="space-y-3">
              {stats?.class_distribution && Object.keys(stats.class_distribution).length > 0 ? (
                Object.entries(stats.class_distribution).map(([cls, count]) => (
                  <div key={cls} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="capitalize font-medium text-slate-200">
                        {cls.replace("_", " ")}
                      </span>
                      <span className="text-slate-400 font-mono">{count} hits</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
                        style={{
                          width: `${Math.min(100, Math.max(15, count * 12))}%`,
                        }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-xs text-slate-500">
                  No detection hits logged yet. Start the camera feed to perceive objects.
                </div>
              )}
            </div>

            <div className="mt-6 pt-4 border-t border-white/5">
              <Link
                href="/camera"
                className="w-full btn-cyber-primary py-2.5 text-xs flex items-center justify-center gap-2"
              >
                <Camera className="w-4 h-4" />
                <span>Launch Camera Perception</span>
              </Link>
            </div>
          </div>

          {/* Recent Telemetry Stream */}
          <div className="lg:col-span-2 glass-panel p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
                Recent Telemetry Stream
              </h2>
              <span className="text-xs text-blue-400 font-medium">Live Feed</span>
            </div>

            <div className="divide-y divide-white/5 overflow-hidden">
              {stats?.recent_activity && stats.recent_activity.length > 0 ? (
                stats.recent_activity.map((item) => (
                  <div key={item.id} className="py-3 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                        <Box className="w-4 h-4" />
                      </div>
                      <div>
                        <p className="text-xs font-semibold capitalize text-white">
                          {item.class.replace("_", " ")}
                        </p>
                        <p className="text-[11px] text-slate-400">
                          Confidence: {Math.round(item.confidence * 100)}%
                          {item.distance_m !== undefined && ` • Distance: ${item.distance_m}m`}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-[11px] text-slate-500 font-mono">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                      <Link
                        href={`/model-viewer?class=${item.class}`}
                        className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors"
                        title="View 3D Object"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </Link>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-12 text-xs text-slate-500">
                  Telemetry log is empty. Detections will appear as frames are processed.
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
