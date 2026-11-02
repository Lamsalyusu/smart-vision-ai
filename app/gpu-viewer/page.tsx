"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from "react";
import Link from "next/link";
import { ArrowLeft, Cpu, Gauge, Sparkles } from "lucide-react";

import { detectGpuBackend, type GpuBackend } from "@/lib/gpu-capabilities";
import { generatePointCloud } from "@/lib/geometry-accel";
import { WebGpuPointCloudRenderer } from "@/lib/webgpu-pointcloud";

const POINT_COUNTS = [100_000, 250_000, 500_000, 1_000_000];

export default function GpuViewerPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rendererRef = useRef<WebGpuPointCloudRenderer | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [backend, setBackend] = useState<GpuBackend>("none");
  const [pointCount, setPointCount] = useState(250_000);
  const [fps, setFps] = useState(0);
  const [message, setMessage] = useState("Initialising WebGPU renderer…");

  useEffect(() => {
    setBackend(detectGpuBackend());
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let fpsTimer: ReturnType<typeof setInterval> | null = null;

    const { positions, colors } = generatePointCloud(pointCount);
    WebGpuPointCloudRenderer.create(canvas, { positions, colors })
      .then((renderer) => {
        if (cancelled) {
          renderer.dispose();
          return;
        }
        rendererRef.current?.dispose();
        rendererRef.current = renderer;
        renderer.start();
        setMessage("");
        fpsTimer = setInterval(() => setFps(renderer.fps), 500);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setMessage(
          error instanceof Error
            ? `WebGPU unavailable: ${error.message}`
            : "WebGPU is unavailable in this browser",
        );
      });

    return () => {
      cancelled = true;
      if (fpsTimer) clearInterval(fpsTimer);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, [pointCount]);

  const handlePointerDown = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const handlePointerMove = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    rendererRef.current?.orbit(event.clientX - drag.x, event.clientY - drag.y);
    dragRef.current = { x: event.clientX, y: event.clientY };
  }, []);

  const handlePointerUp = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handleWheel = useCallback((event: ReactWheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    rendererRef.current?.zoom(event.deltaY * 0.002);
  }, []);

  const supported = backend === "webgpu" && !message;

  return (
    <div className="min-h-screen bg-[#05080f] text-slate-100 flex flex-col">
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link
            href="/model-viewer"
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors flex items-center gap-2 text-xs"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to 3D Studio</span>
          </Link>
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5" />
              {backend === "webgpu" ? "WebGPU" : backend.toUpperCase()}
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-medium flex items-center gap-1">
              <Gauge className="w-3.5 h-3.5" />
              {fps.toFixed(1)} FPS
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-6">
        <section className="relative rounded-2xl border border-white/10 bg-black/50 overflow-hidden min-h-[420px]">
          <canvas
            ref={canvasRef}
            className="w-full h-full min-h-[420px] block touch-none cursor-grab"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onWheel={handleWheel}
          />
          {message && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-8 text-center text-sm text-slate-300">
              <div className="max-w-md space-y-3">
                <Sparkles className="w-6 h-6 mx-auto text-blue-400" />
                <p>{message}</p>
                <p className="text-xs text-slate-500">
                  The Three.js/WebGL viewer remains available for all browsers.
                </p>
                <Link
                  href="/model-viewer"
                  className="inline-block px-3 py-1.5 rounded-lg bg-blue-500/20 text-blue-300 text-xs font-medium"
                >
                  Open WebGL viewer
                </Link>
              </div>
            </div>
          )}
        </section>

        <aside className="space-y-5">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 space-y-3">
            <h2 className="text-sm font-semibold text-slate-200">Point Cloud</h2>
            <div className="grid grid-cols-2 gap-2">
              {POINT_COUNTS.map((count) => (
                <button
                  key={count}
                  type="button"
                  onClick={() => setPointCount(count)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
                    pointCount === count
                      ? "bg-blue-500/20 border-blue-500/40 text-blue-300"
                      : "bg-white/5 border-white/10 text-slate-400 hover:bg-white/10"
                  }`}
                >
                  {(count / 1000).toLocaleString()}k
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              Drag to orbit, scroll to zoom. FPS is sampled over 500 ms and
              reflects real rendering throughput on this device.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 space-y-2 text-xs text-slate-400">
            <div className="flex justify-between">
              <span>Backend</span>
              <span className="text-slate-200">
                {supported ? "WebGPU" : "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Points submitted</span>
              <span className="text-slate-200">{pointCount.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Distance</span>
              <span className="text-slate-200">GPU vertex buffers</span>
            </div>
            <div className="flex justify-between">
              <span>CPU per-frame work</span>
              <span className="text-slate-200">1 matrix upload</span>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
