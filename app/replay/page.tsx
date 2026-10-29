"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Camera, Pause, Play, RotateCcw } from "lucide-react";
import type { Detection } from "@/lib/types";

interface ReplayFrame {
  capturedAt: number;
  detections: Detection[];
}

const STORAGE_KEY = "smartvision_replay_v1";

export default function ReplayPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [frames, setFrames] = useState<ReplayFrame[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const currentFrame = frames[frameIndex];

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      if (Array.isArray(saved)) {
        const valid = saved.filter(
          (frame): frame is ReplayFrame =>
            Number.isFinite(frame?.capturedAt) && Array.isArray(frame?.detections)
        );
        setFrames(valid);
        setFrameIndex(Math.max(0, valid.length - 1));
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setFrameIndex((index) => {
        if (index >= frames.length - 1) {
          setPlaying(false);
          return index;
        }
        return index + 1;
      });
    }, 140);
    return () => window.clearInterval(timer);
  }, [playing, frames.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    context.scale(pixelRatio, pixelRatio);
    context.fillStyle = "#11191d";
    context.fillRect(0, 0, width, height);

    context.strokeStyle = "rgba(179, 207, 194, 0.12)";
    context.lineWidth = 1;
    for (let x = 30; x < width; x += 40) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
    for (let y = 30; y < height; y += 40) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    const pointsByTrack = new Map<number, Array<{ x: number; y: number }>>();
    frames.slice(0, frameIndex + 1).forEach((frame) => {
      frame.detections.forEach((detection, detectionIndex) => {
        const trackId = detection.track_id ?? -(detectionIndex + 1);
        const point = detection.spatial_3d
          ? {
              x: width / 2 + detection.spatial_3d.x * 18,
              y: height / 2 - detection.spatial_3d.z * 18,
            }
          : {
              x: ((detection.bbox[0] + detection.bbox[2]) / 2 / 1280) * width,
              y: ((detection.bbox[1] + detection.bbox[3]) / 2 / 720) * height,
            };
        const track = pointsByTrack.get(trackId) || [];
        track.push(point);
        pointsByTrack.set(trackId, track);
      });
    });

    const palette = ["#78c7a2", "#e4b85c", "#6cb9cf", "#e47b68", "#b0a1d1"];
    Array.from(pointsByTrack.entries()).forEach(([trackId, points], index) => {
      if (points.length > 1) {
        context.beginPath();
        points.forEach((point, pointIndex) => {
          if (pointIndex === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
        });
        context.strokeStyle = palette[index % palette.length];
        context.globalAlpha = 0.55;
        context.lineWidth = 2;
        context.stroke();
        context.globalAlpha = 1;
      }
      const point = points[points.length - 1];
      if (!point) return;
      context.beginPath();
      context.arc(point.x, point.y, 5, 0, Math.PI * 2);
      context.fillStyle = palette[index % palette.length];
      context.fill();
      context.fillStyle = "#e6eee9";
      context.font = "11px monospace";
      context.fillText(trackId > 0 ? `#${trackId}` : "untracked", point.x + 8, point.y - 7);
    });
  }, [frames, frameIndex]);

  const clearReplay = () => {
    localStorage.removeItem(STORAGE_KEY);
    setFrames([]);
    setFrameIndex(0);
    setPlaying(false);
  };

  return (
    <main className="min-h-screen bg-[#111715] text-[#e7eee9]">
      <header className="border-b border-white/10 bg-[#151d19]">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" aria-label="Back to dashboard" className="rounded border border-white/10 p-2 text-[#b8c5bc] hover:bg-white/5">
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <div>
              <p className="text-[11px] uppercase tracking-[0.16em] text-[#8b9b91]">SmartVision / Review</p>
              <h1 className="text-sm font-semibold">Spatial replay</h1>
            </div>
          </div>
          <Link href="/camera" className="inline-flex items-center gap-2 rounded border border-[#78c7a2]/30 px-3 py-2 text-xs text-[#a6dfc2] hover:bg-[#78c7a2]/10">
            <Camera className="h-4 w-4" />
            Open camera
          </Link>
        </div>
      </header>

      <section className="mx-auto max-w-7xl space-y-5 px-5 py-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs text-[#92a197]">LOCAL CAPTURE · UP TO 240 FRAMES</p>
            <h2 className="mt-1 text-xl font-semibold">Track history</h2>
          </div>
          <div className="flex items-center gap-2 text-xs text-[#92a197]">
            <span>{frames.length.toLocaleString()} frames</span>
            {currentFrame && <span>· {new Date(currentFrame.capturedAt).toLocaleTimeString()}</span>}
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-w-0 border border-white/10 bg-[#151d19]">
            <canvas ref={canvasRef} className="block h-[min(58vh,560px)] min-h-[300px] w-full" aria-label="Historical object trajectories" />
            <div className="border-t border-white/10 p-4">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setPlaying((value) => !value)}
                  disabled={frames.length < 2}
                  aria-label={playing ? "Pause replay" : "Play replay"}
                  className="rounded border border-white/10 p-2 text-[#dce7df] hover:bg-white/5 disabled:opacity-40"
                >
                  {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, frames.length - 1)}
                  value={frameIndex}
                  onChange={(event) => setFrameIndex(Number(event.target.value))}
                  disabled={frames.length === 0}
                  aria-label="Replay frame"
                  className="h-1 min-w-0 flex-1 accent-[#78c7a2]"
                />
                <span className="w-20 text-right font-mono text-[11px] text-[#9dacA2]">
                  {frames.length ? `${frameIndex + 1} / ${frames.length}` : "0 / 0"}
                </span>
              </div>
              <div className="mt-3 flex items-center justify-between">
                <p className="text-[11px] text-[#84948a]">Trajectories use metric depth when available; otherwise image-space centers.</p>
                <button onClick={clearReplay} title="Clear local replay" className="rounded p-2 text-[#9ba9a0] hover:bg-white/5 hover:text-white">
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          <aside className="border border-white/10 bg-[#151d19]">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-[#cbd6ce]">Frame detections</h3>
              <span className="font-mono text-[11px] text-[#8d9d92]">{currentFrame?.detections.length ?? 0}</span>
            </div>
            <div className="max-h-[min(58vh,560px)] min-h-[300px] divide-y divide-white/5 overflow-y-auto">
              {currentFrame?.detections.map((detection, index) => (
                <div key={`${detection.track_id ?? detection.class}-${index}`} className="flex items-center justify-between gap-3 px-4 py-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm capitalize">{detection.class.replaceAll("_", " ")}</p>
                    <p className="mt-1 font-mono text-[10px] text-[#829188]">
                      {detection.track_id ? `#${detection.track_id}` : "untracked"}
                      {detection.spatial_3d ? ` · ${detection.spatial_3d.distance_m.toFixed(2)} m` : ""}
                    </p>
                  </div>
                  <span className="shrink-0 font-mono text-xs text-[#a8d7bd]">{Math.round(detection.confidence * 100)}%</span>
                </div>
              ))}
              {!currentFrame && (
                <div className="flex min-h-[300px] flex-col items-center justify-center px-6 text-center">
                  <p className="text-sm text-[#d2ddd5]">No captured frames</p>
                  <p className="mt-2 text-xs leading-relaxed text-[#87958c]">Run detection in Camera Studio to build a local replay history.</p>
                  <Link href="/camera" className="mt-4 text-xs text-[#9ed6b8] hover:text-white">Go to Camera Studio</Link>
                </div>
              )}
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}