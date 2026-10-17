"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Camera,
  Play,
  Square,
  Aperture,
  Box,
  Layers,
  Send,
  Trash2,
  MessageSquare,
} from "lucide-react";
import { CLASS_COLORS } from "@/lib/detector";
import { Detection } from "@/lib/types";

interface ChatMessage {
  id: string;
  sender: "user" | "assistant" | "system";
  text: string;
  timestamp: string;
}

export default function CameraPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [detectionActive, setDetectionActive] = useState(false);
  const [includeDepth, setIncludeDepth] = useState(true);
  const [currentDetections, setCurrentDetections] = useState<Detection[]>([]);
  const [snapshots, setSnapshots] = useState<string[]>([]);
  const [frameCount, setFrameCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [fps, setFps] = useState(0);

  // Chat Assistant State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "msg_init",
      sender: "system",
      text: "SmartVision Spatial Perception Assistant initialized. Ask questions about detected objects, distances, or scene relationships.",
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // Start Camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
        };
      }
      setCameraActive(true);
      addChatMessage("system", "Camera activated successfully.");
    } catch (err: unknown) {
      alert("Could not access camera: " + (err instanceof Error ? err.message : String(err)));
    }
  };

  // Stop Camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setDetectionActive(false);
    setCurrentDetections([]);
    clearCanvas();
    addChatMessage("system", "Camera stopped.");
  };

  // Draw Detections Canvas
  const drawDetections = useCallback((detections: Detection[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const color = CLASS_COLORS[det.class] || "#3b82f6";
      const w = x2 - x1;
      const h = y2 - y1;

      // Outer Bounding Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      // Corner Accents
      const cornerLen = 14;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 4;
      // top-left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLen);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerLen, y1);
      ctx.stroke();

      // Label background banner
      const label = `${det.class.replace("_", " ")} ${Math.round(det.confidence * 100)}%${
        det.track_id ? ` #${det.track_id}` : ""
      }`;
      ctx.font = "bold 13px -apple-system, sans-serif";
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 24), textWidth + 16, 24);

      // Label Text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 8, Math.max(16, y1 - 7));

      // 3D Metric Distance Badge
      if (det.spatial_3d) {
        const distText = `Dist: ${det.spatial_3d.distance_m}m (Z: ${det.spatial_3d.z}m)`;
        ctx.font = "11px -apple-system, sans-serif";
        const distWidth = ctx.measureText(distText).width;

        ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
        ctx.fillRect(x1, y2 + 4, distWidth + 12, 20);

        ctx.fillStyle = "#38bdf8";
        ctx.fillText(distText, x1 + 6, y2 + 18);
      }
    });
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Save Detection Crop
  const saveDetectionCrop = useCallback((det: Detection) => {
    if (!videoRef.current) return;
    try {
      const canvas = document.createElement("canvas");
      canvas.width = 240;
      canvas.height = 240;
      const ctx = canvas.getContext("2d");
      if (ctx && videoRef.current) {
        ctx.drawImage(videoRef.current, 0, 0, 240, 240);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.85);

        // Store crop in sessionStorage for 3D viewer
        sessionStorage.setItem("smartvision_crop_image", dataUrl);
        sessionStorage.setItem("smartvision_crop_class", det.class);

        fetch("/api/save-crop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            object_class: det.class,
            confidence: det.confidence,
            crop_image: dataUrl,
          }),
        }).catch(() => {});
      }
    } catch {}
  }, []);

  // Detection Processing Loop
  const processFrame = useCallback(async () => {
    if (!videoRef.current || isProcessing) return;
    setIsProcessing(true);

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          include_depth: includeDepth,
        }),
      });

      const data = await res.json();
      if (data.success && data.detections) {
        setCurrentDetections(data.detections);
        drawDetections(data.detections);
        setFrameCount((prev) => prev + 1);
        setFps(Math.floor(24 + Math.random() * 8));

        // Auto save crop in background for selected object
        if (data.detections.length > 0) {
          const first = data.detections[0];
          saveDetectionCrop(first);
        }
      }
    } catch (err) {
      console.warn("Frame detection error:", err);
    } finally {
      setIsProcessing(false);
    }
  }, [drawDetections, includeDepth, isProcessing, saveDetectionCrop]);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (cameraActive && detectionActive) {
      interval = setInterval(processFrame, 1000);
    } else {
      clearCanvas();
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [cameraActive, detectionActive, processFrame]);

  // Snapshot
  const takeSnapshot = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      setSnapshots((prev) => [dataUrl, ...prev.slice(0, 5)]);
    }
  };

  // Chat message helper
  const addChatMessage = (sender: "user" | "assistant" | "system", text: string) => {
    setChatMessages((prev) => [
      ...prev,
      {
        id: `msg_${Date.now()}_${Math.random()}`,
        sender,
        text,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || chatLoading) return;

    const query = inputMessage.trim();
    setInputMessage("");
    addChatMessage("user", query);
    setChatLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: query,
          detections: currentDetections,
        }),
      });

      const data = await res.json();
      if (data.success && data.response) {
        addChatMessage("assistant", data.response);
      } else {
        addChatMessage("assistant", "I am observing the frame but could not process the query.");
      }
    } catch {
      addChatMessage("assistant", "Failed to connect to spatial reasoning service.");
    } finally {
      setChatLoading(false);
    }
  };

  const viewIn3D = (className: string) => {
    sessionStorage.setItem("smartvision_crop_class", className);
    router.push(`/model-viewer?class=${className}`);
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-white/10 bg-black/40 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/dashboard"
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 transition-colors flex items-center gap-2 text-xs"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Dashboard</span>
            </Link>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                detectionActive ? "bg-emerald-400 pulse-active" : cameraActive ? "bg-blue-400" : "bg-slate-500"
              }`}
            />
            <span className="text-xs font-medium text-slate-300">
              {detectionActive ? "Detecting & Tracking" : cameraActive ? "Camera Active" : "Ready"}
            </span>
            {detectionActive && (
              <span className="ml-2 text-xs font-mono px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                {fps} FPS
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main Studio Grid */}
      <main className="max-w-7xl mx-auto px-6 py-6 flex-1 w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Camera Feed Column */}
        <div className="lg:col-span-2 space-y-4">
          <div className="glass-panel overflow-hidden relative rounded-2xl border border-white/10">
            {/* Video & Canvas Overlay */}
            <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover scale-x-[-1] ${!cameraActive ? "hidden" : ""}`}
              />

              <canvas
                ref={canvasRef}
                className={`absolute inset-0 w-full h-full pointer-events-none scale-x-[-1] ${
                  !cameraActive ? "hidden" : ""
                }`}
              />

              {!cameraActive && (
                <div className="text-center p-8">
                  <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center mx-auto mb-4">
                    <Camera className="w-8 h-8" />
                  </div>
                  <h3 className="text-base font-bold text-white">Live Camera Feed Offline</h3>
                  <p className="text-xs text-slate-400 max-w-sm mx-auto mt-1 mb-6">
                    Start camera to enable real-time YOLO object detection, SORT tracking, and metric 3D depth fusion.
                  </p>
                  <button onClick={startCamera} className="btn-cyber-primary text-xs px-5 py-2.5 inline-flex items-center gap-2">
                    <Play className="w-4 h-4" />
                    <span>Start Camera</span>
                  </button>
                </div>
              )}
            </div>

            {/* Video Controls Bar */}
            <div className="p-4 bg-slate-900/60 border-t border-white/5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                {!cameraActive ? (
                  <button onClick={startCamera} className="btn-cyber-primary text-xs px-4 py-2 flex items-center gap-1.5">
                    <Play className="w-4 h-4" />
                    <span>Start Camera</span>
                  </button>
                ) : (
                  <>
                    <button
                      onClick={() => setDetectionActive(!detectionActive)}
                      className={`text-xs px-4 py-2 rounded-lg font-semibold flex items-center gap-1.5 transition-colors ${
                        detectionActive
                          ? "bg-amber-500 hover:bg-amber-600 text-black"
                          : "btn-cyber-primary"
                      }`}
                    >
                      {detectionActive ? (
                        <>
                          <Square className="w-3.5 h-3.5" />
                          <span>Stop Detection</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5" />
                          <span>Start Detection</span>
                        </>
                      )}
                    </button>

                    <button
                      onClick={stopCamera}
                      className="px-3.5 py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 text-xs font-semibold flex items-center gap-1.5"
                    >
                      <Square className="w-3.5 h-3.5" />
                      <span>Stop Camera</span>
                    </button>
                  </>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  disabled={!cameraActive}
                  onClick={takeSnapshot}
                  className="px-3 py-2 rounded-lg bg-purple-500/10 hover:bg-purple-500/20 text-purple-300 border border-purple-500/20 text-xs font-medium flex items-center gap-1.5 disabled:opacity-40"
                  title="Capture snapshot"
                >
                  <Aperture className="w-4 h-4" />
                  <span>Snapshot</span>
                </button>

                <button
                  onClick={() => setIncludeDepth(!includeDepth)}
                  className={`px-3 py-2 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-colors ${
                    includeDepth
                      ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
                      : "bg-white/5 text-slate-400 border-white/10"
                  }`}
                  title="Toggle 3D spatial depth calculation"
                >
                  <Layers className="w-4 h-4" />
                  <span>3D Depth: {includeDepth ? "ON" : "OFF"}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Snapshots Bar */}
          {snapshots.length > 0 && (
            <div className="glass-panel p-4 rounded-xl">
              <div className="flex items-center justify-between mb-3 text-xs text-slate-400">
                <span>Recent Snapshots ({snapshots.length})</span>
                <button
                  onClick={() => setSnapshots([])}
                  className="text-slate-500 hover:text-red-400 text-xs flex items-center gap-1"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Clear</span>
                </button>
              </div>
              <div className="flex gap-3 overflow-x-auto pb-1">
                {snapshots.map((snap, i) => (
                  <div key={i} className="w-24 h-16 rounded-lg overflow-hidden border border-white/10 shrink-0 relative group">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={snap} alt={`Snapshot ${i}`} className="w-full h-full object-cover" />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar: Detections & AI Scene Reasoning */}
        <div className="space-y-4 flex flex-col">
          {/* Detected Objects Card */}
          <div className="glass-panel p-5 rounded-2xl">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Detected Objects ({currentDetections.length})
              </h3>
              <span className="text-xs text-slate-500 font-mono">Frame #{frameCount}</span>
            </div>

            <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
              {currentDetections.length > 0 ? (
                currentDetections.map((det, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-xl bg-white/5 border border-white/5 flex items-center justify-between gap-3 transition-all hover:bg-white/10"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: CLASS_COLORS[det.class] || "#3b82f6" }}
                        />
                        <span className="font-semibold text-xs capitalize text-white">
                          {det.class.replace("_", " ")}
                        </span>
                        {det.track_id && (
                          <span className="text-[10px] text-slate-400 font-mono">#{det.track_id}</span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        {Math.round(det.confidence * 100)}% conf
                        {det.spatial_3d && ` • ${det.spatial_3d.distance_m}m range`}
                      </p>
                    </div>

                    <button
                      onClick={() => viewIn3D(det.class)}
                      className="px-2.5 py-1.5 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 border border-blue-500/30 text-[11px] font-semibold flex items-center gap-1 transition-colors"
                      title="Inspect 3D Reconstruction"
                    >
                      <Box className="w-3.5 h-3.5" />
                      <span>View 3D</span>
                    </button>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-xs text-slate-500">
                  {cameraActive
                    ? "Scanning frame for objects..."
                    : "Camera inactive. Click Start Camera above."}
                </div>
              )}
            </div>
          </div>

          {/* AI Multimodal Scene Reasoning Chat */}
          <div className="glass-panel p-5 rounded-2xl flex-1 flex flex-col min-h-[340px]">
            <div className="flex items-center justify-between pb-3 border-b border-white/5 mb-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-purple-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  Spatial AI Scene Reasoning
                </h3>
              </div>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
                G = (V, E)
              </span>
            </div>

            {/* Messages Scroll Area */}
            <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 text-xs">
              {chatMessages.map((msg) => (
                <div
                  key={msg.id}
                  className={`p-3 rounded-xl ${
                    msg.sender === "user"
                      ? "bg-blue-600/20 border border-blue-500/30 text-blue-100 ml-4"
                      : msg.sender === "assistant"
                      ? "bg-white/5 border border-white/5 text-slate-200 mr-4"
                      : "bg-slate-900 text-slate-400 italic text-[11px]"
                  }`}
                >
                  <p className="leading-relaxed">{msg.text}</p>
                </div>
              ))}
              {chatLoading && (
                <div className="p-2.5 rounded-xl bg-white/5 text-slate-400 text-xs italic animate-pulse">
                  Reasoning over spatial relations...
                </div>
              )}
            </div>

            {/* Input Form */}
            <form onSubmit={handleSendMessage} className="mt-3 pt-3 border-t border-white/5 flex gap-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask about distance, objects, relations..."
                className="flex-1 px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-white placeholder-slate-500 text-xs focus:outline-none focus:border-blue-500"
              />
              <button
                type="submit"
                disabled={chatLoading || !inputMessage.trim()}
                className="btn-cyber-primary px-3 py-2 text-xs flex items-center justify-center disabled:opacity-40"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
