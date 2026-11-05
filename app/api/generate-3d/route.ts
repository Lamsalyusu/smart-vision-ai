import { NextRequest, NextResponse } from "next/server";
import { depthEstimator } from "@/lib/depth-estimation";

const MAX_IMAGE_DATA_URL_LENGTH = 2 * 1024 * 1024;

function visionEndpoint(path: string): URL | null {
  const detectUrl =
    process.env.VISION_API_URL?.trim() ||
    (process.env.NODE_ENV === "development" ? "http://localhost:8001/api/detect/" : "");
  if (!detectUrl) return null;
  try {
    return new URL(path, detectUrl);
  } catch {
    return null;
  }
}

/**
 * Point cloud for the 3D viewer.
 *
 * With `image_base64` (a saved detection crop) the vision service's MiDaS model
 * produces a real depth-based point cloud (`source: "midas"`). Without an image
 * a synthetic demo surface is returned (`source: "demo"`), so the viewer always
 * knows which one it is showing.
 */
export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const depthScale = typeof body.depth_scale === "number" ? body.depth_scale : 2.0;
  const maxPoints = typeof body.max_points === "number" ? body.max_points : 10000;
  const image = typeof body.image_base64 === "string" ? body.image_base64 : "";

  if (!image) {
    const result = depthEstimator.generatePointCloud(160, 160, depthScale, maxPoints);
    return NextResponse.json({ ...result, source: "demo" });
  }

  if (!/^data:image\/(jpeg|png|webp);base64,/.test(image) || image.length > MAX_IMAGE_DATA_URL_LENGTH) {
    return NextResponse.json({ error: "image_base64 must be an image data URL under 2 MB" }, { status: 400 });
  }

  const upstreamUrl = visionEndpoint("/api/generate-3d/");
  if (!upstreamUrl) {
    return NextResponse.json({ error: "Vision service is not configured" }, { status: 503 });
  }

  try {
    const upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(process.env.INTERNAL_API_TOKEN ? { "X-Internal-Token": process.env.INTERNAL_API_TOKEN } : {}),
      },
      body: JSON.stringify({ image_base64: image, depth_scale: depthScale, max_points: maxPoints }),
      signal: AbortSignal.timeout(30_000),
      cache: "no-store",
    });
    const data: Record<string, unknown> | null = await upstream.json().catch(() => null);
    if (!upstream.ok || !data || data.success !== true) {
      const error = typeof data?.error === "string" ? data.error : "Depth point cloud failed";
      return NextResponse.json({ error }, { status: upstream.status === 503 ? 503 : 502 });
    }
    return NextResponse.json({
      success: true,
      source: "midas",
      positions: data.positions,
      colors: data.colors,
      point_count: data.point_count,
      width: data.width,
      height: data.height,
    });
  } catch {
    return NextResponse.json({ error: "Vision service is unavailable" }, { status: 503 });
  }
}
