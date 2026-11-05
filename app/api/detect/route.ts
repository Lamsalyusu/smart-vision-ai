import { randomUUID } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { recordDetections } from "@/lib/apiClient";

const MAX_FRAME_SIZE = 5 * 1024 * 1024;
const SESSION_COOKIE = "vision_session_id";
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

function isSessionId(value: string | undefined): value is string {
  return value !== undefined && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

export async function POST(request: NextRequest) {
  const existingSessionId = request.cookies.get(SESSION_COOKIE)?.value;
  const sessionId = isSessionId(existingSessionId) ? existingSessionId : randomUUID();
  const shouldSetCookie = sessionId !== existingSessionId;
  const respond = (body: unknown, status: number) => {
    const response = NextResponse.json(body, { status });
    if (shouldSetCookie) {
      response.cookies.set(SESSION_COOKIE, sessionId, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        path: "/",
        maxAge: 60 * 60 * 24 * 30,
      });
    }
    return response;
  };

  const configuredUrl = process.env.VISION_API_URL?.trim() ||
    (process.env.NODE_ENV === "development" ? "http://localhost:8001/api/detect/" : "");
  if (!configuredUrl) {
    return respond({ error: "Vision inference service is not configured" }, 503);
  }

  let upstreamUrl: URL;
  try {
    upstreamUrl = new URL(configuredUrl);
    if (!["http:", "https:"].includes(upstreamUrl.protocol)) {
      throw new Error("unsupported protocol");
    }
  } catch {
    return respond({ error: "Vision inference service URL is invalid" }, 503);
  }

  if (!request.headers.get("content-type")?.includes("multipart/form-data")) {
    return respond({ error: "Upload a frame as multipart form data" }, 415);
  }
  const contentLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_FRAME_SIZE + 64 * 1024) {
    return respond({ error: "Frame exceeds the 5 MB limit" }, 413);
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return respond({ error: "Frame upload could not be read" }, 400);
  }

  const image = formData.get("image");
  if (!(image instanceof File) || image.size === 0) {
    return respond({ error: "A non-empty image frame is required" }, 400);
  }
  if (!ALLOWED_IMAGE_TYPES.has(image.type)) {
    return respond({ error: "Frame must be JPEG, PNG, or WebP" }, 415);
  }
  if (image.size > MAX_FRAME_SIZE) {
    return respond({ error: "Frame exceeds the 5 MB limit" }, 413);
  }

  const upstreamForm = new FormData();
  upstreamForm.append("image", image, image.name || "camera-frame.jpg");
  upstreamForm.append("include_depth", formData.get("include_depth")?.toString() ?? "true");

  try {
    const startedAt = performance.now();
    const upstream = await fetch(upstreamUrl, {
      method: "POST",
      headers: {
        "X-Tracking-Session": sessionId,
        ...(process.env.INTERNAL_API_TOKEN ? { "X-Internal-Token": process.env.INTERNAL_API_TOKEN } : {}),
      },
      body: upstreamForm,
      signal: AbortSignal.timeout(15_000),
      cache: "no-store",
    });
    const payload: unknown = await upstream.json().catch(() => null);
    const roundtripMs = Math.round((performance.now() - startedAt) * 100) / 100;
    if (!upstream.ok) {
      const error =
        typeof payload === "object" && payload !== null && "error" in payload &&
        typeof payload.error === "string"
          ? payload.error
          : "Vision inference request failed";
      return respond({ error }, upstream.status >= 500 ? 502 : upstream.status);
    }
    if (
      typeof payload !== "object" ||
      payload === null ||
      !("success" in payload) ||
      payload.success !== true ||
      !("detections" in payload) ||
      !Array.isArray(payload.detections)
    ) {
      return respond({ error: "Vision inference returned an invalid response" }, 502);
    }
    const detections = payload.detections;
    if (!detections.every(isDetection)) {
      return respond({ error: "Vision inference returned malformed detections" }, 502);
    }

    let telemetryPersisted = true;
    try {
      await recordDetections(
        detections.map((detection) => ({
          class_name: detection.class,
          confidence: detection.confidence,
          distance_m: detection.spatial_3d?.distance_m,
          has_crop: false,
        })),
        { ...readVisionTimings(payload), roundtrip_ms: roundtripMs }
      );
    } catch (error) {
      telemetryPersisted = false;
      console.error("Could not persist inference telemetry:", error);
    }
    return respond({ ...payload, telemetry_persisted: telemetryPersisted }, 200);
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      return respond({ error: "Vision inference timed out" }, 504);
    }
    return respond({ error: "Vision inference service is unavailable" }, 503);
  }
}

/** Timings measured by the vision service, if it reported valid ones. */
function readVisionTimings(payload: object): { inference_ms?: number; total_ms?: number } {
  const timing = "timing_ms" in payload ? payload.timing_ms : undefined;
  if (typeof timing !== "object" || timing === null) return {};
  const result: { inference_ms?: number; total_ms?: number } = {};
  for (const key of ["inference_ms", "total_ms"] as const) {
    const value = (timing as Record<string, unknown>)[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) result[key] = value;
  }
  return result;
}

function isDetection(value: unknown): value is {
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  spatial_3d?: { distance_m?: number | null } | null;
} {
  if (
    typeof value !== "object" ||
    value === null ||
    !("class" in value) ||
    !("confidence" in value) ||
    !("bbox" in value) ||
    !Array.isArray(value.bbox) ||
    value.bbox.length !== 4 ||
    !value.bbox.every((coordinate) => typeof coordinate === "number" && Number.isFinite(coordinate))
  ) {
    return false;
  }
  return (
    typeof value.class === "string" &&
    typeof value.confidence === "number" &&
    Number.isFinite(value.confidence) &&
    value.confidence >= 0 &&
    value.confidence <= 1 &&
    (!("spatial_3d" in value) ||
      value.spatial_3d === null ||
      (typeof value.spatial_3d === "object" &&
        value.spatial_3d !== null &&
        (!("distance_m" in value.spatial_3d) ||
          value.spatial_3d.distance_m === null ||
          (typeof value.spatial_3d.distance_m === "number" &&
            Number.isFinite(value.spatial_3d.distance_m)))))
  );
}