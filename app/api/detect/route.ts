import { NextRequest, NextResponse } from "next/server";
import { detector } from "@/lib/detector";
import { SortTracker } from "@/lib/tracker";
import { spatialFuser } from "@/lib/spatial-fusion";
import { dataManager } from "@/lib/store";

// Session-level SORT trackers
const sessionTrackers = new Map<string, SortTracker>();

function getTracker(sessionKey: string): SortTracker {
  let tracker = sessionTrackers.get(sessionKey);
  if (!tracker) {
    tracker = new SortTracker(15, 2, 0.3);
    sessionTrackers.set(sessionKey, tracker);
  }
  return tracker;
}

export async function POST(req: NextRequest) {
  try {
    const contentType = req.headers.get("content-type") || "";
    let includeDepth = true;
    let simulatedClasses: string[] | undefined = undefined;

    if (contentType.includes("multipart/form-data")) {
      const formData = await req.formData();
      includeDepth = formData.get("include_depth")?.toString().toLowerCase() !== "false";
    } else if (contentType.includes("application/json")) {
      const body = await req.json().catch(() => ({}));
      if (body.include_depth !== undefined) {
        includeDepth = Boolean(body.include_depth);
      }
      if (body.classes && Array.isArray(body.classes)) {
        simulatedClasses = body.classes;
      }
    }

    const sessionKey = req.cookies.get("session_token")?.value || "default_session";
    const tracker = getTracker(sessionKey);

    // 1. Raw Detections
    const rawDetections = detector.detect(640, 480, simulatedClasses);

    // 2. Multi-Object Tracking
    const trackedDetections = tracker.update(rawDetections);
    let detections = trackedDetections.length > 0 ? trackedDetections : rawDetections;

    // 3. 2D-to-3D Spatial Metric Fusion
    if (includeDepth && detections.length > 0) {
      detections = spatialFuser.estimateSpatialMetrics(detections, 640, 480);
    }

    // 4. Record to telemetry store
    detections.forEach((d) => {
      dataManager.recordDetection(d.class, d.confidence, d.spatial_3d?.distance_m);
    });

    return NextResponse.json({
      success: true,
      total_detections: detections.length,
      detections,
    });
  } catch (err: unknown) {
    console.error("Detect API error:", err);
    return NextResponse.json(
      { error: "Detection pipeline failed" },
      { status: 500 }
    );
  }
}
