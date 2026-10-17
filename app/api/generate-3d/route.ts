import { NextRequest, NextResponse } from "next/server";
import { depthEstimator } from "@/lib/depth-estimation";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const depthScale = typeof body.depth_scale === "number" ? body.depth_scale : 2.0;
    const maxPoints = typeof body.max_points === "number" ? body.max_points : 10000;

    const result = depthEstimator.generatePointCloud(160, 160, depthScale, maxPoints);

    return NextResponse.json({
      success: true,
      positions: result.positions,
      colors: result.colors,
      point_count: result.point_count,
      width: result.width,
      height: result.height,
    });
  } catch (err: unknown) {
    console.error("Generate 3D point cloud error:", err);
    return NextResponse.json(
      { error: "Failed to generate point cloud" },
      { status: 500 }
    );
  }
}
