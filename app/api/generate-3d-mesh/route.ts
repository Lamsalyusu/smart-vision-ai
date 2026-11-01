import { NextRequest, NextResponse } from "next/server";

const VALID_CLASSES = [
  "sunglasses",
  "knife",
  "water_bottle",
  "pen",
  "chair",
  "human_face",
  "mobile_phone",
  "helmet",
  "fire",
  "can",
];

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://127.0.0.1:8000";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const rawClass = (body.class_name || "").toLowerCase().trim();
    const className = VALID_CLASSES.includes(rawClass) ? rawClass : "chair";
    const customPoints = Array.isArray(body.points) ? body.points : undefined;

    // Attempt high-fidelity Marching Cubes reconstruction from Python service
    try {
      const response = await fetch(`${FASTAPI_BASE_URL}/mesh/reconstruct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          object_class: className,
          points: customPoints,
          voxel_size: body.voxel_size || 0.08,
          smoothing_iterations: body.smoothing_iterations || 2,
          output_format: "threejs",
        }),
        signal: AbortSignal.timeout(4000),
      });

      if (response.ok) {
        const meshData = await response.json();
        return NextResponse.json({
          success: true,
          class_name: className,
          format: "threejs_buffer_geometry",
          glb_url: `/models/${className}.glb`,
          threejs_geometry: meshData.threejs_geometry,
          statistics: meshData.statistics,
        });
      }
    } catch {
      // Backend service not reachable or timed out; fall back to static GLB asset
    }

    return NextResponse.json({
      success: true,
      glb_url: `/models/${className}.glb`,
      format: "glb",
      class_name: className,
      statistics: {
        method: "Static GLB Fallback",
      },
    });
  } catch (err: unknown) {
    console.error("Generate 3D mesh error:", err);
    return NextResponse.json(
      { error: "Failed to generate 3D mesh" },
      { status: 500 }
    );
  }
}
