import { NextRequest, NextResponse } from "next/server";
import { internalRequest } from "@/lib/apiClient";
import { CLASS_NAMES } from "@/lib/detector";

const MAX_POINTS = 50_000;

interface MeshResponse {
  threejs_geometry: unknown;
  statistics: unknown;
}

export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const rawClass = String(body.class_name ?? "").toLowerCase().trim();
  const className = (CLASS_NAMES as string[]).includes(rawClass) ? rawClass : "chair";
  const points = Array.isArray(body.points) ? body.points : undefined;
  if (points && points.length > MAX_POINTS) {
    return NextResponse.json({ error: `At most ${MAX_POINTS} points are accepted` }, { status: 413 });
  }

  try {
    const mesh = await internalRequest<MeshResponse>("/mesh/reconstruct", "Mesh reconstruction failed", {
      method: "POST",
      body: {
        object_class: className,
        points,
        voxel_size: typeof body.voxel_size === "number" ? body.voxel_size : 0.08,
        smoothing_iterations:
          typeof body.smoothing_iterations === "number" ? body.smoothing_iterations : 2,
        output_format: "threejs",
      },
      timeoutMs: 10_000,
    });
    return NextResponse.json({
      success: true,
      class_name: className,
      format: "threejs_buffer_geometry",
      glb_url: `/models/${className}.glb`,
      threejs_geometry: mesh.threejs_geometry,
      statistics: mesh.statistics,
    });
  } catch (error) {
    // Reconstruction is optional: fall back to the static GLB asset for the class.
    console.warn("Mesh reconstruction unavailable, serving static GLB:", error);
    return NextResponse.json({
      success: true,
      glb_url: `/models/${className}.glb`,
      format: "glb",
      class_name: className,
      statistics: { method: "Static GLB fallback" },
    });
  }
}
