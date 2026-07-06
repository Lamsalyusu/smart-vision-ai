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

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    const rawClass = (body.class_name || "").toLowerCase().trim();
    const className = VALID_CLASSES.includes(rawClass) ? rawClass : "chair";

    return NextResponse.json({
      success: true,
      glb_url: `/models/${className}.glb`,
      format: "glb",
      class_name: className,
    });
  } catch (err: unknown) {
    console.error("Generate 3D mesh error:", err);
    return NextResponse.json(
      { error: "Failed to generate 3D mesh" },
      { status: 500 }
    );
  }
}
