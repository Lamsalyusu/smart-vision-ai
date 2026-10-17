import { NextRequest, NextResponse } from "next/server";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ className: string }> }
) {
  const { className } = await params;
  const cleanName = (className || "").toLowerCase();

  return NextResponse.json({
    class: cleanName,
    model_url: `/models/${cleanName}.glb`,
    format: "glb",
  });
}
