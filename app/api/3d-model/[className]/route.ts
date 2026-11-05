import { NextRequest, NextResponse } from "next/server";
import { CLASS_NAMES } from "@/lib/detector";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ className: string }> }
) {
  const { className } = await params;
  const cleanName = (className || "").toLowerCase();
  if (!(CLASS_NAMES as string[]).includes(cleanName)) {
    return NextResponse.json({ error: "Unknown object class" }, { status: 404 });
  }

  return NextResponse.json({
    class: cleanName,
    model_url: `/models/${cleanName}.glb`,
    format: "glb",
  });
}
