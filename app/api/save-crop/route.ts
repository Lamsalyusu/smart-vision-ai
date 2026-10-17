import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { object_class, confidence, crop_image } = body;

    if (!object_class || !crop_image) {
      return NextResponse.json(
        { error: "object_class and crop_image are required" },
        { status: 400 }
      );
    }

    const saved = dataManager.saveCrop({
      object_class,
      confidence: Number(confidence) || 0.85,
      crop_image,
    });

    return NextResponse.json({ success: true, crop: saved });
  } catch (err: unknown) {
    console.error("Save crop error:", err);
    return NextResponse.json({ error: "Failed to save crop" }, { status: 500 });
  }
}
