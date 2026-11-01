import { NextRequest, NextResponse } from "next/server";
import { saveCrop } from "@/lib/apiClient";

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

    const saved = await saveCrop({
      object_class: String(object_class),
      confidence: Number(confidence),
      has_crop: true,
      payload: { crop_image },
    });

    return NextResponse.json({ success: true, crop: saved });
  } catch (err: unknown) {
    console.error("Save crop error:", err);
    return NextResponse.json({ error: "Failed to save crop" }, { status: 503 });
  }
}
