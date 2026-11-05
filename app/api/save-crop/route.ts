import { NextRequest, NextResponse } from "next/server";
import { saveCrop } from "@/lib/apiClient";
import { CLASS_NAMES } from "@/lib/detector";
import { upstreamMessage, upstreamStatus } from "@/lib/session";

// A 240px JPEG data URL is well under this; the cap stops arbitrary blobs being stored.
const MAX_CROP_DATA_URL_LENGTH = 512 * 1024;

export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const objectClass = String(body.object_class ?? "");
  const confidence = Number(body.confidence);
  const cropImage = body.crop_image;

  if (!(CLASS_NAMES as string[]).includes(objectClass)) {
    return NextResponse.json({ error: "Unknown object_class" }, { status: 400 });
  }
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    return NextResponse.json({ error: "confidence must be between 0 and 1" }, { status: 400 });
  }
  if (typeof cropImage !== "string" || !/^data:image\/(jpeg|png|webp);base64,/.test(cropImage)) {
    return NextResponse.json({ error: "crop_image must be an image data URL" }, { status: 400 });
  }
  if (cropImage.length > MAX_CROP_DATA_URL_LENGTH) {
    return NextResponse.json({ error: "crop_image is too large" }, { status: 413 });
  }

  try {
    const saved = await saveCrop({
      object_class: objectClass,
      confidence,
      has_crop: true,
      payload: { crop_image: cropImage },
    });
    return NextResponse.json({ success: true, crop: saved.crop });
  } catch (error) {
    console.error("Save crop error:", error);
    return NextResponse.json(
      { error: upstreamMessage(error, "Failed to save crop") },
      { status: upstreamStatus(error) }
    );
  }
}
