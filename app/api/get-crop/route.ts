import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const objectClass = searchParams.get("class") || undefined;

  const crop = dataManager.getLatestCrop(objectClass);
  if (!crop) {
    return NextResponse.json({ success: false, message: "No crop available" }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    crop: {
      object_class: crop.object_class,
      confidence: crop.confidence,
      crop_image: crop.crop_image,
      timestamp: crop.timestamp,
    },
  });
}
