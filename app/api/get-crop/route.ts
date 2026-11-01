import { NextRequest, NextResponse } from "next/server";
import { getLatestCrop } from "@/lib/apiClient";

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const objectClass = searchParams.get("class") || undefined;

  let result: Awaited<ReturnType<typeof getLatestCrop>>;
  try {
    result = await getLatestCrop(objectClass);
  } catch {
    return NextResponse.json(
      { error: "Crop service is unavailable" },
      { status: 503 }
    );
  }
  const crop = result.crop;
  if (!crop) {
    return NextResponse.json({ success: false, message: "No crop available" }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    crop: {
      object_class: crop.object_class,
      confidence: crop.confidence,
      crop_image: crop.payload?.crop_image ?? "",
      timestamp: crop.timestamp,
    },
  });
}
