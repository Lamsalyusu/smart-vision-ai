import { NextResponse } from "next/server";
import { getStats } from "@/lib/apiClient";

export async function GET() {
  try {
    const stats = await getStats();
    return NextResponse.json({ success: true, stats });
  } catch {
    return NextResponse.json(
      { success: false, error: "Telemetry service is unavailable" },
      { status: 503 }
    );
  }
}
