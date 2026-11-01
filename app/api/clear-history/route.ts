import { NextResponse } from "next/server";
import { clearHistory } from "@/lib/apiClient";

export async function POST() {
  try {
    return NextResponse.json(await clearHistory());
  } catch {
    return NextResponse.json(
      { error: "Telemetry service is unavailable" },
      { status: 503 }
    );
  }
}
