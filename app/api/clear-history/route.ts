import { NextRequest, NextResponse } from "next/server";
import { clearHistory } from "@/lib/apiClient";
import { upstreamMessage, upstreamStatus, USER_ROLE_HEADER } from "@/lib/session";

/** Deletes all recorded detection telemetry. Admin only: it affects every user. */
export async function POST(req: NextRequest) {
  if (req.headers.get(USER_ROLE_HEADER) !== "admin") {
    return NextResponse.json({ error: "Only administrators can clear detection history" }, { status: 403 });
  }
  try {
    return NextResponse.json(await clearHistory());
  } catch (error) {
    return NextResponse.json(
      { error: upstreamMessage(error, "Telemetry service is unavailable") },
      { status: upstreamStatus(error) }
    );
  }
}
