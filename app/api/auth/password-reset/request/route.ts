import { NextRequest, NextResponse } from "next/server";
import { requestPasswordReset } from "@/lib/apiClient";
import { clientIp, upstreamMessage, upstreamStatus } from "@/lib/session";

export async function POST(req: NextRequest) {
  const body: unknown = await req.json().catch(() => null);
  const email =
    typeof body === "object" && body !== null && "email" in body ? String(body.email ?? "").trim() : "";
  if (!email) {
    return NextResponse.json({ error: "Email is required" }, { status: 400 });
  }
  try {
    await requestPasswordReset(email, clientIp(req));
    return NextResponse.json({ success: true });
  } catch (error) {
    const status = upstreamStatus(error) === 422 ? 400 : upstreamStatus(error);
    return NextResponse.json({ error: upstreamMessage(error, "Could not request a reset") }, { status });
  }
}
