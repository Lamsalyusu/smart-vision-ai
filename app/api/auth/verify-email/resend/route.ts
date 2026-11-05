import { NextRequest, NextResponse } from "next/server";
import { resendEmailVerification } from "@/lib/apiClient";
import { clientIp, upstreamMessage, upstreamStatus } from "@/lib/session";

export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const email = String(body.email ?? "").trim();
  if (!email) {
    return NextResponse.json({ error: "Email is required" }, { status: 400 });
  }
  try {
    await resendEmailVerification(email, clientIp(req));
    return NextResponse.json({ success: true });
  } catch (error) {
    const status = upstreamStatus(error) === 422 ? 400 : upstreamStatus(error);
    return NextResponse.json({ error: upstreamMessage(error, "Could not send a verification email") }, { status });
  }
}
