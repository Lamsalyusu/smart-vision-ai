import { NextRequest, NextResponse } from "next/server";
import { confirmEmailVerification } from "@/lib/apiClient";
import { clientIp, upstreamMessage, upstreamStatus } from "@/lib/session";

export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const token = String(body.token ?? "");
  if (!token) {
    return NextResponse.json({ error: "Verification token is required" }, { status: 400 });
  }
  try {
    await confirmEmailVerification(token, clientIp(req));
    return NextResponse.json({ success: true });
  } catch (error) {
    const status = upstreamStatus(error) === 422 ? 400 : upstreamStatus(error);
    return NextResponse.json({ error: upstreamMessage(error, "Could not verify the email address") }, { status });
  }
}
