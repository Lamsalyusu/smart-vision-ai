import { NextRequest, NextResponse } from "next/server";
import { confirmPasswordReset } from "@/lib/apiClient";
import { clearSessionCookie, upstreamMessage, upstreamStatus } from "@/lib/session";

export async function POST(req: NextRequest) {
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const token = String(body.token ?? "");
  const newPassword = String(body.new_password ?? "");
  if (!token || !newPassword) {
    return NextResponse.json({ error: "Token and new password are required" }, { status: 400 });
  }
  try {
    await confirmPasswordReset(token, newPassword);
    // All sessions were revoked upstream; drop this browser's cookie too.
    const res = NextResponse.json({ success: true });
    clearSessionCookie(res);
    return res;
  } catch (error) {
    const status = upstreamStatus(error) === 422 ? 400 : upstreamStatus(error);
    return NextResponse.json({ error: upstreamMessage(error, "Could not reset the password") }, { status });
  }
}
