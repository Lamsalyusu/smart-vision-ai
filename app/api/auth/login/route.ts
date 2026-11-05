import { NextRequest, NextResponse } from "next/server";
import { login } from "@/lib/apiClient";
import { clientIp, setSessionCookie, upstreamMessage, upstreamStatus } from "@/lib/session";

/**
 * Authenticates against the FastAPI service, which owns users, sessions and
 * the failed-login lockout, then stores its session token in an httpOnly cookie.
 */
export async function POST(req: NextRequest) {
  const body: unknown = await req.json().catch(() => null);
  const identifier =
    typeof body === "object" && body !== null && "email_or_username" in body
      ? String(body.email_or_username ?? "").trim()
      : "";
  const password =
    typeof body === "object" && body !== null && "password" in body ? String(body.password ?? "") : "";
  if (!identifier || !password) {
    return NextResponse.json({ error: "Please provide email/username and password" }, { status: 400 });
  }

  try {
    const { user, token } = await login(identifier, password, clientIp(req));
    const res = NextResponse.json({ success: true, user });
    setSessionCookie(res, token);
    return res;
  } catch (error) {
    const status = upstreamStatus(error);
    if (status >= 500) console.error("Login failed:", error);
    return NextResponse.json({ error: upstreamMessage(error, "Login failed") }, { status });
  }
}
