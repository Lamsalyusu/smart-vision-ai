import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

/**
 * GET /api/auth/session – returns authentication status based on the session cookie.
 * If the cookie is missing or the token is invalid/expired, the cookie is cleared
 * to avoid stale client state.
 */
export async function GET(req: NextRequest) {
  const token = req.cookies.get("session_token")?.value;
  if (!token) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  const user = dataManager.getUserFromSession(token);
  if (!user) {
    // Clear stale/invalid cookie
    const res = NextResponse.json({ authenticated: false, user: null });
    res.cookies.set("session_token", "", { path: "/", httpOnly: true, sameSite: "lax", maxAge: 0 });
    return res;
  }

  const res = NextResponse.json({
    authenticated: true,
    user: {
      id: user.id,
      full_name: user.full_name,
      email: user.email,
      username: user.username,
      role: user.role || "operator",
    },
  });
  res.headers.set('Content-Security-Policy', "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline';");
  return res;
}
