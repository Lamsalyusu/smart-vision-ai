import { NextRequest, NextResponse } from "next/server";
import { logout } from "@/lib/apiClient";
import { dataManager } from "@/lib/store";
import { SecurityAuditLog } from "@/lib/auth-crypto";

/**
 * Logout route – clears the local session, records an audit entry, and proxies to the FastAPI logout endpoint.
 */
export async function POST(req: NextRequest) {
  // Revoke local session and log the event
  const token = req.cookies.get("session_token")?.value;
  if (token) {
    const user = dataManager.getUserFromSession(token);
    dataManager.revokeSession(token);
    if (user) {
      SecurityAuditLog.record({
        event_type: "LOGOUT",
        user_id: user.id,
        identifier: user.username,
      });
    }
  }

  // Forward logout to FastAPI – ignore errors but log them
  try {
    await logout();
  } catch (err) {
    console.error("FastAPI logout error:", err);
  }

  const res = NextResponse.json({ success: true });
  res.headers.set(
    "Content-Security-Policy",
    "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline';"
  );
  // Clear client‑side cookie
  res.cookies.set("session_token", "", {
    path: "/",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 0,
  });
  return res;
}
