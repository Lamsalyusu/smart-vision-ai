import { NextRequest, NextResponse } from "next/server";
import { logout } from "@/lib/apiClient";
import { clearSessionCookie, readSessionToken } from "@/lib/session";

/** Revokes the FastAPI session and clears the cookie, even if revocation fails. */
export async function POST(req: NextRequest) {
  const token = readSessionToken(req);
  if (token) {
    try {
      await logout(token);
    } catch (error) {
      console.error("Session revocation failed:", error);
    }
  }
  const res = NextResponse.json({ success: true });
  clearSessionCookie(res);
  return res;
}
