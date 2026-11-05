import { NextRequest, NextResponse } from "next/server";
import { getSessionUser } from "@/lib/apiClient";
import { clearSessionCookie, readSessionToken } from "@/lib/session";

/** Reports whether the session cookie maps to a live FastAPI session. */
export async function GET(req: NextRequest) {
  const token = readSessionToken(req);
  if (!token) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  let user;
  try {
    user = await getSessionUser(token);
  } catch (error) {
    console.error("Session check failed:", error);
    return NextResponse.json({ error: "Authentication service is unavailable" }, { status: 503 });
  }

  if (!user) {
    const res = NextResponse.json({ authenticated: false, user: null });
    clearSessionCookie(res);
    return res;
  }
  return NextResponse.json({ authenticated: true, user });
}
