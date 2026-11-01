import { NextRequest, NextResponse } from "next/server";
import { login } from "@/lib/apiClient";
import { dataManager } from "@/lib/store";
import { SecurityAuditLog } from "@/lib/auth-crypto";

/**
 * Handles login requests.
 * - Proxies authentication to FastAPI (`login`).
 * - Performs brute‑force lockout checks.
 * - Issues a secure HTTP‑only session cookie.
 */
export async function POST(req: NextRequest) {
  try {
    const { email_or_username, password } = await req.json();
    if (!email_or_username || !password) {
      return NextResponse.json(
        { error: "Please provide email/username and password" },
        { status: 400 }
      );
    }

    // Brute‑force lockout check
    const cleanIdentifier = String(email_or_username).trim();
    if (dataManager.isAccountLocked(cleanIdentifier)) {
      return NextResponse.json(
        {
          error: "Account temporarily locked due to consecutive failed attempts. Please try again in 15 minutes."
        },
        { status: 429 }
      );
    }

    // Authenticate via FastAPI
    const user = await login(email_or_username, password);

    // Successful login: clear lockout, create session, audit
    dataManager.clearFailedLogins(cleanIdentifier);
    const sessionToken = dataManager.createSession(user.id);
    SecurityAuditLog.record({
      event_type: "LOGIN_SUCCESS",
      user_id: user.id,
      identifier: cleanIdentifier
    });

    const res = NextResponse.json({ success: true, user });
    res.headers.set(
      "Content-Security-Policy",
      "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline';"
    );
    res.cookies.set("session_token", sessionToken, {
      path: "/",
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });
    return res;
  } catch (err: unknown) {
    console.error("Login error:", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Login failed" },
      { status: 500 }
    );
  }
}
