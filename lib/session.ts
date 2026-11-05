import type { NextRequest, NextResponse } from "next/server";
import { UpstreamError } from "./apiClient";

export const SESSION_COOKIE = "session_token";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

/** Headers the middleware sets after verifying a session. Never trust them from the client. */
export const USER_ID_HEADER = "x-smartvision-user-id";
export const USER_ROLE_HEADER = "x-smartvision-user-role";

/**
 * The browser's address for rate limiting. Proxy headers can be forged by
 * clients, so they are only used when TRUST_PROXY_HEADERS=true (set this when a
 * reverse proxy you control sets X-Forwarded-For). Otherwise every client shares
 * one "direct" bucket.
 */
export function clientIp(request: NextRequest): string {
  if (process.env.TRUST_PROXY_HEADERS === "true") {
    const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
    const real = request.headers.get("x-real-ip")?.trim();
    if (forwarded) return forwarded;
    if (real) return real;
  }
  return "direct";
}

export function readSessionToken(request: NextRequest): string | undefined {
  return request.cookies.get(SESSION_COOKIE)?.value || undefined;
}

export function setSessionCookie(response: NextResponse, token: string): void {
  response.cookies.set(SESSION_COOKIE, token, {
    path: "/",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: SESSION_MAX_AGE_SECONDS,
  });
}

export function clearSessionCookie(response: NextResponse): void {
  response.cookies.set(SESSION_COOKIE, "", {
    path: "/",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: 0,
  });
}

/** Maps an upstream failure to the status the browser should see. */
export function upstreamStatus(error: unknown): number {
  if (error instanceof UpstreamError) {
    // 5xx from FastAPI means our backend failed, which the client sees as 502.
    return error.status >= 500 && error.status !== 503 ? 502 : error.status;
  }
  return 500;
}

export function upstreamMessage(error: unknown, fallback: string): string {
  return error instanceof UpstreamError ? error.message : fallback;
}
