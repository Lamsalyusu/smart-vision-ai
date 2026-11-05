import { NextRequest, NextResponse } from "next/server";
import { USER_ID_HEADER, USER_ROLE_HEADER, upstreamMessage, upstreamStatus } from "./session";

/** 403 response unless the middleware verified an admin session; null when allowed. */
export function requireAdmin(req: NextRequest): NextResponse | null {
  if (req.headers.get(USER_ROLE_HEADER) !== "admin") {
    return NextResponse.json({ error: "Administrator access required" }, { status: 403 });
  }
  return null;
}

export function actorId(req: NextRequest): string | undefined {
  return req.headers.get(USER_ID_HEADER) ?? undefined;
}

export function upstreamFailure(error: unknown, fallback: string): NextResponse {
  return NextResponse.json({ error: upstreamMessage(error, fallback) }, { status: upstreamStatus(error) });
}
