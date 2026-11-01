import { randomUUID } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

const SESSION_COOKIE = "vision_session_id";
const MAX_DETECTIONS_PER_FRAME = 100;

function getSessionId(request: NextRequest): { id: string; shouldSetCookie: boolean } {
  const existing = request.cookies.get(SESSION_COOKIE)?.value;
  if (existing && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(existing)) {
    return { id: existing, shouldSetCookie: false };
  }
  return { id: randomUUID(), shouldSetCookie: true };
}

function respond(
  body: unknown,
  status: number,
  session: { id: string; shouldSetCookie: boolean }
) {
  const response = NextResponse.json(body, { status });
  if (session.shouldSetCookie) {
    response.cookies.set(SESSION_COOKIE, session.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 30,
    });
  }
  return response;
}

async function forwardReplayRequest(
  request: NextRequest,
  method: "GET" | "POST" | "DELETE"
) {
  const session = getSessionId(request);
  const upstreamBase = process.env.AUTH_API_URL ?? "http://localhost:8000";
  const internalToken = process.env.INTERNAL_API_TOKEN;
  if (!internalToken) {
    return respond({ error: "Replay service is not configured" }, 503, session);
  }
  const upstreamUrl = new URL("/replay/frames", upstreamBase);
  upstreamUrl.searchParams.set("session_id", session.id);

  let body: string | undefined;
  if (method === "POST") {
    let payload: unknown;
    try {
      payload = await request.json();
    } catch {
      return respond({ error: "Replay frame must be valid JSON" }, 400, session);
    }
    if (
      typeof payload !== "object" || payload === null ||
      !("capturedAt" in payload) || typeof payload.capturedAt !== "number" ||
      !Number.isFinite(payload.capturedAt) ||
      !("detections" in payload) || !Array.isArray(payload.detections) ||
      payload.detections.length > MAX_DETECTIONS_PER_FRAME
    ) {
      return respond({ error: "Replay frame is invalid or too large" }, 400, session);
    }
    body = JSON.stringify({
      session_id: session.id,
      captured_at: payload.capturedAt,
      detections: payload.detections,
    });
  }

  try {
    const upstream = await fetch(upstreamUrl, {
      method,
      headers: {
        "X-Internal-Token": internalToken,
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body,
      signal: AbortSignal.timeout(5_000),
      cache: "no-store",
    });
    const result = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      return respond({ error: "Replay service request failed" }, 502, session);
    }
    return respond(result, 200, session);
  } catch {
    return respond({ error: "Replay service is unavailable" }, 503, session);
  }
}

export async function GET(request: NextRequest) {
  return forwardReplayRequest(request, "GET");
}

export async function POST(request: NextRequest) {
  return forwardReplayRequest(request, "POST");
}

export async function DELETE(request: NextRequest) {
  return forwardReplayRequest(request, "DELETE");
}