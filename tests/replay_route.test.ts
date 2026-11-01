import assert from "node:assert/strict";
import test from "node:test";
import { NextRequest } from "next/server";
import { DELETE, GET, POST } from "../app/api/replay/route";

const SESSION_ID = "a4f8e98d-c19d-4c84-9d4c-82f4a1f68b08";

test("replay reads are scoped to the opaque session cookie", async () => {
  const originalAuthUrl = process.env.AUTH_API_URL;
  const originalInternalToken = process.env.INTERNAL_API_TOKEN;
  const originalFetch = globalThis.fetch;
  process.env.AUTH_API_URL = "http://auth.test";
  process.env.INTERNAL_API_TOKEN = "test-internal-token";
  globalThis.fetch = (async (input, init) => {
    assert.equal(String(input), `http://auth.test/replay/frames?session_id=${SESSION_ID}`);
    assert.equal(init?.method, "GET");
    assert.equal(new Headers(init?.headers).get("x-internal-token"), "test-internal-token");
    return Response.json({ success: true, frames: [] });
  }) as typeof fetch;

  try {
    const response = await GET(new NextRequest("http://localhost/api/replay", {
      headers: { cookie: `vision_session_id=${SESSION_ID}` },
    }));
    assert.equal(response.status, 200);
    assert.equal(response.cookies.get("vision_session_id"), undefined);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthUrl === undefined) delete process.env.AUTH_API_URL;
    else process.env.AUTH_API_URL = originalAuthUrl;
    if (originalInternalToken === undefined) delete process.env.INTERNAL_API_TOKEN;
    else process.env.INTERNAL_API_TOKEN = originalInternalToken;
  }
});

test("replay writes discard caller session IDs and attach a server session cookie", async () => {
  const originalAuthUrl = process.env.AUTH_API_URL;
  const originalInternalToken = process.env.INTERNAL_API_TOKEN;
  const originalFetch = globalThis.fetch;
  process.env.AUTH_API_URL = "http://auth.test";
  process.env.INTERNAL_API_TOKEN = "test-internal-token";
  globalThis.fetch = (async (input, init) => {
    const upstreamUrl = new URL(String(input));
    assert.equal(upstreamUrl.origin, "http://auth.test");
    assert.equal(upstreamUrl.pathname, "/replay/frames");
    assert.equal(new Headers(init?.headers).get("x-internal-token"), "test-internal-token");
    assert.match(upstreamUrl.searchParams.get("session_id") || "", /^[0-9a-f-]{36}$/i);
    const forwarded = JSON.parse(String(init?.body));
    assert.equal(forwarded.session_id, upstreamUrl.searchParams.get("session_id"));
    assert.notEqual(forwarded.session_id, "attacker-chosen-session");
    return Response.json({ success: true, frame: { capturedAt: 123, detections: [] } });
  }) as typeof fetch;

  try {
    const response = await POST(new NextRequest("http://localhost/api/replay", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        session_id: "attacker-chosen-session",
        capturedAt: 123,
        detections: [],
      }),
    }));
    assert.equal(response.status, 200);
    assert.ok(response.cookies.get("vision_session_id")?.value);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthUrl === undefined) delete process.env.AUTH_API_URL;
    else process.env.AUTH_API_URL = originalAuthUrl;
    if (originalInternalToken === undefined) delete process.env.INTERNAL_API_TOKEN;
    else process.env.INTERNAL_API_TOKEN = originalInternalToken;
  }
});

test("replay clear is forwarded as DELETE", async () => {
  const originalAuthUrl = process.env.AUTH_API_URL;
  const originalInternalToken = process.env.INTERNAL_API_TOKEN;
  const originalFetch = globalThis.fetch;
  process.env.AUTH_API_URL = "http://auth.test";
  process.env.INTERNAL_API_TOKEN = "test-internal-token";
  globalThis.fetch = (async (_input, init) => {
    assert.equal(init?.method, "DELETE");
    assert.equal(new Headers(init?.headers).get("x-internal-token"), "test-internal-token");
    return Response.json({ success: true });
  }) as typeof fetch;

  try {
    const response = await DELETE(new NextRequest("http://localhost/api/replay", {
      method: "DELETE",
      headers: { cookie: `vision_session_id=${SESSION_ID}` },
    }));
    assert.equal(response.status, 200);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthUrl === undefined) delete process.env.AUTH_API_URL;
    else process.env.AUTH_API_URL = originalAuthUrl;
    if (originalInternalToken === undefined) delete process.env.INTERNAL_API_TOKEN;
    else process.env.INTERNAL_API_TOKEN = originalInternalToken;
  }
});