import assert from "node:assert/strict";
import test from "node:test";
import { NextRequest } from "next/server";
import { POST as generate3d } from "../app/api/generate-3d/route";
import { GET as adminUsers } from "../app/api/admin/users/route";
import { POST as adminRole } from "../app/api/admin/users/[userId]/role/route";
import { GET as adminAudit } from "../app/api/admin/audit/route";
import { POST as verifyConfirm } from "../app/api/auth/verify-email/confirm/route";
import { POST as verifyResend } from "../app/api/auth/verify-email/resend/route";
import { POST as signup } from "../app/api/auth/signup/route";
import { POST as login } from "../app/api/auth/login/route";
import { clientIp, USER_ID_HEADER, USER_ROLE_HEADER } from "../lib/session";

type Handler = (url: URL, init: RequestInit | undefined) => Response | Promise<Response>;

async function withFetch(env: Record<string, string>, handler: Handler, run: () => Promise<void>) {
  const savedFetch = globalThis.fetch;
  const savedEnv: Record<string, string | undefined> = {};
  for (const [key, value] of Object.entries(env)) {
    savedEnv[key] = process.env[key];
    process.env[key] = value;
  }
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) =>
    handler(new URL(String(input)), init)) as typeof fetch;
  try {
    await run();
  } finally {
    globalThis.fetch = savedFetch;
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

const BACKEND = { AUTH_API_URL: "http://auth.test", INTERNAL_API_TOKEN: "internal-secret" };

function post(path: string, body: unknown, headers: Record<string, string> = {}) {
  return new NextRequest(`http://localhost${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
}

const CROP = "data:image/jpeg;base64,/9j/AAAA";

test("point cloud without a crop is the labelled synthetic demo", async () => {
  await withFetch({}, () => {
    throw new Error("no upstream call expected");
  }, async () => {
    const res = await generate3d(post("/api/generate-3d", { depth_scale: 2, max_points: 2000 }));
    const body = await res.json();
    assert.equal(res.status, 200);
    assert.equal(body.source, "demo");
    assert.ok(body.point_count > 0);
  });
});

test("point cloud with a crop is computed by the vision service's depth model", async () => {
  await withFetch(
    { VISION_API_URL: "http://vision.test/api/detect/", INTERNAL_API_TOKEN: "internal-secret" },
    (url, init) => {
      assert.equal(url.toString(), "http://vision.test/api/generate-3d/");
      assert.equal(new Headers(init?.headers).get("x-internal-token"), "internal-secret");
      const body = JSON.parse(String(init?.body));
      assert.equal(body.image_base64, CROP);
      assert.equal(body.depth_scale, 3);
      return Response.json({ success: true, positions: [0, 0, 1], colors: [1, 1, 1], point_count: 1, width: 4, height: 4 });
    },
    async () => {
      const res = await generate3d(post("/api/generate-3d", { image_base64: CROP, depth_scale: 3 }));
      const body = await res.json();
      assert.equal(res.status, 200);
      assert.equal(body.source, "midas");
      assert.equal(body.point_count, 1);
    }
  );
});

test("point cloud reports depth unavailability instead of faking it", async () => {
  await withFetch(
    { VISION_API_URL: "http://vision.test/api/detect/" },
    () => Response.json({ error: "Depth estimation is unavailable" }, { status: 503 }),
    async () => {
      const res = await generate3d(post("/api/generate-3d", { image_base64: CROP }));
      assert.equal(res.status, 503);
      assert.match((await res.json()).error, /unavailable/);
    }
  );
  const bad = await generate3d(post("/api/generate-3d", { image_base64: "not-an-image" }));
  assert.equal(bad.status, 400);
});

test("admin routes refuse non-admins without calling the backend", async () => {
  await withFetch(BACKEND, () => {
    throw new Error("backend must not be called");
  }, async () => {
    const operator = { [USER_ROLE_HEADER]: "operator" };
    assert.equal((await adminUsers(new NextRequest("http://localhost/api/admin/users", { headers: operator }))).status, 403);
    assert.equal((await adminAudit(new NextRequest("http://localhost/api/admin/audit", { headers: operator }))).status, 403);
    const res = await adminRole(post("/api/admin/users/u1/role", { role: "admin" }, operator), {
      params: Promise.resolve({ userId: "u1" }),
    });
    assert.equal(res.status, 403);
  });
});

test("admin role change forwards the acting admin and validates the role", async () => {
  await withFetch(BACKEND, (url, init) => {
    assert.equal(url.pathname, "/admin/users/usr%2F1/role");
    assert.deepEqual(JSON.parse(String(init?.body)), { role: "viewer", actor_id: "usr_admin" });
    return Response.json({ success: true, previous: "operator", role: "viewer" });
  }, async () => {
    const admin = { [USER_ROLE_HEADER]: "admin", [USER_ID_HEADER]: "usr_admin" };
    const ok = await adminRole(post("/api/admin/users/x/role", { role: "viewer" }, admin), {
      params: Promise.resolve({ userId: "usr/1" }),
    });
    assert.equal(ok.status, 200);
    const invalid = await adminRole(post("/api/admin/users/x/role", { role: "root" }, admin), {
      params: Promise.resolve({ userId: "usr/1" }),
    });
    assert.equal(invalid.status, 400);
  });
});

test("last-admin protection from the backend reaches the admin", async () => {
  await withFetch(BACKEND, () => Response.json({ detail: "Cannot remove the last administrator" }, { status: 400 }), async () => {
    const res = await adminRole(post("/api/admin/users/x/role", { role: "viewer" }, { [USER_ROLE_HEADER]: "admin" }), {
      params: Promise.resolve({ userId: "u1" }),
    });
    assert.equal(res.status, 400);
    assert.match((await res.json()).error, /last administrator/);
  });
});

test("email verification confirm and resend proxy to the backend", async () => {
  await withFetch(BACKEND, (url, init) => {
    if (url.pathname === "/email-verification/confirm") {
      return JSON.parse(String(init?.body)).token === "good"
        ? Response.json({ success: true })
        : Response.json({ detail: "This verification link is invalid or has expired" }, { status: 400 });
    }
    assert.equal(url.pathname, "/email-verification/resend");
    return Response.json({ success: true });
  }, async () => {
    assert.equal((await verifyConfirm(post("/api/auth/verify-email/confirm", { token: "good" }))).status, 200);
    const bad = await verifyConfirm(post("/api/auth/verify-email/confirm", { token: "bad" }));
    assert.equal(bad.status, 400);
    assert.equal((await verifyResend(post("/api/auth/verify-email/resend", { email: "a@b.co" }))).status, 200);
    assert.equal((await verifyResend(post("/api/auth/verify-email/resend", {}))).status, 400);
  });
});

test("signup that requires verification sets no session cookie", async () => {
  await withFetch(BACKEND, () => Response.json({ success: true, user_id: "u1", verification_required: true, verification_sent: true }), async () => {
    const res = await signup(post("/api/auth/signup", { full_name: "A", email: "a@b.co", username: "abc", password: "long-enough" }));
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { success: true, verification_required: true, verification_sent: true });
    assert.equal(res.cookies.get("session_token"), undefined);
  });
});

test("client IP is only taken from proxy headers when they are trusted", async () => {
  const req = new NextRequest("http://localhost/x", { headers: { "x-forwarded-for": "198.51.100.4, 10.0.0.1" } });
  const saved = process.env.TRUST_PROXY_HEADERS;
  try {
    delete process.env.TRUST_PROXY_HEADERS;
    assert.equal(clientIp(req), "direct");
    process.env.TRUST_PROXY_HEADERS = "true";
    assert.equal(clientIp(req), "198.51.100.4");
  } finally {
    if (saved === undefined) delete process.env.TRUST_PROXY_HEADERS;
    else process.env.TRUST_PROXY_HEADERS = saved;
  }

  await withFetch({ ...BACKEND, TRUST_PROXY_HEADERS: "true" }, (_url, init) => {
    assert.equal(new Headers(init?.headers).get("x-client-ip"), "198.51.100.4");
    return Response.json({ detail: "Too many requests. Please wait and try again." }, { status: 429 });
  }, async () => {
    const res = await login(post("/api/auth/login", { email_or_username: "a", password: "b" }, { "x-forwarded-for": "198.51.100.4" }));
    assert.equal(res.status, 429);
  });
});
