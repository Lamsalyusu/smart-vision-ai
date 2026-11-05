import assert from "node:assert/strict";
import test from "node:test";
import { NextRequest } from "next/server";
import { POST as login } from "../app/api/auth/login/route";
import { POST as signup } from "../app/api/auth/signup/route";
import { GET as session } from "../app/api/auth/session/route";
import { POST as logout } from "../app/api/auth/logout/route";
import { POST as clearHistory } from "../app/api/clear-history/route";
import { POST as resetRequest } from "../app/api/auth/password-reset/request/route";
import { POST as resetConfirm } from "../app/api/auth/password-reset/confirm/route";
import { middleware } from "../middleware";
import { USER_ID_HEADER, USER_ROLE_HEADER } from "../lib/session";

const USER = {
  id: "usr_1",
  full_name: "Test User",
  email: "test@example.com",
  username: "tester",
  role: "operator",
};

type FetchHandler = (url: URL, init: RequestInit | undefined) => Response | Promise<Response>;

async function withBackend(handler: FetchHandler, run: () => Promise<void>) {
  const saved = {
    fetch: globalThis.fetch,
    url: process.env.AUTH_API_URL,
    token: process.env.INTERNAL_API_TOKEN,
  };
  process.env.AUTH_API_URL = "http://auth.test";
  process.env.INTERNAL_API_TOKEN = "internal-secret";
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    assert.equal(headers.get("x-internal-token"), "internal-secret");
    return handler(new URL(String(input)), init);
  }) as typeof fetch;
  try {
    await run();
  } finally {
    globalThis.fetch = saved.fetch;
    if (saved.url === undefined) delete process.env.AUTH_API_URL;
    else process.env.AUTH_API_URL = saved.url;
    if (saved.token === undefined) delete process.env.INTERNAL_API_TOKEN;
    else process.env.INTERNAL_API_TOKEN = saved.token;
  }
}

function jsonRequest(path: string, body: unknown, cookie?: string) {
  return new NextRequest(`http://localhost${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...(cookie ? { cookie } : {}) },
    body: JSON.stringify(body),
  });
}

test("login stores the FastAPI session token in an httpOnly cookie", async () => {
  await withBackend(
    (url) => {
      assert.equal(url.pathname, "/login");
      return Response.json({ success: true, token: "tok-from-fastapi", user: USER });
    },
    async () => {
      const res = await login(jsonRequest("/api/auth/login", { email_or_username: "tester", password: "pw" }));
      assert.equal(res.status, 200);
      const cookie = res.cookies.get("session_token");
      assert.equal(cookie?.value, "tok-from-fastapi");
      assert.equal(cookie?.httpOnly, true);
      assert.deepEqual((await res.json()).user, USER);
    }
  );
});

test("login passes through 401 and 429 with FastAPI's message", async () => {
  for (const [status, detail] of [
    [401, "Invalid email/username or password"],
    [429, "Account temporarily locked"],
  ] as const) {
    await withBackend(
      () => Response.json({ detail }, { status }),
      async () => {
        const res = await login(jsonRequest("/api/auth/login", { email_or_username: "x", password: "y" }));
        assert.equal(res.status, status);
        assert.equal((await res.json()).error, detail);
        assert.equal(res.cookies.get("session_token"), undefined);
      }
    );
  }
});

test("login reports 503 when the backend is unreachable", async () => {
  await withBackend(
    () => {
      throw new TypeError("fetch failed");
    },
    async () => {
      const res = await login(jsonRequest("/api/auth/login", { email_or_username: "x", password: "y" }));
      assert.equal(res.status, 503);
    }
  );
});

test("signup signs the user in and maps validation errors to 400", async () => {
  await withBackend(
    (url, init) => {
      assert.equal(url.pathname, "/signup");
      const body = JSON.parse(String(init?.body));
      assert.equal(body.email, "new@example.com");
      assert.equal("role" in body, false);
      return Response.json({ success: true, token: "new-token", user: USER });
    },
    async () => {
      const res = await signup(
        jsonRequest("/api/auth/signup", {
          full_name: "New",
          email: " New@Example.com ",
          username: "newuser",
          password: "long-enough",
          role: "admin",
        })
      );
      assert.equal(res.status, 200);
      assert.equal(res.cookies.get("session_token")?.value, "new-token");
    }
  );

  await withBackend(
    () => Response.json({ detail: [{ msg: "String should have at least 8 characters" }] }, { status: 422 }),
    async () => {
      const res = await signup(
        jsonRequest("/api/auth/signup", { full_name: "a", email: "a@b.co", username: "abc", password: "x" })
      );
      assert.equal(res.status, 400);
      assert.match((await res.json()).error, /at least 8/);
    }
  );
});

test("session forwards the cookie token and clears invalid sessions", async () => {
  await withBackend(
    (url, init) => {
      assert.equal(url.pathname, "/session");
      const token = new Headers(init?.headers).get("x-session-token");
      return token === "good"
        ? Response.json({ user: USER })
        : Response.json({ detail: "Invalid or expired session" }, { status: 401 });
    },
    async () => {
      const ok = await session(new NextRequest("http://localhost/api/auth/session", { headers: { cookie: "session_token=good" } }));
      assert.deepEqual(await ok.json(), { authenticated: true, user: USER });

      const bad = await session(new NextRequest("http://localhost/api/auth/session", { headers: { cookie: "session_token=bad" } }));
      assert.equal((await bad.json()).authenticated, false);
      assert.equal(bad.cookies.get("session_token")?.value, "");
    }
  );
});

test("logout revokes the session upstream and clears the cookie", async () => {
  let revoked: string | null = null;
  await withBackend(
    (url, init) => {
      assert.equal(url.pathname, "/logout");
      revoked = new Headers(init?.headers).get("x-session-token");
      return Response.json({ success: true });
    },
    async () => {
      const res = await logout(jsonRequest("/api/auth/logout", {}, "session_token=abc"));
      assert.equal(res.status, 200);
      assert.equal(revoked, "abc");
      assert.equal(res.cookies.get("session_token")?.value, "");
    }
  );
});

test("middleware redirects anonymous page visits to login", async () => {
  await withBackend(
    () => Response.json({ detail: "No session token" }, { status: 401 }),
    async () => {
      const res = await middleware(new NextRequest("http://localhost/camera"));
      assert.equal(res.status, 307);
      const location = new URL(res.headers.get("location") ?? "");
      assert.equal(location.pathname, "/login");
      assert.equal(location.searchParams.get("next"), "/camera");
    }
  );
});

test("middleware returns 401 for anonymous API calls", async () => {
  await withBackend(
    () => Response.json({ detail: "Invalid or expired session" }, { status: 401 }),
    async () => {
      const res = await middleware(
        new NextRequest("http://localhost/api/detect", { headers: { cookie: "session_token=stale" } })
      );
      assert.equal(res.status, 401);
    }
  );
});

test("middleware forwards the verified identity and drops spoofed identity headers", async () => {
  await withBackend(
    () => Response.json({ user: USER }),
    async () => {
      const res = await middleware(
        new NextRequest("http://localhost/api/clear-history", {
          headers: {
            cookie: "session_token=good",
            [USER_ROLE_HEADER]: "admin",
            [USER_ID_HEADER]: "usr_attacker",
          },
        })
      );
      assert.equal(res.status, 200);
      // NextResponse.next() records overridden request headers with this prefix.
      assert.equal(res.headers.get(`x-middleware-request-${USER_ROLE_HEADER}`), "operator");
      assert.equal(res.headers.get(`x-middleware-request-${USER_ID_HEADER}`), "usr_1");
    }
  );
});

test("clear-history is limited to admins", async () => {
  const operator = new NextRequest("http://localhost/api/clear-history", {
    method: "POST",
    headers: { [USER_ROLE_HEADER]: "operator" },
  });
  assert.equal((await clearHistory(operator)).status, 403);

  await withBackend(
    (url) => {
      assert.equal(url.pathname, "/history/clear");
      return Response.json({ success: true });
    },
    async () => {
      const admin = new NextRequest("http://localhost/api/clear-history", {
        method: "POST",
        headers: { [USER_ROLE_HEADER]: "admin" },
      });
      assert.equal((await clearHistory(admin)).status, 200);
    }
  );
});

test("password reset request forwards the email and passes through 503", async () => {
  await withBackend(
    (url, init) => {
      assert.equal(url.pathname, "/password-reset/request");
      assert.deepEqual(JSON.parse(String(init?.body)), { email: "user@example.com" });
      return Response.json({ success: true });
    },
    async () => {
      const res = await resetRequest(jsonRequest("/api/auth/password-reset/request", { email: " user@example.com " }));
      assert.equal(res.status, 200);
    }
  );
  await withBackend(
    () => Response.json({ detail: "Password reset email is not configured" }, { status: 503 }),
    async () => {
      const res = await resetRequest(jsonRequest("/api/auth/password-reset/request", { email: "a@b.co" }));
      assert.equal(res.status, 503);
      assert.match((await res.json()).error, /not configured/);
    }
  );
});

test("password reset confirm clears the session cookie and reports invalid tokens", async () => {
  await withBackend(
    (url, init) => {
      assert.equal(url.pathname, "/password-reset/confirm");
      assert.deepEqual(JSON.parse(String(init?.body)), { token: "tok", new_password: "new-password" });
      return Response.json({ success: true });
    },
    async () => {
      const res = await resetConfirm(
        jsonRequest("/api/auth/password-reset/confirm", { token: "tok", new_password: "new-password" }, "session_token=old")
      );
      assert.equal(res.status, 200);
      assert.equal(res.cookies.get("session_token")?.value, "");
    }
  );
  await withBackend(
    () => Response.json({ detail: "This reset link is invalid or has expired" }, { status: 400 }),
    async () => {
      const res = await resetConfirm(
        jsonRequest("/api/auth/password-reset/confirm", { token: "bad", new_password: "new-password" })
      );
      assert.equal(res.status, 400);
      assert.match((await res.json()).error, /invalid or has expired/);
    }
  );
});
