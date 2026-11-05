/**
 * Server-side client for the internal FastAPI service.
 *
 * FastAPI is the single source of truth for users, sessions, crops and
 * telemetry. Every request carries the shared INTERNAL_API_TOKEN; browser
 * code must never import this module.
 */

export interface UserInfo {
  id: string;
  full_name: string;
  email: string;
  username: string;
  role: string;
}

export interface AuthResult {
  user: UserInfo;
  token: string;
}

/** An upstream error that carries the HTTP status FastAPI returned. */
export class UpstreamError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "UpstreamError";
  }
}

export function authApiBase(): string {
  return process.env.AUTH_API_URL?.trim() || "http://localhost:8000";
}

async function readErrorMessage(resp: Response, fallback: string): Promise<string> {
  const body: unknown = await resp.json().catch(() => null);
  if (typeof body === "object" && body !== null) {
    const detail = "detail" in body ? body.detail : "error" in body ? body.error : undefined;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors: [{ loc, msg, ... }]
    if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
  }
  return fallback;
}

interface RequestOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  sessionToken?: string;
  searchParams?: Record<string, string | undefined>;
  timeoutMs?: number;
}

export async function internalRequest<T>(
  path: string,
  fallbackError: string,
  { method = "GET", body, sessionToken, searchParams, timeoutMs = 5_000 }: RequestOptions = {}
): Promise<T> {
  const internalToken = process.env.INTERNAL_API_TOKEN;
  if (!internalToken) {
    throw new UpstreamError("INTERNAL_API_TOKEN is not configured", 503);
  }
  const url = new URL(path, authApiBase());
  for (const [key, value] of Object.entries(searchParams ?? {})) {
    if (value !== undefined) url.searchParams.set(key, value);
  }
  const headers: Record<string, string> = { "X-Internal-Token": internalToken };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (sessionToken) headers["X-Session-Token"] = sessionToken;

  let resp: Response;
  try {
    resp = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
      cache: "no-store",
    });
  } catch {
    throw new UpstreamError("Backend service is unavailable", 503);
  }
  if (!resp.ok) {
    throw new UpstreamError(await readErrorMessage(resp, fallbackError), resp.status);
  }
  return (await resp.json()) as T;
}

export async function login(emailOrUsername: string, password: string): Promise<AuthResult> {
  const data = await internalRequest<{ user: UserInfo; token: string }>("/login", "Login failed", {
    method: "POST",
    body: { email_or_username: emailOrUsername, password },
  });
  return { user: data.user, token: data.token };
}

export async function signup(
  fullName: string,
  email: string,
  username: string,
  password: string
): Promise<AuthResult> {
  const data = await internalRequest<{ user: UserInfo; token: string }>("/signup", "Signup failed", {
    method: "POST",
    body: { full_name: fullName, email, username, password },
  });
  return { user: data.user, token: data.token };
}

/** Returns the user for a session token, or null when it is missing, invalid or expired. */
export async function getSessionUser(sessionToken: string | undefined): Promise<UserInfo | null> {
  if (!sessionToken) return null;
  try {
    const data = await internalRequest<{ user: UserInfo }>("/session", "Session check failed", {
      sessionToken,
    });
    return data.user;
  } catch (error) {
    if (error instanceof UpstreamError && error.status === 401) return null;
    throw error;
  }
}

export async function logout(sessionToken: string): Promise<void> {
  await internalRequest("/logout", "Logout failed", { method: "POST", sessionToken });
}

export async function saveCrop(cropData: {
  object_class: string;
  confidence: number;
  distance_m?: number;
  has_crop?: boolean;
  payload?: Record<string, unknown>;
}) {
  return internalRequest<{ success: boolean; crop: unknown }>("/crops", "Failed to save crop", {
    method: "POST",
    body: cropData,
  });
}

export async function getLatestCrop(objectClass?: string) {
  return internalRequest<{
    crop: {
      object_class: string;
      confidence: number;
      timestamp: string;
      payload?: { crop_image?: string } | null;
    } | null;
  }>("/crops/latest", "Failed to get latest crop", { searchParams: { object_class: objectClass } });
}

export async function recordDetections(
  detections: Array<{
    class_name: string;
    confidence: number;
    distance_m?: number | null;
    has_crop?: boolean;
  }>
) {
  if (detections.length === 0) return { success: true, count: 0 };
  return internalRequest<{ success: boolean; count: number }>(
    "/detections/batch",
    "Failed to record detections",
    { method: "POST", body: { detections } }
  );
}

export async function getStats() {
  return internalRequest<Record<string, unknown>>("/stats", "Failed to get stats");
}

export async function clearHistory() {
  return internalRequest<{ success: boolean }>("/history/clear", "Failed to clear history", {
    method: "POST",
  });
}

export async function requestPasswordReset(email: string) {
  return internalRequest<{ success: boolean }>("/password-reset/request", "Could not request a reset", {
    method: "POST",
    body: { email },
    timeoutMs: 15_000,
  });
}

export async function confirmPasswordReset(token: string, newPassword: string) {
  return internalRequest<{ success: boolean }>("/password-reset/confirm", "Could not reset the password", {
    method: "POST",
    body: { token, new_password: newPassword },
  });
}
