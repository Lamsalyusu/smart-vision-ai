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
  email_verified?: boolean;
}

export interface AuthResult {
  user: UserInfo;
  token: string;
}

export interface SignupResult {
  /** Absent when email verification is required before the first sign-in. */
  auth?: AuthResult;
  verificationRequired: boolean;
  verificationSent: boolean;
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
  /** Browser client address, used by FastAPI's per-client rate limits. */
  clientIp?: string;
  searchParams?: Record<string, string | undefined>;
  timeoutMs?: number;
}

export async function internalRequest<T>(
  path: string,
  fallbackError: string,
  { method = "GET", body, sessionToken, clientIp, searchParams, timeoutMs = 5_000 }: RequestOptions = {}
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
  if (clientIp) headers["X-Client-IP"] = clientIp;

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

export async function login(emailOrUsername: string, password: string, clientIp?: string): Promise<AuthResult> {
  const data = await internalRequest<{ user: UserInfo; token: string }>("/login", "Login failed", {
    method: "POST",
    body: { email_or_username: emailOrUsername, password },
    clientIp,
  });
  return { user: data.user, token: data.token };
}

export async function signup(
  fullName: string,
  email: string,
  username: string,
  password: string,
  clientIp?: string
): Promise<SignupResult> {
  const data = await internalRequest<{
    user?: UserInfo;
    token?: string;
    verification_required?: boolean;
    verification_sent?: boolean;
  }>("/signup", "Signup failed", {
    method: "POST",
    body: { full_name: fullName, email, username, password },
    clientIp,
    timeoutMs: 15_000,
  });
  return {
    auth: data.user && data.token ? { user: data.user, token: data.token } : undefined,
    verificationRequired: data.verification_required === true,
    verificationSent: data.verification_sent === true,
  };
}

export async function confirmEmailVerification(token: string, clientIp?: string) {
  return internalRequest<{ success: boolean }>("/email-verification/confirm", "Could not verify the email address", {
    method: "POST",
    body: { token },
    clientIp,
  });
}

export async function resendEmailVerification(email: string, clientIp?: string) {
  return internalRequest<{ success: boolean }>("/email-verification/resend", "Could not send a verification email", {
    method: "POST",
    body: { email },
    clientIp,
    timeoutMs: 15_000,
  });
}

export interface AdminUser extends UserInfo {
  created_at: string | null;
  email_verified: boolean;
}

export interface AuditEvent {
  event_type: string;
  user_id: string | null;
  details: Record<string, unknown>;
  timestamp: string;
}

export async function adminListUsers() {
  return internalRequest<{ users: AdminUser[] }>("/admin/users", "Could not list users");
}

export async function adminSetRole(userId: string, role: string, actorId?: string) {
  return internalRequest<{ success: boolean; previous: string; role: string }>(
    `/admin/users/${encodeURIComponent(userId)}/role`,
    "Could not change the role",
    { method: "POST", body: { role, actor_id: actorId } }
  );
}

export async function adminVerifyEmail(userId: string, actorId?: string) {
  return internalRequest<{ success: boolean }>(
    `/admin/users/${encodeURIComponent(userId)}/verify-email`,
    "Could not mark the email verified",
    { method: "POST", searchParams: { actor_id: actorId } }
  );
}

export async function adminAuditLog(limit = 200) {
  return internalRequest<{ events: AuditEvent[] }>("/admin/audit", "Could not load the audit log", {
    searchParams: { limit: String(limit) },
  });
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

export interface FrameTiming {
  inference_ms?: number;
  total_ms?: number;
  roundtrip_ms?: number;
}

/** Records one processed frame: its detections (possibly none) and measured timings. */
export async function recordDetections(
  detections: Array<{
    class_name: string;
    confidence: number;
    distance_m?: number | null;
    has_crop?: boolean;
  }>,
  frame?: FrameTiming
) {
  if (detections.length === 0 && !frame) return { success: true, count: 0 };
  return internalRequest<{ success: boolean; count: number }>(
    "/detections/batch",
    "Failed to record detections",
    { method: "POST", body: { detections, ...(frame ? { frame } : {}) } }
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

export async function requestPasswordReset(email: string, clientIp?: string) {
  return internalRequest<{ success: boolean }>("/password-reset/request", "Could not request a reset", {
    method: "POST",
    body: { email },
    clientIp,
    timeoutMs: 15_000,
  });
}

export async function confirmPasswordReset(token: string, newPassword: string) {
  return internalRequest<{ success: boolean }>("/password-reset/confirm", "Could not reset the password", {
    method: "POST",
    body: { token, new_password: newPassword },
  });
}
