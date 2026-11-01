/**
 * Minimal wrapper around the FastAPI authentication service.
 * All Next.js auth routes will delegate to these functions.
 */

export interface UserInfo {
  id: string;
  full_name: string;
  email: string;
  username: string;
  role: string;
}

export async function login(emailOrUsername: string, password: string): Promise<UserInfo> {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email_or_username: emailOrUsername, password }),
    credentials: "include",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error ?? "Login failed");
  }
  const data = await resp.json();
  return data.user;
}

export async function signup(fullName: string, email: string, username: string, password: string, role?: string) {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_name: fullName, email, username, password, role }),
    credentials: "include",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Signup failed");
  }
  return await resp.json();
}

export async function getSession(): Promise<UserInfo> {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/session`, {
    method: "GET",
    credentials: "include",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Session fetch failed");
  }
  const data = await resp.json();
  return data.user;
}

export async function logout(): Promise<void> {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/logout`, {
    method: "POST",
    credentials: "include",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Logout failed");
  }
}

export async function saveCrop(cropData: {
  object_class: string;
  confidence: number;
  distance_m?: number;
  has_crop?: boolean;
  payload?: Record<string, unknown>;
}) {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/crops`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cropData),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to save crop");
  }
  return await resp.json();
}

export async function getLatestCrop(objectClass?: string) {
  const url = new URL(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/crops/latest`);
  if (objectClass) {
    url.searchParams.set("object_class", objectClass);
  }
  const resp = await fetch(url.toString(), {
    method: "GET",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to get latest crop");
  }
  return await resp.json();
}

export async function recordDetection(
  className: string,
  confidence: number,
  distanceM?: number,
  hasCrop: boolean = false
) {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/detections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      class_name: className,
      confidence,
      distance_m: distanceM,
      has_crop: hasCrop,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to record detection");
  }
  return await resp.json();
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
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/detections/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detections }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to record detections");
  }
  return await resp.json();
}

export async function getStats() {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/stats`, {
    method: "GET",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to get stats");
  }
  return await resp.json();
}

export async function clearHistory() {
  const resp = await fetch(`${process.env.AUTH_API_URL ?? "http://localhost:8000"}/history/clear`, {
    method: "POST",
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail ?? "Failed to clear history");
  }
  return await resp.json();
}
