import { NextRequest, NextResponse } from "next/server";
import { getSessionUser } from "@/lib/apiClient";
import { readSessionToken, USER_ID_HEADER, USER_ROLE_HEADER } from "@/lib/session";

/**
 * Requires a valid FastAPI session for the app pages and the API routes that
 * read or write user data, run inference, or call paid services.
 *
 * On success the verified user id and role are forwarded to route handlers in
 * request headers. Copies of those headers sent by the client are removed first.
 */
export async function middleware(request: NextRequest) {
  const isApi = request.nextUrl.pathname.startsWith("/api/");
  const forwarded = new Headers(request.headers);
  forwarded.delete(USER_ID_HEADER);
  forwarded.delete(USER_ROLE_HEADER);

  let user;
  try {
    user = await getSessionUser(readSessionToken(request));
  } catch {
    if (isApi) {
      return NextResponse.json({ error: "Authentication service is unavailable" }, { status: 503 });
    }
    return redirectToLogin(request, "unavailable");
  }

  if (!user) {
    if (isApi) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 });
    }
    return redirectToLogin(request);
  }

  forwarded.set(USER_ID_HEADER, user.id);
  forwarded.set(USER_ROLE_HEADER, user.role);
  return NextResponse.next({ request: { headers: forwarded } });
}

function redirectToLogin(request: NextRequest, reason?: string) {
  const url = new URL("/login", request.url);
  url.searchParams.set("next", request.nextUrl.pathname);
  if (reason) url.searchParams.set("reason", reason);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/camera/:path*",
    "/model-viewer/:path*",
    "/gpu-viewer/:path*",
    "/replay/:path*",
    "/api/detect/:path*",
    "/api/chat/:path*",
    "/api/replay/:path*",
    "/api/save-crop/:path*",
    "/api/get-crop/:path*",
    "/api/user/:path*",
    "/api/clear-history/:path*",
    "/api/generate-3d/:path*",
    "/api/generate-3d-mesh/:path*",
    "/api/3d-model/:path*",
  ],
};
