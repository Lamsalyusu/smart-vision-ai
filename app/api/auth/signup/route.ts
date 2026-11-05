import { NextRequest, NextResponse } from "next/server";
import { signup } from "@/lib/apiClient";
import { clientIp, setSessionCookie, upstreamMessage, upstreamStatus } from "@/lib/session";

/** Creates the account in FastAPI (validation and role assignment live there) and signs the user in. */
export async function POST(req: NextRequest) {
  const body: unknown = await req.json().catch(() => null);
  const field = (name: string) =>
    typeof body === "object" && body !== null && name in body
      ? String((body as Record<string, unknown>)[name] ?? "")
      : "";
  const fullName = field("full_name").trim();
  const email = field("email").trim().toLowerCase();
  const username = field("username").trim();
  const password = field("password");

  if (!fullName || !email || !username || !password) {
    return NextResponse.json({ error: "All fields are required" }, { status: 400 });
  }

  try {
    const result = await signup(fullName, email, username, password, clientIp(req));
    if (!result.auth) {
      // Verification is required before the first sign-in, so no session yet.
      return NextResponse.json({
        success: true,
        verification_required: true,
        verification_sent: result.verificationSent,
      });
    }
    const res = NextResponse.json({
      success: true,
      user: result.auth.user,
      verification_required: false,
      verification_sent: result.verificationSent,
    });
    setSessionCookie(res, result.auth.token);
    return res;
  } catch (error) {
    // FastAPI returns 422 for invalid input; present it as a plain 400.
    const status = upstreamStatus(error) === 422 ? 400 : upstreamStatus(error);
    if (status >= 500) console.error("Signup failed:", error);
    return NextResponse.json({ error: upstreamMessage(error, "Signup failed") }, { status });
  }
}
