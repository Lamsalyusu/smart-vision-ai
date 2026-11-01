
// app/api/auth/signup/route.ts
import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";
import { hashPassword, SecurityAuditLog, UserRole } from "@/lib/auth-crypto";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { full_name, email, username, password } = body;

    if (!full_name || !email || !username || !password) {
      return NextResponse.json(
        { error: "All fields are required" },
        { status: 400 }
      );
    }

    const cleanEmail = String(email).toLowerCase().trim();
    const cleanUsername = String(username).trim();

    // Input validation
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail)) {
      return NextResponse.json({ error: "Invalid email format" }, { status: 400 });
    }
    if (cleanUsername.length < 3 || cleanUsername.length > 30) {
      return NextResponse.json(
        { error: "Username must be between 3 and 30 characters" },
        { status: 400 }
      );
    }
    if (String(password).length < 8) {
      return NextResponse.json(
        { error: "Password must be at least 8 characters long" },
        { status: 400 }
      );
    }

    if (dataManager.findUserByIdentifier(cleanEmail)) {
      return NextResponse.json(
        { error: "Email is already registered" },
        { status: 400 }
      );
    }

    if (dataManager.findUserByIdentifier(cleanUsername)) {
      return NextResponse.json(
        { error: "Username is already taken" },
        { status: 400 }
      );
    }

    // Cryptographic PBKDF2-SHA256 hashing
    const passwordHash = hashPassword(String(password));

    const newUser = dataManager.createUser(
      String(full_name).trim(),
      cleanEmail,
      cleanUsername,
      passwordHash,
      UserRole.OPERATOR
    );

    const sessionToken = dataManager.createSession(newUser.id);

    SecurityAuditLog.record({
      event_type: "LOGIN_SUCCESS",
      user_id: newUser.id,
      identifier: newUser.username,
      details: { action: "user_registered" },
    });

    const res = NextResponse.json({
      success: true,
      user: {
        id: newUser.id,
        full_name: newUser.full_name,
        email: newUser.email,
        username: newUser.username,
        role: newUser.role || "operator",
      },
    });
    res.headers.set('Content-Security-Policy', "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline';");

    // Secure HTTP-only session cookie
    res.cookies.set("session_token", sessionToken, {
      path: "/",
      httpOnly: true, // Secure against XSS token exfiltration
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 7,
    });

    return res;
  } catch (err: unknown) {
    console.error("Signup error:", err);
    return NextResponse.json({ error: "Signup failed" }, { status: 500 });
  }
}
