import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { email_or_username, password } = body;

    if (!email_or_username || !password) {
      return NextResponse.json(
        { error: "Please provide email/username and password" },
        { status: 400 }
      );
    }

    const user = dataManager.findUserByIdentifier(email_or_username);
    if (!user) {
      return NextResponse.json(
        { error: "Invalid credentials" },
        { status: 401 }
      );
    }

    const sessionToken = dataManager.createSession(user.id);

    const res = NextResponse.json({
      success: true,
      user: {
        id: user.id,
        full_name: user.full_name,
        email: user.email,
        username: user.username,
      },
    });

    res.cookies.set("session_token", sessionToken, {
      path: "/",
      httpOnly: false, // accessible to client for simplicity
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 7,
    });

    return res;
  } catch (err: unknown) {
    console.error("Login error:", err);
    return NextResponse.json({ error: "Login failed" }, { status: 500 });
  }
}
