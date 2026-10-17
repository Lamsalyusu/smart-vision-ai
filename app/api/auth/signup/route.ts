import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

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

    if (dataManager.findUserByIdentifier(email)) {
      return NextResponse.json(
        { error: "Email is already registered" },
        { status: 400 }
      );
    }

    if (dataManager.findUserByIdentifier(username)) {
      return NextResponse.json(
        { error: "Username is already taken" },
        { status: 400 }
      );
    }

    const newUser = dataManager.createUser(
      full_name,
      email,
      username,
      `hash_${password}`
    );

    const sessionToken = dataManager.createSession(newUser.id);

    const res = NextResponse.json({
      success: true,
      user: {
        id: newUser.id,
        full_name: newUser.full_name,
        email: newUser.email,
        username: newUser.username,
      },
    });

    res.cookies.set("session_token", sessionToken, {
      path: "/",
      httpOnly: false,
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 7,
    });

    return res;
  } catch (err: unknown) {
    console.error("Signup error:", err);
    return NextResponse.json({ error: "Signup failed" }, { status: 500 });
  }
}
