import { NextRequest, NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function GET(req: NextRequest) {
  const token = req.cookies.get("session_token")?.value;
  if (!token) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  const user = dataManager.getUserFromSession(token);
  if (!user) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  return NextResponse.json({
    authenticated: true,
    user: {
      id: user.id,
      full_name: user.full_name,
      email: user.email,
      username: user.username,
    },
  });
}
