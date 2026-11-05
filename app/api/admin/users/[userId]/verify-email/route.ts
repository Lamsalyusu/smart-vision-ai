import { NextRequest, NextResponse } from "next/server";
import { adminVerifyEmail } from "@/lib/apiClient";
import { actorId, requireAdmin, upstreamFailure } from "@/lib/admin";

export async function POST(req: NextRequest, { params }: { params: Promise<{ userId: string }> }) {
  const denied = requireAdmin(req);
  if (denied) return denied;
  const { userId } = await params;
  try {
    return NextResponse.json(await adminVerifyEmail(userId, actorId(req)));
  } catch (error) {
    return upstreamFailure(error, "Could not mark the email verified");
  }
}
