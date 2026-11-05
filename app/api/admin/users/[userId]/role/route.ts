import { NextRequest, NextResponse } from "next/server";
import { adminSetRole } from "@/lib/apiClient";
import { actorId, requireAdmin, upstreamFailure } from "@/lib/admin";

const ROLES = new Set(["admin", "analyst", "operator", "viewer"]);

export async function POST(req: NextRequest, { params }: { params: Promise<{ userId: string }> }) {
  const denied = requireAdmin(req);
  if (denied) return denied;
  const { userId } = await params;
  const body: Record<string, unknown> = await req.json().catch(() => ({}));
  const role = String(body.role ?? "");
  if (!ROLES.has(role)) {
    return NextResponse.json({ error: "Unknown role" }, { status: 400 });
  }
  try {
    return NextResponse.json(await adminSetRole(userId, role, actorId(req)));
  } catch (error) {
    return upstreamFailure(error, "Could not change the role");
  }
}
