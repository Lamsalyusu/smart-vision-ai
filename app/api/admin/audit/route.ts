import { NextRequest, NextResponse } from "next/server";
import { adminAuditLog } from "@/lib/apiClient";
import { requireAdmin, upstreamFailure } from "@/lib/admin";

export async function GET(req: NextRequest) {
  const denied = requireAdmin(req);
  if (denied) return denied;
  try {
    return NextResponse.json(await adminAuditLog(200));
  } catch (error) {
    return upstreamFailure(error, "Could not load the audit log");
  }
}
