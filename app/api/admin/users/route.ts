import { NextRequest, NextResponse } from "next/server";
import { adminListUsers } from "@/lib/apiClient";
import { requireAdmin, upstreamFailure } from "@/lib/admin";

export async function GET(req: NextRequest) {
  const denied = requireAdmin(req);
  if (denied) return denied;
  try {
    return NextResponse.json(await adminListUsers());
  } catch (error) {
    return upstreamFailure(error, "Could not list users");
  }
}
