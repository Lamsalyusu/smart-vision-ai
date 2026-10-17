import { NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function GET() {
  const stats = dataManager.getStats();
  return NextResponse.json({
    success: true,
    stats,
  });
}
