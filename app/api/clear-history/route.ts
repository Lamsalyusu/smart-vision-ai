import { NextResponse } from "next/server";
import { dataManager } from "@/lib/store";

export async function POST() {
  dataManager.clearHistory();
  return NextResponse.json({ success: true });
}
