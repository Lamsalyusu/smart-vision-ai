"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Eye, MailCheck, CheckCircle2 } from "lucide-react";

// The token is confirmed on a button click rather than on page load, so link
// scanners in mail clients that open the URL do not use it up.
export default function VerifyEmailPage() {
  const [token, setToken] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "working" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [email, setEmail] = useState("");
  const [resent, setResent] = useState(false);

  useEffect(() => {
    const value = new URLSearchParams(window.location.search).get("token");
    setToken(value ?? "");
    if (value) window.history.replaceState(null, "", "/verify-email");
  }, []);

  const confirm = async () => {
    setStatus("working");
    const res = await fetch("/api/auth/verify-email/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }).catch(() => null);
    const data = res ? await res.json().catch(() => ({})) : {};
    if (res?.ok) {
      setStatus("done");
    } else {
      setStatus("error");
      setMessage(data.error || "Could not verify the email address.");
    }
  };

  const resend = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = await fetch("/api/auth/verify-email/resend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    }).catch(() => null);
    const data = res ? await res.json().catch(() => ({})) : {};
    if (res?.ok) setResent(true);
    else setMessage(data.error || "Could not send a verification email.");
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex items-center justify-center p-6 sm:p-12">
      <div className="w-full max-w-md glass-panel p-8 sm:p-10">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/25">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg text-white">SmartVision AI</span>
        </div>

        <h1 className="text-2xl font-bold tracking-tight mb-6">Verify your email</h1>

        {status === "done" ? (
          <div className="p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
            <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-3" />
            <p className="text-sm text-emerald-300">Your email address is verified.</p>
            <Link href="/login" className="inline-block mt-4 text-xs text-blue-400 hover:underline">
              Continue to sign in
            </Link>
          </div>
        ) : token ? (
          <div className="space-y-4">
            <p className="text-sm text-slate-300">Confirm that this email address belongs to you.</p>
            {status === "error" && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">{message}</div>
            )}
            <button
              onClick={confirm}
              disabled={status === "working"}
              className="w-full btn-cyber-primary py-3 text-sm flex items-center justify-center gap-2 disabled:opacity-60"
            >
              <MailCheck className="w-4 h-4" />
              {status === "working" ? "Verifying…" : "Verify email address"}
            </button>
          </div>
        ) : token === "" ? (
          <form onSubmit={resend} className="space-y-4">
            <p className="text-sm text-slate-300">
              Open the link from your verification email, or request a new one below.
            </p>
            {resent ? (
              <p className="text-sm text-emerald-300">
                If an account uses {email} and is not verified yet, a new link is on its way.
              </p>
            ) : (
              <>
                {message && (
                  <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">{message}</div>
                )}
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-4 py-2.5 rounded-xl bg-slate-900/80 border border-white/10 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500"
                />
                <button type="submit" className="w-full btn-cyber-primary py-3 text-sm">
                  Send a new verification link
                </button>
              </>
            )}
          </form>
        ) : null}
      </div>
    </div>
  );
}
