"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Eye, Lock, CheckCircle2 } from "lucide-react";

export default function ResetPasswordPage() {
  const router = useRouter();
  const [token, setToken] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Read the token once and drop it from the address bar and history.
    const value = new URLSearchParams(window.location.search).get("token");
    setToken(value ?? "");
    if (value) window.history.replaceState(null, "", "/reset-password");
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/auth/password-reset/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, new_password: password }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setDone(true);
        setTimeout(() => router.push("/login"), 2000);
      } else {
        setError(data.error || "Could not reset the password.");
      }
    } catch {
      setError("Network or server connection failed.");
    } finally {
      setLoading(false);
    }
  };

  const inputClass =
    "w-full px-4 py-2.5 pl-10 rounded-xl bg-slate-900/80 border border-white/10 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500";

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 flex items-center justify-center p-6 sm:p-12">
      <div className="w-full max-w-md glass-panel p-8 sm:p-10">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/25">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-lg text-white">SmartVision AI</span>
        </div>

        <h1 className="text-2xl font-bold tracking-tight mb-6">Choose a new password</h1>

        {token === "" ? (
          <p className="text-sm text-slate-300">
            This page needs the link from your reset email.{" "}
            <Link href="/forgot-password" className="text-blue-400 hover:underline">
              Request a new link
            </Link>
            .
          </p>
        ) : done ? (
          <div className="p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
            <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-3" />
            <p className="text-sm text-emerald-300">Password updated. Redirecting to sign in…</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">{error}</div>
            )}
            <div className="relative">
              <input
                type="password"
                required
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="New password (8+ characters)"
                className={inputClass}
              />
              <Lock className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
            </div>
            <div className="relative">
              <input
                type="password"
                required
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                placeholder="Repeat new password"
                className={inputClass}
              />
              <Lock className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
            </div>
            <button
              type="submit"
              disabled={loading || token === null}
              className="w-full btn-cyber-primary py-3 text-sm mt-2 disabled:opacity-60"
            >
              {loading ? "Saving…" : "Set new password"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
