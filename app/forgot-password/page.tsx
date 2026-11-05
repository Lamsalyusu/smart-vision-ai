"use client";

import { useState } from "react";
import Link from "next/link";
import { Eye, Mail, ArrowLeft, CheckCircle2 } from "lucide-react";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await fetch("/api/auth/password-reset/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        setSubmitted(true);
      } else {
        setError(data.error || "Could not request a password reset.");
      }
    } catch {
      setError("Network or server connection failed.");
    } finally {
      setLoading(false);
    }
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

        <h1 className="text-2xl font-bold tracking-tight">Reset your password</h1>
        <p className="text-slate-400 text-xs mt-1.5 mb-6">
          Enter the email address of your account and we will send you a link to choose a new password.
        </p>

        {submitted ? (
          <div className="p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
            <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto mb-3" />
            <h3 className="font-semibold text-emerald-300 text-sm">Check your email</h3>
            <p className="text-xs text-slate-400 mt-2">
              If an account uses {email}, a reset link is on its way. The link expires in 30 minutes.
            </p>
            <Link href="/login" className="inline-block mt-6 text-xs text-blue-400 font-medium hover:underline">
              Back to Sign In
            </Link>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">{error}</div>
            )}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Email Address</label>
              <div className="relative">
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-4 py-2.5 pl-10 rounded-xl bg-slate-900/80 border border-white/10 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500"
                />
                <Mail className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-cyber-primary py-3 text-sm flex items-center justify-center gap-2 mt-4 disabled:opacity-60"
            >
              {loading ? "Sending…" : "Send Recovery Link"}
            </button>

            <div className="pt-4 text-center">
              <Link
                href="/login"
                className="inline-flex items-center gap-2 text-xs text-slate-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                Back to Sign In
              </Link>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
