"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ShieldCheck, Users, ScrollText, RefreshCw } from "lucide-react";

interface AdminUser {
  id: string;
  full_name: string;
  email: string;
  username: string;
  role: string;
  created_at: string | null;
  email_verified: boolean;
}

interface AuditEvent {
  event_type: string;
  user_id: string | null;
  details: Record<string, unknown>;
  timestamp: string;
}

const ROLES = ["admin", "analyst", "operator", "viewer"];

export default function AdminPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [usersRes, auditRes] = await Promise.all([fetch("/api/admin/users"), fetch("/api/admin/audit")]);
      const usersData = await usersRes.json().catch(() => ({}));
      const auditData = await auditRes.json().catch(() => ({}));
      if (!usersRes.ok) throw new Error(usersData.error || `Could not load users (${usersRes.status})`);
      if (!auditRes.ok) throw new Error(auditData.error || `Could not load the audit log (${auditRes.status})`);
      setUsers(usersData.users ?? []);
      setEvents(auditData.events ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load admin data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const act = async (url: string, body: unknown, success: string) => {
    setNotice(null);
    setError(null);
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => null);
    const data = res ? await res.json().catch(() => ({})) : {};
    if (res?.ok) {
      setNotice(success);
      await load();
    } else {
      setError(data.error || "The change failed");
    }
  };

  const userName = (id: string | null) => users.find((u) => u.id === id)?.username ?? id ?? "—";

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100">
      <header className="border-b border-white/10 bg-black/40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-blue-400" />
            <span className="font-bold">Administration</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={load}
              className="text-xs text-slate-300 flex items-center gap-1.5 hover:text-white"
              aria-label="Refresh"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Refresh
            </button>
            <Link href="/dashboard" className="text-xs text-slate-300 flex items-center gap-1.5 hover:text-white">
              <ArrowLeft className="w-3.5 h-3.5" /> Dashboard
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {error && (
          <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-sm text-red-300">{error}</div>
        )}
        {notice && (
          <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-sm text-emerald-300">
            {notice}
          </div>
        )}

        <section className="glass-panel p-5">
          <h2 className="font-semibold flex items-center gap-2 mb-4">
            <Users className="w-4 h-4 text-blue-400" /> Users
          </h2>
          {loading && users.length === 0 ? (
            <p className="text-sm text-slate-400">Loading…</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-xs text-slate-400 text-left">
                  <tr>
                    <th className="py-2 pr-4">User</th>
                    <th className="py-2 pr-4">Email</th>
                    <th className="py-2 pr-4">Role</th>
                    <th className="py-2 pr-4">Email verified</th>
                    <th className="py-2 pr-4">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user) => (
                    <tr key={user.id} className="border-t border-white/5">
                      <td className="py-2 pr-4">
                        <div className="font-medium">{user.username}</div>
                        <div className="text-xs text-slate-500">{user.full_name}</div>
                      </td>
                      <td className="py-2 pr-4 text-slate-300">{user.email}</td>
                      <td className="py-2 pr-4">
                        <select
                          value={user.role}
                          onChange={(e) =>
                            act(
                              `/api/admin/users/${encodeURIComponent(user.id)}/role`,
                              { role: e.target.value },
                              `${user.username} is now ${e.target.value}`
                            )
                          }
                          className="bg-slate-900 border border-white/10 rounded-lg px-2 py-1 text-xs"
                          aria-label={`Role for ${user.username}`}
                        >
                          {ROLES.map((role) => (
                            <option key={role} value={role}>
                              {role}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="py-2 pr-4">
                        {user.email_verified ? (
                          <span className="text-emerald-400 text-xs">Verified</span>
                        ) : (
                          <button
                            onClick={() =>
                              act(
                                `/api/admin/users/${encodeURIComponent(user.id)}/verify-email`,
                                {},
                                `${user.username}'s email marked verified`
                              )
                            }
                            className="text-xs text-amber-300 hover:underline"
                          >
                            Not verified · mark verified
                          </button>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-xs text-slate-400">
                        {user.created_at ? new Date(user.created_at).toLocaleString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="glass-panel p-5">
          <h2 className="font-semibold flex items-center gap-2 mb-4">
            <ScrollText className="w-4 h-4 text-purple-400" /> Audit log
            <span className="text-xs font-normal text-slate-500">latest {events.length} events</span>
          </h2>
          {events.length === 0 ? (
            <p className="text-sm text-slate-400">{loading ? "Loading…" : "No events recorded yet."}</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-xs text-slate-400 text-left">
                  <tr>
                    <th className="py-2 pr-4">Time</th>
                    <th className="py-2 pr-4">Event</th>
                    <th className="py-2 pr-4">User</th>
                    <th className="py-2 pr-4">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event, index) => (
                    <tr key={`${event.timestamp}-${index}`} className="border-t border-white/5 align-top">
                      <td className="py-2 pr-4 text-xs text-slate-400 whitespace-nowrap">
                        {new Date(event.timestamp).toLocaleString()}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">{event.event_type}</td>
                      <td className="py-2 pr-4 text-xs">{userName(event.user_id)}</td>
                      <td className="py-2 pr-4 text-xs text-slate-400 font-mono break-all">
                        {Object.keys(event.details ?? {}).length ? JSON.stringify(event.details) : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
