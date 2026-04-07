"use client";

import { useRouter } from "next/navigation";
import { Film, Gamepad2, Sparkles, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { getUsers, createUser } from "@/lib/api";
import type { SampleUser } from "@/types";

export default function HomePage() {
  const router = useRouter();
  const [users, setUsers] = useState<SampleUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    getUsers(20, 30)
      .then(setUsers)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const selectUser = (user: SampleUser) => {
    localStorage.setItem("crossrec_user", JSON.stringify(user));
    router.push(`/recommendations?user=${user.id}`);
  };

  const handleCreateUser = async () => {
    const name = newName.trim();
    if (!name || creating) return;
    setCreating(true);
    try {
      const user = await createUser(name);
      setNewName("");
      selectUser(user);
    } catch {
      alert("Failed to create user");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="relative overflow-hidden px-6 pt-16 pb-20">
        <div className="absolute inset-0 bg-gradient-to-b from-indigo-950/40 via-transparent to-transparent" />
        <div className="relative mx-auto max-w-5xl text-center">
          <div className="mb-6 flex items-center justify-center gap-3">
            <Film className="h-10 w-10 text-blue-400" />
            <span className="text-3xl font-thin text-gray-500">/</span>
            <Gamepad2 className="h-10 w-10 text-purple-400" />
          </div>
          <h1 className="mb-4 text-5xl font-extrabold tracking-tight sm:text-6xl">
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent">
              CrossRec
            </span>
          </h1>
          <p className="mx-auto mb-3 max-w-2xl text-xl text-gray-300">
            Hybrid Movie & Game Recommender
          </p>
          <p className="mx-auto max-w-xl text-sm text-gray-500">
            Discover movies and games tailored to your taste — powered by multiple
            recommendation signals fused into personalized rows.
          </p>
        </div>
      </section>

      {/* Quick Create User */}
      <section className="mx-auto max-w-md px-6 pb-12">
        <div className="rounded-xl border border-gray-800 bg-[#12121a] p-6">
          <h2 className="mb-3 text-center text-lg font-bold text-white">Start Fresh</h2>
          <p className="mb-4 text-center text-xs text-gray-500">
            Enter your name to get started — no password needed
          </p>
          <form
            onSubmit={(e) => { e.preventDefault(); handleCreateUser(); }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Your name"
              maxLength={50}
              className="flex-1 rounded-lg border border-gray-700 bg-[#0a0a14] px-4 py-2.5 text-sm text-white placeholder-gray-500 outline-none focus:border-indigo-500 transition-colors"
            />
            <button
              type="submit"
              disabled={!newName.trim() || creating}
              className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 px-5 py-2.5 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              {creating ? (
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              Go
            </button>
          </form>
        </div>
      </section>

      {/* User Selection */}
      <section className="mx-auto max-w-5xl px-6 pb-24">
        <h2 className="mb-2 text-center text-2xl font-bold">Or Choose a Sample User</h2>
        <p className="mb-8 text-center text-sm text-gray-500">
          Each user has a unique taste profile and rating history
        </p>

        {loading ? (
          <div className="flex justify-center py-16">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {users.map((user) => (
              <button
                key={user.id}
                onClick={() => selectUser(user)}
                className="group relative flex items-start gap-4 rounded-xl border border-gray-800 bg-[#12121a] p-5 text-left transition-all hover:border-indigo-500/50 hover:bg-[#1a1a2e]"
              >
                <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 text-2xl">
                  {user.avatar}
                </span>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-white">{user.name}</h3>
                  <p className="text-sm text-indigo-300">{user.taste_summary}</p>
                  <div className="mt-2 flex items-center gap-3 text-xs text-gray-500">
                    <span>{user.total_ratings} ratings</span>
                    <span>avg {user.avg_rating.toFixed(1)}★</span>
                  </div>
                </div>
                <ChevronRight className="mt-1 h-5 w-5 shrink-0 text-gray-600 transition-colors group-hover:text-indigo-400" />
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
