"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getRecommendations, searchItems, getUser, retrainModels } from "@/lib/api";
import type { SampleUser, UserProfile, RecommendationRow, RecommendationResponse, ItemOut, SearchResponse } from "@/types";
import { RefreshCw, Search, Star, ChevronLeft, ChevronRight, ChevronDown, X, LogOut } from "lucide-react";

export default function RecommendationsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const userId = Number(searchParams.get("user")) || 0;

  const [user, setUser] = useState<UserProfile | null>(null);
  const [rows, setRows] = useState<RecommendationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRetrain, setLastRetrain] = useState<{ models: string[]; elapsed_s: number } | null>(null);

  const [ratedOpen, setRatedOpen] = useState(false);

  // Search state
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ItemOut[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);

  const fetchRecs = useCallback(async (uid: number) => {
    try {
      const res = await getRecommendations(uid);
      setRows(res.rows);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const fetchUserById = useCallback(async (uid: number) => {
    try {
      const userData = await getUser(uid);
      setUser(userData);
      localStorage.setItem("crossrec_user", JSON.stringify(userData));
    } catch (e) {
      console.error("Failed to fetch user:", e);
      // Clear user if not found
      setUser(null);
      localStorage.removeItem("crossrec_user");
    }
  }, []);

  useEffect(() => {
    if (!userId) return;
    setLoading(true);
    Promise.all([
      fetchRecs(userId),
      fetchUserById(userId),
    ]).finally(() => setLoading(false));
  }, [userId, fetchRecs, fetchUserById]);

  // If a rating was submitted elsewhere (item page), refresh recommendations
  // so MF-style models (no retrain) at least exclude the newly rated item.
  useEffect(() => {
    if (!userId) return;

    const maybeRefresh = async () => {
      try {
        const raw = localStorage.getItem("crossrec_needs_refresh");
        if (!raw) return;
        const parsed = JSON.parse(raw) as { userId?: number; ts?: number };
        if (parsed?.userId !== userId) return;
        localStorage.removeItem("crossrec_needs_refresh");
        await Promise.all([fetchRecs(userId), fetchUserById(userId)]);
      } catch {
        // ignore
      }
    };

    void maybeRefresh();

    const onStorage = (e: StorageEvent) => {
      if (e.key === "crossrec_needs_refresh") void maybeRefresh();
    };
    // visibilitychange: tab switch / window minimize+restore
    const onVisibility = () => {
      if (document.visibilityState === "visible") void maybeRefresh();
    };
    // popstate: browser back/forward button and router.back()
    const onPopState = () => void maybeRefresh();
    // focus: window regains focus
    const onFocus = () => void maybeRefresh();
    // Poll every 1s as a catch-all for Next.js SPA navigation that fires no events
    const interval = setInterval(() => {
      if (localStorage.getItem("crossrec_needs_refresh")) void maybeRefresh();
    }, 1000);

    window.addEventListener("storage", onStorage);
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("popstate", onPopState);
    window.addEventListener("focus", onFocus);
    return () => {
      clearInterval(interval);
      window.removeEventListener("storage", onStorage);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("popstate", onPopState);
      window.removeEventListener("focus", onFocus);
    };
  }, [userId, fetchRecs, fetchUserById]);

  const handleRefresh = async () => {
    // Step 1: retrain fast models (MF SGD, ~5s) on updated ratings
    setRetraining(true);
    try {
      const result = await retrainModels();
      setLastRetrain({ models: result.models, elapsed_s: result.elapsed_s });
    } catch (e) {
      console.error("Retrain failed:", e);
    }
    setRetraining(false);

    // Step 2: fetch fresh recommendations using the new embeddings
    setRefreshing(true);
    await Promise.all([fetchRecs(userId), fetchUserById(userId)]);
    setRefreshing(false);
  };

  const handleSearch = async (q: string) => {
    setSearchQuery(q);
    if (q.length < 2) { setSearchResults([]); return; }
    setSearchLoading(true);
    try {
      const res = await searchItems(q, 15, userId);
      setSearchResults(res.items);
    } catch {} finally { setSearchLoading(false); }
  };

  const handleLogout = () => {
    localStorage.removeItem("crossrec_user");
    router.push("/");
  };

  if (!userId) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-gray-400">No user selected</p>
          <Link href="/" className="mt-4 inline-block text-indigo-400 hover:underline">Go back to select a user</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-12">
      {/* Top Nav */}
      <nav className="sticky top-0 z-40 border-b border-gray-800/50 bg-[#0a0a0f]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Link href={`/recommendations?user=${userId}`} className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              CrossRec
            </Link>
            {user && (
              <div className="flex items-center gap-2">
                <span className="text-xl">{user.avatar}</span>
                <span className="text-sm font-medium text-gray-300">{user.name}</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSearchOpen(!searchOpen)}
              className="rounded-lg p-2 text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
            >
              <Search className="h-5 w-5" />
            </button>
            <div className="flex items-center gap-2">
              {lastRetrain && !retraining && !refreshing && (
                <span className="rounded-full bg-green-500/20 px-2 py-0.5 text-[10px] font-medium text-green-400">
                  {lastRetrain.models.join(", ").toUpperCase()} retrained ({lastRetrain.elapsed_s}s)
                </span>
              )}
              <button
                onClick={handleRefresh}
                disabled={retraining || refreshing}
                className="flex items-center gap-1.5 rounded-lg border border-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-800 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${(retraining || refreshing) ? "animate-spin" : ""}`} />
                {retraining ? "Retraining MF…" : refreshing ? "Refreshing…" : "Refresh"}
              </button>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 rounded-lg border border-gray-700 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-red-900/30 hover:border-red-700/50 hover:text-red-300 transition-colors"
            >
              <LogOut className="h-3.5 w-3.5" />
              Log Out
            </button>
          </div>
        </div>
      </nav>

      {/* Search Overlay */}
      {searchOpen && (
        <div className="sticky top-[57px] z-30 border-b border-gray-800 bg-[#12121a] px-4 py-3">
          <div className="mx-auto flex max-w-3xl items-center gap-3">
            <Search className="h-5 w-5 shrink-0 text-gray-500" />
            <input
              autoFocus
              type="text"
              placeholder="Search movies & games..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="flex-1 bg-transparent text-sm text-white placeholder-gray-500 outline-none"
            />
            <button onClick={() => { setSearchOpen(false); setSearchQuery(""); setSearchResults([]); }}>
              <X className="h-5 w-5 text-gray-500 hover:text-white" />
            </button>
          </div>
          {searchResults.length > 0 && (
            <div className="mx-auto mt-3 max-w-3xl space-y-1 max-h-80 overflow-y-auto">
              {searchResults.map((item) => (
                <Link key={item.idx} href={`/item/${item.external_id}?user=${userId}`} className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-[#1a1a2e]">
                  <span className="text-lg">{item.domain === "movie" ? "🎬" : "🎮"}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{item.title}</p>
                    <p className="text-xs text-gray-500 truncate">{item.genres}</p>
                  </div>
                  {item.avg_rating != null && (
                    <div className="flex items-center gap-1 text-xs text-gray-500">
                      <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                      {item.avg_rating.toFixed(1)}
                    </div>
                  )}
                  {item.score > 0 && (
                    <span className="flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">
                      Your rating: {item.score.toFixed(0)}★
                    </span>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Past Rated Items — collapsible */}
      {user?.ratings && user.ratings.length > 0 && (
        <div className="mx-auto mt-4 max-w-7xl px-4 sm:px-6">
          <button
            onClick={() => setRatedOpen((o) => !o)}
            className="flex w-full items-center justify-between rounded-xl border border-gray-800 bg-[#12121a] px-4 py-3 text-left transition-colors hover:bg-[#1a1a2e]"
          >
            <div className="flex items-center gap-2">
              <span className="text-base">⭐</span>
              <span className="text-sm font-semibold text-gray-200">Your Rated Items</span>
              <span className="rounded-full bg-gray-800 px-2 py-0.5 text-[10px] text-gray-400">
                {user.ratings.length}
              </span>
            </div>
            <ChevronDown className={`h-4 w-4 text-gray-500 transition-transform ${ratedOpen ? "rotate-180" : ""}`} />
          </button>

          {ratedOpen && (
            <div className="mt-2 rounded-xl border border-gray-800 bg-[#0d0d15] divide-y divide-gray-800/60">
              {user.ratings.map((r) => (
                <Link
                  key={r.item_id}
                  href={`/item/${r.external_id || r.item_id}?user=${userId}`}
                  className="flex items-center gap-3 px-4 py-2.5 hover:bg-[#1a1a2e] transition-colors first:rounded-t-xl last:rounded-b-xl"
                >
                  <span className="text-base shrink-0">{r.domain === "movie" ? "🎬" : "🎮"}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-200 truncate">{r.title}</p>
                    {r.genres && <p className="text-[10px] text-gray-500 truncate">{r.genres}</p>}
                  </div>
                  <div className="flex shrink-0 items-center gap-0.5">
                    {[1,2,3,4,5].map((s) => (
                      <Star key={s} className={`h-3 w-3 ${s <= r.rating ? "fill-amber-400 text-amber-400" : "text-gray-700"}`} />
                    ))}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recommendation Rows */}
      <div className="mt-6 space-y-8">
        {loading ? (
          <div className="flex justify-center py-24">
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          </div>
        ) : (
          rows.map((row) => (
            <RecommendationRowSection key={row.key} row={row} userId={userId} />
          ))
        )}
      </div>
    </div>
  );
}

/* ── Recommendation Row ─────────────────────────────────────── */

function DomainLane({ items, userId, label, emoji, domain }: {
  items: ItemOut[];
  userId: number;
  label: string;
  emoji: string;
  domain: "movie" | "game";
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (dir: "left" | "right") => {
    if (!scrollRef.current) return;
    const amount = scrollRef.current.clientWidth * 0.75;
    scrollRef.current.scrollBy({ left: dir === "left" ? -amount : amount, behavior: "smooth" });
  };

  if (items.length === 0) return null;

  const bg = domain === "movie"
    ? "bg-gradient-to-r from-blue-950/30 via-blue-950/10 to-transparent border-l-2 border-blue-500/40"
    : "bg-gradient-to-r from-purple-950/30 via-purple-950/10 to-transparent border-l-2 border-purple-500/40";

  return (
    <div className={`rounded-lg px-3 py-3 ${bg}`}>
      <p className={`mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide ${
        domain === "movie" ? "text-blue-400" : "text-purple-400"
      }`}>
        <span>{emoji}</span> {label}
        <span className={`ml-1 rounded-full px-1.5 py-0.5 text-[10px] font-normal ${
          domain === "movie" ? "bg-blue-500/20 text-blue-300" : "bg-purple-500/20 text-purple-300"
        }`}>
          {items.length}
        </span>
      </p>
      <div className="group relative">
        <button
          onClick={() => scroll("left")}
          className="absolute -left-1 top-1/2 z-10 -translate-y-1/2 rounded-full bg-black/70 p-2 text-white opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <div ref={scrollRef} className="flex gap-3 overflow-x-auto scrollbar-hide pb-2">
          {items.map((item) => (
            <ItemCard key={`${item.external_id}-${item.idx}`} item={item} userId={userId} />
          ))}
        </div>
        <button
          onClick={() => scroll("right")}
          className="absolute -right-1 top-1/2 z-10 -translate-y-1/2 rounded-full bg-black/70 p-2 text-white opacity-0 transition-opacity group-hover:opacity-100 hover:bg-black"
        >
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
}

function RecommendationRowSection({ row, userId }: { row: RecommendationRow; userId: number }) {
  const movies = row.items.filter((it) => it.domain === "movie");
  const games  = row.items.filter((it) => it.domain === "game" || !it.domain);

  const rowIcon: Record<string, string> = {
    lightgcn_cooc: "🧠", cdr_transfer: "🌉", cooc: "🔗",
    sbert_games: "🎯", sbert_movies: "🎬", popular: "🔥",
  };

  // Determine dominant domain for row background tint
  const dominantDomain = movies.length > games.length ? "movie" : "game";
  const sectionBg = dominantDomain === "movie"
    ? "bg-blue-950/10 border border-blue-900/20"
    : "bg-purple-950/10 border border-purple-900/20";

  return (
    <section className="px-4 sm:px-6">
      <div className={`mx-auto max-w-7xl rounded-xl p-4 ${sectionBg}`}>
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xl">{rowIcon[row.key] || "📌"}</span>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-bold text-white">{row.title}</h2>
              {row.model_tag && (
                <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-[10px] font-medium text-indigo-300 border border-indigo-500/30">
                  {row.model_tag}
                </span>
              )}
            </div>
            <p className="text-xs text-gray-500">{row.subtitle}</p>
          </div>
        </div>

        <div className="space-y-3">
          <DomainLane items={movies} userId={userId} label="Movies" emoji="🎬" domain="movie" />
          <DomainLane items={games}  userId={userId} label="Games"  emoji="🎮" domain="game" />
        </div>
      </div>
    </section>
  );
}

/* ── Item Card ──────────────────────────────────────────────── */

function ItemCard({ item, userId }: { item: ItemOut; userId: number }) {
  const [hovered, setHovered] = useState(false);

  return (
    <Link href={`/item/${item.external_id}?user=${userId}`}>
      <div
        className={`card-hover group relative w-[160px] shrink-0 cursor-pointer overflow-hidden rounded-lg bg-[#12121a] sm:w-[180px] ${
          item.domain === "movie"
            ? "border border-blue-800/40 hover:border-blue-500/60"
            : "border border-purple-800/40 hover:border-purple-500/60"
        }`}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Image */}
        <div className="relative aspect-[3/4] w-full overflow-hidden bg-gray-900">
          {item.image_url ? (
            <img src={item.image_url} alt={item.title} className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full items-center justify-center text-4xl text-gray-700">
              {item.domain === "movie" ? "🎬" : "🎮"}
            </div>
          )}
          <span className={`absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
            item.domain === "movie" ? "bg-blue-600/90 text-blue-100" : "bg-purple-600/90 text-purple-100"
          }`}>
            {item.domain}
          </span>
        </div>

        {/* Title */}
        <div className="p-2.5">
          <p className="line-clamp-2 text-xs font-semibold text-gray-200 leading-tight">{item.title}</p>
          {item.genres && (
            <p className="mt-1 line-clamp-1 text-[10px] text-gray-500">{item.genres.slice(0, 50)}</p>
          )}
          {item.avg_rating != null && (
            <div className="mt-1 flex items-center gap-1">
              <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
              <span className="text-[10px] text-gray-400">{item.avg_rating.toFixed(1)}</span>
            </div>
          )}
        </div>

        {/* Hover overlay */}
        {hovered && (
          <div className="absolute inset-0 flex flex-col justify-end bg-gradient-to-t from-black/95 via-black/60 to-transparent p-3">
            <p className="text-xs font-semibold text-white">{item.title}</p>
            {item.genres && <p className="mt-0.5 text-[10px] text-gray-300">{item.genres.slice(0, 60)}</p>}
            {item.description && <p className="mt-1 line-clamp-2 text-[10px] text-gray-400">{item.description}</p>}
            {item.reason && (
              <p className="mt-1.5 rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] text-indigo-300 italic">
                {item.reason.slice(0, 80)}
              </p>
            )}
          </div>
        )}
      </div>
    </Link>
  );
}
