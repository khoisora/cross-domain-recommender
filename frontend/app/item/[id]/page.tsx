"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import dynamic from "next/dynamic";
import { getItem, getItemGraph, getItemExplanation, submitRating, searchItems, getSimilarItems } from "@/lib/api";
import type { ItemDetail, GraphResponse, GraphNode, GraphEdge, ExplanationResponse, ItemOut } from "@/types";
import { ArrowLeft, Star, Home, Info, Network, Tag, Search, X } from "lucide-react";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function ItemDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const itemId = String(params.id); // external_id (e.g. ASIN)
  const userId = Number(searchParams.get("user")) || 0;

  const [item, setItem] = useState<ItemDetail | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [sbertSimilar, setSbertSimilar] = useState<ItemOut[]>([]);
  const [crossDomainSimilar, setCrossDomainSimilar] = useState<ItemOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRating, setUserRating] = useState(0);
  const [ratingMsg, setRatingMsg] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<ItemOut[]>([]);

  const handleSearch = async (q: string) => {
    setSearchQuery(q);
    if (q.length < 2) { setSearchResults([]); return; }
    try {
      const res = await searchItems(q, 15, userId);
      setSearchResults(res.items);
    } catch {}
  };

  useEffect(() => {
    if (!itemId) return;
    setLoading(true);
    setSbertSimilar([]);
    setCrossDomainSimilar([]);
    getItem(itemId).then((it) => {
      setItem(it);
      // SBERT similar items are included in the item detail response
      if (it?.similar_items) {
        setSbertSimilar(it.similar_items.map((s: any) => ({
          idx: 0, external_id: s.external_id, title: s.title,
          domain: s.domain, image_url: s.image_url || "",
          avg_rating: s.avg_rating, rating_count: 0,
          score: s.similarity, reason: `${(s.similarity * 100).toFixed(0)}% similar (SBERT)`,
          description: "", genres: "", tags: "", year: "",
        })));
      }
    }).catch(() => null).finally(() => setLoading(false));
  }, [itemId, userId]);

  const handleRate = async (rating: number) => {
    if (!userId) return;
    setUserRating(rating);
    try {
      const res = await submitRating({ user_id: userId, external_id: itemId, rating });
      setRatingMsg(res.message);
      setTimeout(() => setRatingMsg(""), 3000);
      try {
        localStorage.setItem(
          "crossrec_needs_refresh",
          JSON.stringify({ userId, ts: Date.now() }),
        );
      } catch {}
    } catch (e: any) {
      setRatingMsg(e.message || "Failed");
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-10 w-10 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
      </div>
    );
  }

  if (!item) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-gray-400">Item not found</p>
      </div>
    );
  }

  const tagsList = item.tags
    ? item.tags.split(",").map((t) => t.trim()).filter(Boolean).slice(0, 10)
    : [];


  return (
    <div className="min-h-screen pb-16">
      {/* Nav */}
      <nav className="sticky top-0 z-40 border-b border-gray-800/50 bg-[#0a0a0f]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Link href={userId ? `/recommendations?user=${userId}` : "/"} className="text-gray-400 hover:text-white"><Home className="h-5 w-5" /></Link>
            <button onClick={() => router.back()} className="flex items-center gap-1 text-sm text-gray-400 hover:text-white">
              <ArrowLeft className="h-4 w-4" /> Back
            </button>
          </div>
          <button onClick={() => setSearchOpen(!searchOpen)} className="rounded-lg p-2 text-gray-400 hover:bg-gray-800 hover:text-white transition-colors">
            <Search className="h-5 w-5" />
          </button>
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
              {searchResults.map((sr) => (
                <Link key={sr.idx} href={`/item/${sr.idx}?user=${userId}`} onClick={() => setSearchOpen(false)} className="flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-[#1a1a2e]">
                  <span className="text-lg">{sr.domain === "movie" ? "🎬" : "🎮"}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{sr.title}</p>
                    <p className="text-xs text-gray-500 truncate">{sr.genres}</p>
                  </div>
                  {sr.avg_rating != null && (
                    <div className="flex items-center gap-1 text-xs text-gray-500">
                      <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                      {sr.avg_rating.toFixed(1)}
                    </div>
                  )}
                  {sr.score > 0 && (
                    <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-xs text-amber-300">
                      Your rating: {sr.score.toFixed(0)}★
                    </span>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="mx-auto max-w-6xl px-4 pt-8">
        {/* Item Header */}
        <div className="flex flex-col gap-8 md:flex-row">
          {/* Poster */}
          <div className="shrink-0">
            <div className="relative h-80 w-56 overflow-hidden rounded-xl bg-gray-900 shadow-2xl shadow-indigo-500/10">
              {item.image_url ? (
                <img src={item.image_url} alt={item.title} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center text-6xl text-gray-700">
                  {item.domain === "movie" ? "🎬" : "🎮"}
                </div>
              )}
              <span className={`absolute left-2 top-2 rounded px-2 py-0.5 text-xs font-bold uppercase ${
                item.domain === "movie" ? "bg-blue-600 text-blue-100" : "bg-purple-600 text-purple-100"
              }`}>
                {item.domain}
              </span>
            </div>
          </div>

          {/* Info */}
          <div className="flex-1 space-y-4">
            <h1 className="text-3xl font-bold text-white">{item.title}</h1>

            {item.avg_rating != null && (
              <div className="flex items-center gap-2">
                <Star className="h-5 w-5 fill-amber-400 text-amber-400" />
                <span className="text-xl font-bold text-white">{item.avg_rating.toFixed(1)}</span>
                <span className="text-sm text-gray-500">({item.rating_count} ratings)</span>
              </div>
            )}

            {/* Rate — placed right below avg rating */}
            {userId > 0 && (
              <div className="rounded-xl border border-gray-800 bg-[#12121a] p-4">
                <h3 className="mb-2 text-sm font-semibold text-gray-300">
                  Rate this {item.domain === "movie" ? "Movie" : "Game"}
                </h3>
                <div className="flex items-center gap-4">
                  <div className="flex gap-1">
                    {[1, 2, 3, 4, 5].map((s) => (
                      <button key={s} onClick={() => handleRate(s)} className="star-btn">
                        <Star className={`h-6 w-6 transition-colors ${s <= userRating ? "fill-amber-400 text-amber-400" : "text-gray-600 hover:text-amber-300"}`} />
                      </button>
                    ))}
                  </div>
                  {userRating > 0 && <span className="text-sm text-amber-400 font-medium">{userRating}/5</span>}
                  {ratingMsg && <span className="text-xs text-indigo-400">{ratingMsg}</span>}
                </div>
              </div>
            )}

            {item.genres && (
              <div className="flex flex-wrap gap-2">
                {item.genres.split(",").slice(0, 8).map((g) => (
                  <span key={g.trim()} className="rounded-full border border-gray-700 bg-gray-800/50 px-3 py-1 text-xs text-gray-300">
                    {g.trim()}
                  </span>
                ))}
              </div>
            )}

            {tagsList.length > 0 && (
              <div>
                <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-gray-400">
                  <Tag className="h-3.5 w-3.5" /> Tags / Features
                </h3>
                <div className="flex flex-wrap gap-2">
                  {tagsList.map((t) => (
                    <span key={t} className="rounded-full border border-indigo-800/50 bg-indigo-950/40 px-3 py-1 text-xs text-indigo-300">
                      {t.length > 60 ? t.slice(0, 57) + "..." : t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {item.description && (
              <p className="max-w-2xl leading-relaxed text-gray-400">{item.description}</p>
            )}

            {/* Explanation Summary */}
            {explanation && explanation.reasons.length > 0 && (
              <div className="rounded-xl border border-gray-800 bg-[#12121a] p-4">
                <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-gray-300">
                  <Info className="h-4 w-4 text-indigo-400" /> Why This Is Recommended
                </h3>
                <p className="mb-3 text-sm text-indigo-300">{explanation.summary}</p>
                <ul className="space-y-1.5">
                  {explanation.reasons.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-gray-400">
                      <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
                      {r.text}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>


        {/* SBERT Semantically Similar Items */}
        {sbertSimilar.length > 0 && (
          <div className="mt-10">
            <div className="mb-4 flex items-center gap-3">
              <h2 className="text-xl font-bold text-white">Semantically Similar</h2>
              <span className="rounded-full border border-indigo-800/50 bg-indigo-950/40 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-indigo-400">
                SBERT
              </span>
            </div>
            <p className="mb-4 text-xs text-gray-500">
              Items with similar descriptions, genres, and themes — powered by sentence embeddings
            </p>
            <SimilarGrid items={sbertSimilar} userId={userId} />
          </div>
        )}

        {/* Cross-Domain Similar Items */}
        {crossDomainSimilar.length > 0 && item && (
          <div className="mt-10">
            <div className="mb-4 flex items-center gap-3">
              <h2 className="text-xl font-bold text-white">
                {item.domain === "movie" ? "🎮 Similar Games" : "🎬 Similar Movies"}
              </h2>
              <span className="rounded-full border border-emerald-800/50 bg-emerald-950/40 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-400">
                Cross-Domain
              </span>
            </div>
            <p className="mb-4 text-xs text-gray-500">
              {item.domain === "movie"
                ? "Games with themes and vibes similar to this movie"
                : "Movies with themes and vibes similar to this game"}
              {" "}— semantically matched across domains
            </p>
            <SimilarGrid items={crossDomainSimilar} userId={userId} />
          </div>
        )}

      </div>
    </div>
  );
}

/* ── SimilarGrid ──────────────────────────────────────────────── */

function SimilarGrid({ items, userId }: { items: ItemOut[]; userId: number }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
      {items.map((it) => (
        <Link key={it.idx} href={`/item/${it.idx}?user=${userId}`}>
          <div className="group overflow-hidden rounded-lg border border-gray-800 bg-[#12121a] transition-all hover:border-indigo-500/50 hover:bg-[#1a1a2e]">
            <div className="relative aspect-[3/4] w-full overflow-hidden bg-gray-900">
              {it.image_url ? (
                <img src={it.image_url} alt={it.title} className="h-full w-full object-cover" />
              ) : (
                <div className="flex h-full items-center justify-center text-4xl text-gray-700">
                  {it.domain === "movie" ? "🎬" : "🎮"}
                </div>
              )}
              <span className={`absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                it.domain === "movie" ? "bg-blue-600/90 text-blue-100" : "bg-purple-600/90 text-purple-100"
              }`}>
                {it.domain}
              </span>
              <span className="absolute right-1.5 top-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-indigo-300">
                {(it.score * 100).toFixed(0)}% match
              </span>
            </div>
            <div className="p-2.5">
              <p className="line-clamp-2 text-xs font-semibold text-gray-200 leading-tight">{it.title}</p>
              {it.genres && (
                <p className="mt-1 line-clamp-1 text-[10px] text-gray-500">{it.genres.slice(0, 50)}</p>
              )}
              {it.avg_rating != null && (
                <div className="mt-1 flex items-center gap-1">
                  <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                  <span className="text-[10px] text-gray-400">{it.avg_rating.toFixed(1)}</span>
                </div>
              )}
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
}

/* ── Helpers ──────────────────────────────────────────────────── */

function timeAgo(ts: string | undefined): string {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) {
      const num = Number(ts);
      if (!isNaN(num)) {
        const ms = num > 1e12 ? num : num * 1000;
        const diff = Date.now() - ms;
        return formatDiff(diff);
      }
      return "";
    }
    return formatDiff(Date.now() - d.getTime());
  } catch {
    return "";
  }
}

function formatDiff(ms: number): string {
  const secs = Math.abs(ms) / 1000;
  if (secs < 60) return "just now";
  const mins = secs / 60;
  if (mins < 60) return `${Math.floor(mins)}m ago`;
  const hrs = mins / 60;
  if (hrs < 24) return `${Math.floor(hrs)}h ago`;
  const days = hrs / 24;
  if (days < 30) return `${Math.floor(days)}d ago`;
  const months = days / 30;
  if (months < 12) return `${Math.floor(months)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

function nodeColor(n: GraphNode): string {
  if (n.type === "center") return "#6366f1";
  if (n.type === "rated") return "#f59e0b";
  return n.domain === "movie" ? "#3b82f6" : "#8b5cf6";
}

/* ── Interactive Force Graph ───────────────────────────────── */

interface ForceNode extends GraphNode {
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
  __img?: HTMLImageElement;
  __imgLoaded?: boolean;
}

function InteractiveGraph({
  graph,
  currentItemIdx,
  userId,
  router,
}: {
  graph: GraphResponse;
  currentItemIdx: number;
  userId: number;
  router: ReturnType<typeof useRouter>;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [hovered, setHovered] = useState<ForceNode | null>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 700 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      if (width > 100) setDimensions({ width, height: 700 });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Increase charge strength after mount to spread nodes apart
  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force("charge")?.strength(-400).distanceMax(500);
      fgRef.current.d3Force("link")?.distance(120);
    }
  }, [graph]);

  const graphData = useMemo(() => {
    const nodes: ForceNode[] = graph.nodes.map((n) => ({ ...n }));
    const links = graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
      label: e.label,
    }));
    return { nodes, links };
  }, [graph]);

  // Preload images for nodes
  useEffect(() => {
    for (const n of graphData.nodes) {
      if (n.image_url && !n.__img) {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => { n.__imgLoaded = true; };
        img.src = n.image_url;
        n.__img = img;
      }
    }
  }, [graphData.nodes]);

  const paintNode = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as ForceNode;
      const x = n.x ?? 0;
      const y = n.y ?? 0;
      const r = n.type === "center" ? 26 : 20;
      const col = nodeColor(n);
      const isHovered = hovered?.id === n.id;

      // Outer glow
      if (n.type === "center" || n.type === "rated" || isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, r + 6, 0, 2 * Math.PI);
        ctx.fillStyle = n.type === "rated"
          ? "rgba(245,158,11,0.18)"
          : n.type === "center"
          ? "rgba(99,102,241,0.18)"
          : "rgba(255,255,255,0.08)";
        ctx.fill();
      }

      // Clip circle for image
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.clip();

      // Draw image or fallback
      if (n.__img && n.__imgLoaded) {
        ctx.drawImage(n.__img, x - r, y - r, r * 2, r * 2);
      } else {
        ctx.fillStyle = col + "33";
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
        ctx.font = `${r}px serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(n.domain === "movie" ? "🎬" : "🎮", x, y);
      }
      ctx.restore();

      // Border ring
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.strokeStyle = col;
      ctx.lineWidth = isHovered ? 4 : n.type === "center" ? 3 : n.type === "rated" ? 2.5 : 1.5;
      ctx.stroke();

      // Label below
      const label = n.label.length > 25 ? n.label.slice(0, 22) + "..." : n.label;
      const fontSize = Math.max(10, 12 / Math.sqrt(globalScale));
      ctx.font = `${n.type === "center" ? "bold " : ""}${fontSize}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";

      // Text background for readability
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(10,10,15,0.8)";
      ctx.fillRect(x - textWidth / 2 - 2, y + r + 2, textWidth + 4, fontSize + 4);

      ctx.fillStyle = n.type === "rated" ? "#fbbf24" : "#d1d5db";
      ctx.fillText(label, x, y + r + 4);

      // Rating badge for rated nodes
      if (n.user_rating) {
        const badgeX = x + r - 2;
        const badgeY = y - r + 2;
        ctx.beginPath();
        ctx.arc(badgeX, badgeY, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "#f59e0b";
        ctx.fill();
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#000";
        ctx.fillText(`${n.user_rating}`, badgeX, badgeY);
      }
    },
    [hovered],
  );

  const handleNodeClick = useCallback(
    (node: any) => {
      const n = node as ForceNode;
      if (n.id !== currentItemIdx) {
        router.push(`/item/${n.id}?user=${userId}`);
      }
    },
    [currentItemIdx, userId, router],
  );

  // Persist drag position — don't snap back
  const handleNodeDragEnd = useCallback((node: any) => {
    node.fx = node.x;
    node.fy = node.y;
  }, []);

  // Compute min/max weight for edge scaling
  const weightRange = useMemo(() => {
    const weights = graphData.links.map((l) => l.weight);
    return { min: Math.min(...weights, 0.1), max: Math.max(...weights, 1) };
  }, [graphData.links]);

  return (
    <div ref={containerRef} className="relative rounded-xl border border-gray-800 bg-[#0d0d15] p-2">
      {/* Legend */}
      <div className="mb-2 flex flex-wrap gap-4 px-2 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-indigo-500" /> Selected item
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-blue-500" /> Related movie
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-purple-500" /> Related game
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-amber-500" /> Your rated item
        </span>
        <span className="flex items-center gap-1 ml-auto">
          <span className="inline-block h-0.5 w-6 bg-indigo-400/80" /> Strong similarity
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-px w-6 bg-indigo-400/30" /> Weak similarity
        </span>
      </div>

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={dimensions.width - 16}
        height={dimensions.height}
        backgroundColor="#0d0d15"
        nodeRelSize={20}
        nodeCanvasObject={paintNode}
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          const r = (node as ForceNode).type === "center" ? 26 : 20;
          ctx.beginPath();
          ctx.arc(node.x ?? 0, node.y ?? 0, r + 4, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
        }}
        linkColor={(link: any) => {
          const w = link.weight ?? 0.5;
          const alpha = 0.15 + 0.6 * ((w - weightRange.min) / (weightRange.max - weightRange.min + 0.001));
          return `rgba(99,102,241,${alpha.toFixed(2)})`;
        }}
        linkWidth={(link: any) => {
          const w = link.weight ?? 0.5;
          const norm = (w - weightRange.min) / (weightRange.max - weightRange.min + 0.001);
          return 1 + norm * 5;
        }}
        linkDirectionalParticles={0}
        linkLabel={(link: any) => link.label || ""}
        onNodeClick={handleNodeClick}
        onNodeHover={(node: any) => setHovered(node as ForceNode | null)}
        onNodeDragEnd={handleNodeDragEnd}
        cooldownTicks={120}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.2}
        enableZoomrating={true}
        enablePanrating={true}
        enableNodeDrag={true}
      />

      {/* Hover tooltip */}
      {hovered && (
        <div className="pointer-events-none absolute bottom-4 left-4 z-50 max-w-xs rounded-lg border border-gray-700 bg-[#1a1a2e] p-3 shadow-xl">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">{hovered.domain === "movie" ? "🎬" : "🎮"}</span>
            <span className="font-semibold text-sm text-white">{hovered.label}</span>
          </div>
          {hovered.genres && (
            <p className="text-xs text-gray-400 mb-1">
              <span className="text-gray-500">Genres:</span>{" "}
              {hovered.genres.split(",").slice(0, 5).map((g) => g.trim()).join(", ")}
            </p>
          )}
          {hovered.tags && (
            <p className="text-xs text-indigo-300 mb-1">
              <span className="text-gray-500">Tags:</span>{" "}
              {hovered.tags.split(",").slice(0, 3).map((t) => t.trim()).join(", ")}
            </p>
          )}
          {hovered.user_rating != null && (
            <p className="text-xs text-amber-400">
              Your rating: {hovered.user_rating}★
              {hovered.rating_timestamp ? ` · ${timeAgo(hovered.rating_timestamp)}` : ""}
            </p>
          )}
          {hovered.avg_rating != null && (
            <p className="text-xs text-gray-500">Avg: {hovered.avg_rating.toFixed(1)}/5</p>
          )}
        </div>
      )}
    </div>
  );
}
