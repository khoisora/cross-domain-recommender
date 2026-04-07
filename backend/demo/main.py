"""Standalone demo FastAPI app — no database required.

Loads precomputed artifacts into memory and serves:
  - Sample user selection
  - 5-row hybrid recommendations
  - Item search + rating
  - Item detail + explanation graph

Run:  uvicorn app.demo.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Optional

from pathlib import Path as _Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_STATIC_DIR = _Path(__file__).resolve().parent.parent.parent / "static"

from backend.demo.store import DemoStore
from backend.demo.recommender import HybridRecommender
from backend.demo.graph import GraphService
from backend.demo.database import DemoDB

logger = logging.getLogger(__name__)

# ── Pydantic schemas ────────────────────────────────────────────────────────

class SampleUser(BaseModel):
    id: int
    external_id: str
    name: str
    avatar: str
    taste_summary: str
    total_ratings: int
    avg_rating: float
    is_sample: bool = True

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)

class ItemOut(BaseModel):
    idx: int
    external_id: str
    title: str
    domain: str
    genres: str = ""
    tags: str = ""
    image_url: str = ""
    description: str = ""
    avg_rating: Optional[float] = None
    rating_count: int = 0
    year: str = ""
    score: float = 0
    reason: str = ""

class RecommendationRow(BaseModel):
    key: str
    title: str
    subtitle: str
    items: list[ItemOut]

class RecommendationResponse(BaseModel):
    user_id: int
    user_name: str
    rows: list[RecommendationRow]

class RatingRequest(BaseModel):
    user_id: int
    item_idx: int
    rating: float = Field(..., ge=1.0, le=5.0)

class RatingResponse(BaseModel):
    success: bool
    message: str

class GraphNode(BaseModel):
    id: int
    label: str
    domain: str
    type: str
    genres: str = ""
    tags: str = ""
    image_url: str = ""
    avg_rating: Optional[float] = None
    user_rating: Optional[float] = None
    rating_timestamp: Optional[str] = None

class GraphEdge(BaseModel):
    source: int
    target: int
    weight: float
    label: str = ""

class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    explanation_text: str

class ExplanationReason(BaseModel):
    type: str
    text: str
    weight: float = 0
    source_item_idx: Optional[int] = None

class ExplanationResponse(BaseModel):
    reasons: list[ExplanationReason]
    summary: str

class ItemDetail(BaseModel):
    idx: int
    external_id: str
    title: str
    domain: str
    genres: str = ""
    tags: str = ""
    image_url: str = ""
    description: str = ""
    avg_rating: Optional[float] = None
    rating_count: int = 0
    year: str = ""

class SearchResponse(BaseModel):
    items: list[ItemOut]
    total: int

# ── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    store = DemoStore.get()
    store.load()
    # Initialize database and seed data
    db = DemoDB.get()
    db.upsert_sample_users(store.sample_users)
    # Restore runtime ratings from DB into in-memory store
    runtime_ratings = db.get_all_runtime_ratings()
    restored = 0
    for r in runtime_ratings:
        ext_id = r["user_id"]
        item_ext = r["item_id"]
        # Check if this rating already exists in store (from artifacts)
        existing = store.user_ratings.get(ext_id, [])
        already = any(x["item_id"] == item_ext for x in existing)
        if not already:
            store.add_rating(ext_id, item_ext, r["rating"])
            restored += 1
    if restored:
        logger.info("Restored %d runtime ratings from DB into memory", restored)
    yield

app = FastAPI(
    title="CrossRec Demo — Hybrid Recommender Portal",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static frontend (jQuery/jQueryUI) ─────────────────────────────────────
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/index.html")

# ── Dependency helpers ──────────────────────────────────────────────────────

def _store() -> DemoStore:
    return DemoStore.get()

def _db() -> DemoDB:
    return DemoDB.get()

def _recommender() -> HybridRecommender:
    return HybridRecommender(_store())

def _graph() -> GraphService:
    return GraphService(_store())

def _resolve_user(user_id: int) -> dict:
    s = _store()
    # Try sample users first (fast path)
    user = s.sample_users_by_id.get(user_id)
    if user:
        return user
    # Try database for custom users
    db = _db()
    db_user = db.get_user(user_id)
    if not db_user:
        raise HTTPException(404, f"User {user_id} not found")
    # Build a compatible user dict
    rating_count = db.get_user_rating_count(user_id)
    return {
        "id": db_user["id"],
        "external_id": db_user["external_id"],
        "name": db_user["name"],
        "avatar": db_user["avatar"],
        "taste_summary": db_user.get("taste_summary", ""),
        "total_ratings": rating_count,
        "avg_rating": 0.0,
        "is_sample": bool(db_user.get("is_sample", 0)),
        "ratings": [],
    }

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "loaded": _store().loaded}


@app.get("/api/users", response_model=list[SampleUser])
async def list_users():
    """List all users (sample + custom)."""
    db = _db()
    s = _store()
    users = []
    # Sample users from store (richer data)
    for u in s.sample_users:
        users.append(SampleUser(
            id=u["id"],
            external_id=u["external_id"],
            name=u["name"],
            avatar=u["avatar"],
            taste_summary=u["taste_summary"],
            total_ratings=u["total_ratings"],
            avg_rating=u["avg_rating"],
            is_sample=True,
        ))
    # Custom users from DB
    for row in db.list_users():
        if row.get("is_sample"):
            continue  # already added from store
        rc = db.get_user_rating_count(row["id"])
        users.append(SampleUser(
            id=row["id"],
            external_id=row["external_id"],
            name=row["name"],
            avatar=row.get("avatar", "👤"),
            taste_summary=row.get("taste_summary", ""),
            total_ratings=rc,
            avg_rating=0.0,
            is_sample=False,
        ))
    return users


@app.post("/api/users", response_model=SampleUser)
async def create_user(req: CreateUserRequest):
    """Create a new user with just a name."""
    db = _db()
    user = db.create_user(req.name)
    return SampleUser(
        id=user["id"],
        external_id=user["external_id"],
        name=user["name"],
        avatar=user["avatar"],
        taste_summary=user["taste_summary"],
        total_ratings=0,
        avg_rating=0.0,
        is_sample=False,
    )


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get sample user profile with rating history."""
    user = _resolve_user(user_id)
    s = _store()
    # Enrich ratings with item metadata
    ratings = []
    for r in user.get("ratings", [])[:50]:
        item_meta = s.items_by_ext.get(r["item_id"], {})
        ratings.append({
            "item_id": r["item_id"],
            "item_idx": s.sd_item_to_idx.get(r["item_id"]),
            "title": item_meta.get("title", r.get("title", "")),
            "domain": item_meta.get("domain", r.get("domain", "movie")),
            "rating": r["rating"],
            "image_url": item_meta.get("image_url", ""),
            "genres": item_meta.get("genres", ""),
        })
    return {**user, "ratings": ratings}


@app.get("/api/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int):
    """Get 5 recommendation rows for a sample user."""
    user = _resolve_user(user_id)
    rec = _recommender()
    rows = rec.recommend_rows(user["external_id"], k_per_row=15)
    return RecommendationResponse(
        user_id=user_id,
        user_name=user["name"],
        rows=[RecommendationRow(**r) for r in rows],
    )


@app.get("/api/items/search", response_model=SearchResponse)
async def search_items(
    q: str = Query("", min_length=0),
    limit: int = Query(20, ge=1, le=50),
    user_id: Optional[int] = Query(None),
):
    """Search items by title substring. Optionally includes user's ratings."""
    s = _store()
    if not q:
        results = sorted(s.items_list, key=lambda x: x.get("rating_count", 0), reverse=True)[:limit]
    else:
        results = s.search_items(q, limit=limit)

    # Build user rating lookup if user specified
    user_ratings_map: dict[int, float] = {}
    if user_id is not None:
        try:
            user = _resolve_user(user_id)
            ext_id = user["external_id"]
            for r in s.user_ratings.get(ext_id, []):
                idx = s.sd_item_to_idx.get(r["item_id"])
                if idx is not None:
                    user_ratings_map[idx] = r["rating"]
        except Exception:
            pass

    items = [
        ItemOut(
            idx=it.get("idx", 0),
            external_id=it.get("external_id", ""),
            title=it.get("title", ""),
            domain=it.get("domain", "movie"),
            genres=it.get("genres", ""),
            image_url=it.get("image_url", ""),
            description=it.get("description", "")[:150],
            avg_rating=it.get("avg_rating"),
            rating_count=it.get("rating_count", 0),
            year=it.get("year", ""),
            score=user_ratings_map.get(it.get("idx", 0), 0),
        )
        for it in results
    ]
    return SearchResponse(items=items, total=len(items))


@app.get("/api/items/{item_idx}", response_model=ItemDetail)
async def get_item(item_idx: int):
    """Get item details by index."""
    s = _store()
    meta = s.sd_items_by_idx.get(item_idx)
    if not meta:
        raise HTTPException(404, f"Item {item_idx} not found")
    return ItemDetail(
        idx=meta.get("idx", item_idx),
        external_id=meta.get("external_id", ""),
        title=meta.get("title", ""),
        domain=meta.get("domain", "movie"),
        genres=meta.get("genres", ""),
        tags=meta.get("tags", ""),
        image_url=meta.get("image_url", ""),
        description=meta.get("description", ""),
        avg_rating=meta.get("avg_rating"),
        rating_count=meta.get("rating_count", 0),
        year=meta.get("year", ""),
    )


@app.post("/api/ratings", response_model=RatingResponse)
async def submit_rating(req: RatingRequest):
    """Submit a rating and trigger MF online update."""
    user = _resolve_user(req.user_id)
    s = _store()
    db = _db()
    ext_item = s.sd_idx_to_item.get(req.item_idx)
    if not ext_item:
        raise HTTPException(404, f"Item index {req.item_idx} not found")
    meta = s.sd_items_by_idx.get(req.item_idx, {})
    domain = meta.get("domain", "")
    # Persist to DB
    db.add_rating(req.user_id, req.item_idx, ext_item, req.rating)
    # Update in-memory store for immediate recommendation impact (cooc + SBERT refresh instantly)
    s.add_rating(user["external_id"], ext_item, req.rating, domain=domain)
    return RatingResponse(
        success=True,
        message=f"Rating {req.rating:.1f} saved. Recommendations will update.",
    )


@app.get("/api/items/{item_idx}/graph", response_model=GraphResponse)
async def get_item_graph(
    item_idx: int,
    user_id: Optional[int] = Query(None),
):
    """Get local explanation graph centered on an item."""
    s = _store()
    if item_idx not in s.sd_items_by_idx:
        raise HTTPException(404, f"Item {item_idx} not found")
    user_ext = None
    if user_id is not None:
        try:
            user = _resolve_user(user_id)
            user_ext = user["external_id"]
        except Exception:
            pass
    g = _graph()
    result = g.build_item_graph(item_idx, user_ext_id=user_ext)
    return GraphResponse(**result)


@app.get("/api/items/{item_idx}/explanation", response_model=ExplanationResponse)
async def get_item_explanation(
    item_idx: int,
    user_id: Optional[int] = Query(None),
):
    """Get structured explanation for why an item is recommended."""
    s = _store()
    if item_idx not in s.sd_items_by_idx:
        raise HTTPException(404, f"Item {item_idx} not found")
    user_ext = None
    if user_id is not None:
        try:
            user = _resolve_user(user_id)
            user_ext = user["external_id"]
        except Exception:
            pass
    g = _graph()
    result = g.get_item_explanation(item_idx, user_ext_id=user_ext)
    return ExplanationResponse(**result)
