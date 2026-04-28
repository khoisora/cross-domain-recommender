"""Standalone demo FastAPI app.

Loads precomputed artifacts into memory and serves:
  - Sample user selection with user groups (balanced, movie-heavy, cold-start)
  - 5-row hybrid recommendations (LightGCN, CDR, cooc, SBERT, popularity)
  - Item search + rating (persisted to SQLite)
  - Item detail with SBERT similar items

Run:  PYTHONPATH=. uvicorn backend.demo.main:app --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import numpy as np

from backend.demo.store import DemoStore
from backend.demo.recommender import HybridRecommender
from backend.demo.database import DemoDB
from backend.demo.retrain import start_retrain_scheduler, stop_retrain_scheduler

logger = logging.getLogger(__name__)

# ── Pydantic schemas ────────────────────────────────────────────────────────

class SampleUser(BaseModel):
    id: int
    external_id: str
    name: str
    avatar: str = ""
    taste_summary: str = ""
    total_ratings: int = 0
    avg_rating: float = 0.0
    is_sample: bool = True
    group: str = ""

class CreateUserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)

class ItemOut(BaseModel):
    idx: int = 0
    external_id: str = ""
    title: str = ""
    domain: str = ""
    image_url: str = ""
    description: str = ""
    avg_rating: float | None = None
    rating_count: int = 0
    user_rating: float | None = None
    score: float = 0
    reason: str = ""

class RecommendationRow(BaseModel):
    model_config = {"protected_namespaces": ()}
    key: str
    title: str
    subtitle: str
    model_tag: str = ""
    items: list[ItemOut]

class SegmentInfo(BaseModel):
    segment: str
    explainer: str
    game_count: int
    movie_count: int


class RecommendationResponse(BaseModel):
    user_id: int
    user_name: str
    segment: SegmentInfo
    rows: list[RecommendationRow]

class RatingRequest(BaseModel):
    user_id: int
    external_id: str
    rating: float = Field(..., ge=1.0, le=5.0)

class RatingResponse(BaseModel):
    success: bool
    message: str

class SimilarItem(BaseModel):
    external_id: str
    title: str
    domain: str
    image_url: str = ""
    avg_rating: float | None = None
    similarity: float = 0

class ItemDetail(BaseModel):
    idx: int = 0
    external_id: str
    title: str
    domain: str
    image_url: str = ""
    description: str = ""
    avg_rating: float | None = None
    rating_count: int = 0
    user_rating: float | None = None
    similar_games: list[SimilarItem] = []
    similar_movies: list[SimilarItem] = []

class SearchResponse(BaseModel):
    items: list[ItemOut]
    total: int

class UserGroupResponse(BaseModel):
    groups: dict[str, list[SampleUser]]


# ── App lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    store = DemoStore.get()
    store.load()
    db = DemoDB.get()
    db.upsert_sample_users(store.sample_users)
    # Seed lightweight users (≤30 ratings) from the ratings store into the DB
    # so all users shown on the home page are real DB users.
    _seed_lightweight_users(store, db)
    # Restore runtime ratings from DB into in-memory store. Look up domain from
    # the item catalog so segment classification, cooc, and SBERT profiling all
    # work for restored runtime ratings.
    for r in db.get_all_runtime_ratings():
        existing = store.user_ratings.get(r["user_id"], [])
        if not any(x["item_id"] == r["item_id"] for x in existing):
            meta = store.items_by_ext.get(r["item_id"], {})
            store.add_rating(r["user_id"], r["item_id"], r["rating"],
                             domain=meta.get("domain", ""))
    # Start hourly retrain scheduler (LightGCN re-trains with latest ratings)
    start_retrain_scheduler(interval_seconds=3600)
    logger.info("Retrain scheduler started (hourly)")
    yield
    stop_retrain_scheduler()

app = FastAPI(title="CrossRec Demo", version="2.0.0", lifespan=lifespan)


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that tells the browser never to cache — useful in dev so
    edits to html/js/css land on the next reload without manual hard-refresh."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


# Serve jQuery frontend from /frontend_jquery/
_JQUERY_DIR = Path(__file__).resolve().parent.parent.parent / "frontend_jquery"
if _JQUERY_DIR.is_dir():
    app.mount("/static", _NoCacheStaticFiles(directory=str(_JQUERY_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup helpers ───────────────────────────────────────────────────────────

def _seed_lightweight_users(store: DemoStore, db: DemoDB) -> None:
    """Insert users with ≤30 ratings into the DB so they can be resolved by ID.

    The 200 sample users are all power users (50+ ratings). For the home page
    we want CDR-interesting users with few ratings, so we pull them from the
    full ratings store and persist them in SQLite.
    """
    # Seed enough users per CDR group to fill the home page
    needed = {"cold_start": 30, "few_target": 30, "balanced": 30}
    count = 0
    for ext_id, ratings in store.user_ratings.items():
        n = len(ratings)
        if n == 0 or n > 30:
            continue
        # Classify to check which group still needs users
        games = sum(1 for r in ratings if r.get("domain") == "game")
        if games == 0:
            grp = "cold_start"
        elif games <= 3:
            grp = "few_target"
        else:
            grp = "balanced"
        if needed.get(grp, 0) <= 0:
            continue
        if all(v <= 0 for v in needed.values()):
            break
        existing = db.get_user_by_external_id(ext_id)
        if existing:
            needed[grp] -= 1
            continue
        name = f"User {ext_id[:8]}"
        db.create_user_with_ext_id(ext_id, name)
        needed[grp] -= 1
        count += 1
    if count:
        logger.info("Seeded %d lightweight users (≤30 ratings) into DB", count)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store() -> DemoStore:
    return DemoStore.get()

def _db() -> DemoDB:
    return DemoDB.get()

def _recommender() -> HybridRecommender:
    return HybridRecommender(_store())

def _classify_user(user: dict, store: DemoStore) -> str:
    """Classify user into a CDR-relevant group for the home page.

    Groups reflect cross-domain cold-start severity:
      cold_start:  0 game ratings — pure CDR transfer from movies
      few_target:  1-3 game ratings + movies
      balanced:    4+ game ratings (warm user)
    """
    ext_id = user.get("external_id", "")
    ratings = store.user_ratings.get(ext_id, [])
    games = sum(1 for r in ratings if r.get("domain") == "game")
    if games == 0:
        return "cold_start"
    if games <= 3:
        return "few_target"
    return "balanced"

def _resolve_user(user_id: int) -> dict:
    s = _store()
    user = s.sample_users_by_id.get(user_id)
    if user:
        return user
    db = _db()
    db_user = db.get_user(user_id)
    if not db_user:
        raise HTTPException(404, f"User {user_id} not found")
    return {
        "id": db_user["id"],
        "external_id": db_user["external_id"],
        "name": db_user["name"],
        "avatar": db_user.get("avatar", ""),
        "taste_summary": db_user.get("taste_summary", ""),
        "total_ratings": db.get_user_rating_count(user_id),
        "avg_rating": 0.0,
        "is_sample": bool(db_user.get("is_sample", 0)),
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "loaded": _store().loaded}


@app.get("/api/user-groups", response_model=UserGroupResponse)
async def get_user_groups():
    """Get users organized by group for the home page."""
    s = _store()
    groups: dict[str, list[SampleUser]] = {
        "cold_start": [], "few_target": [], "balanced": [], "new": [],
    }
    # Scan all DB users, filter to ≤30 ratings, classify by CDR group.
    # Lightweight users were seeded into DB at startup from the ratings store.
    db = _db()
    for row in db.list_users():
        ext_id = row["external_id"]
        ratings = s.user_ratings.get(ext_id, [])
        n_ratings = len(ratings)
        if n_ratings == 0 or n_ratings > 30:
            continue
        if all(len(v) >= 30 for v in (groups["cold_start"], groups["few_target"], groups["balanced"])):
            break
        u_dict = {"external_id": ext_id}
        group = _classify_user(u_dict, s)
        if group not in groups or len(groups[group]) >= 30:
            continue
        movies = sum(1 for r in ratings if r.get("domain") == "movie")
        games = sum(1 for r in ratings if r.get("domain") == "game")
        groups[group].append(SampleUser(
            id=row["id"], external_id=ext_id,
            name=row["name"], avatar=row.get("avatar", ""),
            taste_summary=f"{movies} movies, {games} games",
            total_ratings=n_ratings,
            avg_rating=0.0, group=group,
        ))

    # "new" group — last 10 registered custom users, regardless of rating count
    for row in db.list_recent_custom_users(limit=10):
        ext_id = row["external_id"]
        ratings = s.user_ratings.get(ext_id, [])
        movies = sum(1 for r in ratings if r.get("domain") == "movie")
        games = sum(1 for r in ratings if r.get("domain") == "game")
        summary = f"{movies} movies, {games} games" if (movies or games) else "Just joined"
        groups["new"].append(SampleUser(
            id=row["id"], external_id=ext_id,
            name=row["name"], avatar=row.get("avatar", "👤") or "👤",
            taste_summary=summary,
            total_ratings=len(ratings),
            avg_rating=0.0, group="new",
        ))
    return UserGroupResponse(groups=groups)


@app.post("/api/users", response_model=SampleUser)
async def create_user(req: CreateUserRequest):
    db = _db()
    user = db.create_user(req.name)
    return SampleUser(
        id=user["id"], external_id=user["external_id"],
        name=user["name"], avatar=user.get("avatar", ""),
        taste_summary=user.get("taste_summary", ""),
        total_ratings=0, avg_rating=0.0, is_sample=False, group="new_user",
    )


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = _resolve_user(user_id)
    s = _store()
    ext_id = user["external_id"]
    ratings = []
    for r in s.user_ratings.get(ext_id, [])[:50]:
        meta = s.items_by_ext.get(r["item_id"], {})
        ratings.append({
            "item_id": r["item_id"],
            "external_id": r["item_id"],
            "title": meta.get("title", ""),
            "domain": r.get("domain", meta.get("domain", "")),
            "rating": r["rating"],
            "image_url": meta.get("image_url") or "",
        })
    return {**user, "ratings": ratings}


@app.get("/api/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int):
    user = _resolve_user(user_id)
    rec = _recommender()
    rows = rec.recommend_rows(user["external_id"], k_per_row=15)
    segment = rec.segment_info(user["external_id"])
    return RecommendationResponse(
        user_id=user_id,
        user_name=user.get("name", ""),
        segment=SegmentInfo(**segment),
        rows=[RecommendationRow(**r) for r in rows],
    )


@app.get("/api/items/search", response_model=SearchResponse)
async def search_items(
    q: str = Query("", min_length=0),
    limit: int = Query(20, ge=1, le=50),
    domain: str | None = Query(None),
    user_id: int | None = Query(None),
):
    s = _store()
    if not q:
        results = sorted(s.items_list, key=lambda x: x.get("rating_count") or 0, reverse=True)
    else:
        results = s.search_items(q, limit=200)
    if domain:
        results = [it for it in results if it.get("domain") == domain]
    results = results[:limit]

    user_rating_map: dict[str, float] = {}
    if user_id is not None:
        try:
            user = _resolve_user(user_id)
            for r in s.user_ratings.get(user["external_id"], []):
                user_rating_map[r["item_id"]] = r["rating"]
        except HTTPException:
            pass

    items = [ItemOut(
        idx=it.get("idx", 0),
        external_id=it.get("external_id", ""),
        title=it.get("title", ""),
        domain=it.get("domain", ""),
        image_url=it.get("image_url") or "",
        description=(it.get("description") or "")[:150],
        avg_rating=it.get("avg_rating"),
        rating_count=it.get("rating_count") or 0,
        user_rating=user_rating_map.get(it.get("external_id", "")),
    ) for it in results]
    return SearchResponse(items=items, total=len(items))


@app.get("/api/items/{external_id}", response_model=ItemDetail)
async def get_item(external_id: str, user_id: int | None = Query(None)):
    """Get item details by external_id. Returns SBERT similar items + user's rating."""
    s = _store()
    meta = s.items_by_ext.get(external_id)
    if not meta:
        try:
            db_idx = int(external_id)
            meta = s.items_by_db_idx.get(db_idx)
        except ValueError:
            pass
    if not meta:
        raise HTTPException(404, f"Item {external_id} not found")

    # Look up user's rating for this item
    user_rating = None
    if user_id is not None:
        try:
            user = _resolve_user(user_id)
            ext_id = user["external_id"]
            item_ext = meta.get("external_id", "")
            for r in s.user_ratings.get(ext_id, []):
                if r["item_id"] == item_ext:
                    user_rating = r["rating"]
                    break
            # Also check DB
            if user_rating is None:
                db_rating = _db().get_user_rating_for_item(user_id, meta.get("idx", 0))
                if db_rating is not None:
                    user_rating = db_rating
        except Exception:
            pass

    similar_games: list[SimilarItem] = []
    cd_idx = meta.get("cd_idx")
    if cd_idx is not None and cd_idx < s.content_emb.shape[0]:
        item_vec = s.content_emb[cd_idx]
        norm = np.linalg.norm(item_vec)
        if norm > 0:
            item_vec = item_vec / norm
            all_scores = s.content_emb @ item_vec
            top_indices = np.argsort(-all_scores)
            for si in top_indices:
                si = int(si)
                if si == cd_idx:
                    continue
                score = float(all_scores[si])
                if score < 0.1:
                    break
                sim_meta = s.cd_items_by_idx.get(si, {})
                if not sim_meta or sim_meta.get("domain") != "game":
                    continue
                similar_games.append(SimilarItem(
                    external_id=sim_meta.get("external_id", ""),
                    title=sim_meta.get("title", ""),
                    domain="game",
                    image_url=sim_meta.get("image_url") or "",
                    avg_rating=sim_meta.get("avg_rating"),
                    similarity=round(score, 3),
                ))
                if len(similar_games) >= 12:
                    break

    return ItemDetail(
        idx=meta.get("idx", 0),
        external_id=meta.get("external_id", ""),
        title=meta.get("title", ""),
        domain=meta.get("domain", ""),
        image_url=meta.get("image_url") or "",
        description=meta.get("description") or "",
        avg_rating=meta.get("avg_rating"),
        rating_count=meta.get("rating_count") or 0,
        user_rating=user_rating,
        similar_games=similar_games,
        similar_movies=[],
    )


@app.post("/api/ratings", response_model=RatingResponse)
async def submit_rating(req: RatingRequest):
    """Submit a rating — immediately updates cooc + SBERT recs."""
    user = _resolve_user(req.user_id)
    s = _store()
    db = _db()
    meta = s.items_by_ext.get(req.external_id)
    if not meta:
        raise HTTPException(404, f"Item {req.external_id} not found")
    domain = meta.get("domain", "")
    # Persist to DB
    db_idx = meta.get("idx", 0)
    db.add_rating(req.user_id, db_idx, req.external_id, req.rating)
    # Update in-memory store (cooc + SBERT rows refresh instantly)
    s.add_rating(user["external_id"], req.external_id, req.rating, domain=domain)
    return RatingResponse(
        success=True,
        message=f"Rated {meta.get('title', '')} {req.rating:.0f}/5. Recommendations updated.",
    )
