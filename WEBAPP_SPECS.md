# Web Application Specifications

Comprehensive specification for the CrossRec demo web application — a hybrid movie & game recommender powered by cross-domain and single-domain ML models.

---

## Architecture Overview

```
Frontend (Next.js 14, port 3000)
  ├── Home page — user selection with group tabs
  ├── Recommendations page — 6 model-powered rows
  ├── Item detail page — metadata + SBERT similar items
  └── Search — cross-domain item search

FastAPI Backend (port 8000)
  ├── /api/users — user listing + creation
  ├── /api/recommendations/{user_id} — 6 rec rows
  ├── /api/items/{external_id} — item detail + similar
  ├── /api/items/search — full-text search
  ├── /api/ratings — submit rating (persists to DB + memory)
  ├── /api/retrain — trigger model retrain
  └── /api/retrain/status — last retrain info

In-Memory Store (DemoStore)
  ├── LightGCN embeddings (19,880 users × 96d, 10,034 items × 96d)
  ├── NCF embeddings (19,880 × 64d, 10,034 × 64d)
  ├── EMCDR embeddings (19,880 × 64d, 50,322 × 64d)
  ├── PTUPCDR embeddings (19,880 × 64d, 50,322 × 64d)
  ├── SBERT content embeddings (50,322 × 384d)
  ├── Co-occurrence matrix (34,596 movies × game associations)
  └── User ratings (in-memory dict, synced to SQLite)

SQLite Database (data/demo.db)
  ├── users — sample + custom users
  └── ratings — persistent rating storage

Background Retrain Scheduler
  └── Hourly LightGCN retrain → hot-swap embeddings
```

---

## Page Specifications

### 1. Home Page (`/`)

**Purpose**: User selection and onboarding entry point.

**Components**:

| Component | Description | Justification |
|---|---|---|
| Hero section | CrossRec branding with Film + Gamepad icons | Immediately communicates the cross-domain nature of the system |
| Create user form | Name input → creates new user with no history | Allows demo of cold-start scenario (zero ratings) |
| User group tabs | 4 tabs: Balanced, Movie Heavy, Cold Start, Game Heavy | **Key design decision**: different user groups trigger different recommendation strategies. Professors can see how the routing rule works in practice |
| User cards (20 per group) | Avatar, name, taste summary, rating count | Limited to 20 per group for readability |

**User Groups**:

| Group | Definition | Recommendation Strategy |
|---|---|---|
| Balanced | Has both movie + game ratings (ratio 1:5 to 5:1) | Full pipeline: LightGCN+cooc, CDR, SBERT, popularity |
| Movie Heavy | Movie/game ratio > 5:1 | CDR transfer rows (PTUPCDR) are most valuable — rich movie profile |
| Cold Start | Zero game ratings, movies only | EMCDR+cooc cold-start path — demonstrates CDR's core value proposition |
| Game Heavy | More games than movies | LightGCN dominates — graph convolution on dense game interaction graph |

**Design justification**: The group tabs directly map to the routing rule from our experiments (Lesson 6). A professor can select a cold-start user and see that EMCDR provides recommendations purely from movie history, while LightGCN collapses. This is the experiment findings made interactive.

---

### 2. Recommendations Page (`/recommendations?user={id}`)

**Purpose**: Main recommendation experience — 6 model-powered rows, each clearly labeled with the algorithm that generates it.

**Recommendation Rows**:

| Row | Model | Title | Refresh Speed | Visual |
|---|---|---|---|---|
| 1 | LightGCN + cooc | "Top Picks for You" | Batch (hourly retrain) | Purple (games) with Gamepad2 watermark |
| 2 | PTUPCDR or EMCDR | "Based on Your Movie Taste" | Batch | Purple/blue depending on output domain |
| 3 | Cooc standalone | "Players Who Watched Your Movies Also Played" | **Instant** | Purple (games) |
| 4 | SBERT | "Games Similar in Theme" | **Instant** | Purple (games) with Film watermark for cross-domain origin |
| 5 | SBERT | "Movies You Might Enjoy" | **Instant** | Blue (movies) with Film watermark |
| 6 | Popularity | "Trending Games" | Instant (static) | Purple (games) |

**Row selection logic**:
- Row 2 uses PTUPCDR if user has ≥1 game rating (personalized MoE mapping activates), EMCDR otherwise (global MLP works at cold-start)
- Rows 3-5 read from `store.user_ratings` on every request — they reflect the user's latest ratings immediately, no retrain needed
- Row 1 uses embeddings that are updated hourly by the background retrain scheduler

**Design justification**: Each row demonstrates a different recommendation paradigm from our experiments:
- Row 1: collaborative filtering (Lesson 2-4 finding: LightGCN is strongest on warm users)
- Row 2: cross-domain transfer (Lesson 3-6 finding: CDR works with sufficient overlap)
- Row 3: behavioral co-occurrence (Lesson 8 finding: cooc improves every model)
- Row 4-5: content-based (Lesson 7 finding: SBERT wins 11× on niche items)
- Row 6: popularity baseline (Lesson 6 finding: strong at cold-start)

**Visual Components per Row**:

| Component | Description | Justification |
|---|---|---|
| Model tag pill | Blue badge showing algorithm name (e.g. "LightGCN + Co-occurrence") | Transparency: user/professor sees which model powers each row |
| Domain watermark | Large faded Film/Gamepad2 icon behind the lane (12% opacity, 192px) | Instant visual domain identification without reading text |
| Domain lane | Blue gradient (movies) or purple gradient (games) with colored left border | Consistent domain color coding throughout the app |
| Item count badge | Colored pill showing number of items per lane | Quick overview of recommendation volume |
| Scroll arrows | Left/right arrows on hover for horizontal scrolling | Netflix-style browsing UX |
| Item cards | 180px wide, poster-style with domain badge, title, rating | Standard rec system card layout |
| Hover overlay | Shows description + recommendation reason on hover | Explains why each item was recommended |

**Your Rated Items Section**:
- Collapsible (closed by default)
- Shows horizontal card lanes split by domain (movies/games)
- Same card style as recommendation rows for visual consistency
- Shows the user's star rating on each card

---

### 3. Item Detail Page (`/item/{external_id}?user={id}`)

**Purpose**: Full item information + SBERT content-similar items in both domains.

**Components**:

| Component | Description | Justification |
|---|---|---|
| Item metadata | Title, description, domain badge, image, avg rating, rating count | Standard item detail |
| Star rating | 5-star clickable rating (persists to DB + memory) | Rating triggers instant cooc+SBERT refresh on next recommendation page visit |
| Similar Games (SBERT) | 12 games ranked by SBERT cosine similarity, purple badge | Demonstrates content-based same-domain recommendations |
| Similar Movies (SBERT) | 12 movies ranked by SBERT cosine similarity, blue badge | **Key feature**: demonstrates cross-domain content matching — SBERT operates in a shared text space so "Halo: Combat Evolved" (game) → "Halo: The Complete Video Collection" (movie) naturally |

**Similar items design justification**: Showing both same-domain and cross-domain similar items on every item page demonstrates SBERT's shared semantic space (Lesson 7 finding). A professor can navigate from a game to a semantically similar movie and back — the cross-domain bridge is visible and interactive.

**Rating persistence**:
1. User clicks star → `POST /api/ratings` saves to SQLite DB + in-memory store
2. Page refresh → `GET /api/items/{id}?user_id=N` returns `user_rating` from store/DB
3. Stars display the saved rating immediately

---

### 4. Search (`/recommendations` search overlay)

**Purpose**: Find any movie or game in the catalog.

| Feature | Description |
|---|---|
| Trigger | Search icon in top nav |
| Input | Auto-focus text input, substring match |
| Results | Up to 20 items, shows domain icon, title, rating |
| Navigation | Click result → item detail page |

---

## Refresh Architecture

### Instant Refresh (no retrain needed)

| Signal | Mechanism | Latency |
|---|---|---|
| Co-occurrence (Row 3) | Dict lookup: `cooc[movie_id][game_id]` summed over user's movie history | ~1ms |
| SBERT (Rows 4-5) | Mean of user's rated item embeddings → cosine similarity against all items | ~50ms |
| Popularity (Row 6) | Pre-sorted by rating count, static | ~0ms |

These rows re-compute from `store.user_ratings` on every `/api/recommendations` call. When a user rates a new movie, the next page refresh immediately shows different cooc and SBERT recommendations.

### Batch Refresh (retrain required)

| Signal | Mechanism | Latency |
|---|---|---|
| LightGCN (Row 1) | Full GCN retrain on current ratings → hot-swap embeddings | ~40s |
| CDR (Row 2) | EMCDR/PTUPCDR retrain (not currently scheduled) | ~60s |

The background scheduler re-trains LightGCN every hour. Manual trigger via `POST /api/retrain`.

**Design justification**: The split between instant and batch refresh mirrors real production systems. Users get immediate feedback on their ratings (cooc+SBERT update), while collaborative models batch-update on a schedule. This is a practical demonstration of the trade-off between freshness and compute cost.

---

## Data Flow

```
User rates item on Item Detail page
  │
  ├─→ POST /api/ratings
  │     ├─→ SQLite DB (persist)
  │     └─→ store.user_ratings (in-memory, instant)
  │
  ├─→ User navigates to Recommendations page
  │     ├─→ Row 3 (Cooc): re-computes from user_ratings → instant update
  │     ├─→ Row 4-5 (SBERT): re-builds profile from user_ratings → instant update
  │     └─→ Row 1 (LightGCN): uses current embeddings → updated on next hourly retrain
  │
  └─→ Background retrain (every hour)
        └─→ Re-trains LightGCN with all current user_ratings
        └─→ Hot-swaps store.lgcn_user / store.lgcn_item
```

---

## Model-to-Page Mapping Summary

| Model | Page | Section | Why Here |
|---|---|---|---|
| LightGCN | Recommendations | Row 1: "Top Picks" | Best overall model (Recall@10=0.060 on L3). Graph convolution captures behavioral patterns |
| PTUPCDR | Recommendations | Row 2: "Movie Taste" (warm users) | Personalized MoE mapping activates with ≥1 game interaction (Lesson 4: closes to 17% of LightGCN) |
| EMCDR | Recommendations | Row 2: "Movie Taste" (cold users) | Global MLP works at zero-game history (Lesson 6: best CDR at cold-start) |
| Cooc | Recommendations | Row 3: "Also Played" | Training-free, instant refresh, universally improves every model (Lesson 8) |
| SBERT | Recommendations | Rows 4-5: "Similar Theme" (games + movies) | Content similarity in shared text space — 11× on niche items (Lesson 7) |
| Popularity | Recommendations | Row 6: "Trending" | Strong cold-start baseline, always available (Lesson 6: 0.038 Recall@10) |
| SBERT | Item Detail | Similar Games + Similar Movies | Cross-domain content matching — demonstrates shared semantic space |

---

## Technical Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | Next.js (App Router) + React + TypeScript | 14.0.4 / 18.2 |
| Styling | Tailwind CSS | 3.4.0 |
| Icons | lucide-react (Film, Gamepad2, Star, etc.) | — |
| Backend | FastAPI + Uvicorn | — |
| Database | SQLite (demo.db) | — |
| ML Models | PyTorch, RecBole, RecBole-CDR, sentence-transformers | — |
| Embeddings | NumPy arrays loaded into memory at startup | — |

---

## API Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| GET | `/health` | Health check | None |
| GET | `/api/users` | List all users (with group labels) | None |
| GET | `/api/user-groups` | Users organized by group | None |
| POST | `/api/users` | Create new user | None |
| GET | `/api/users/{id}` | User profile + rating history | None |
| GET | `/api/recommendations/{id}` | 6 recommendation rows | None |
| GET | `/api/items/search?q=...` | Full-text item search | None |
| GET | `/api/items/{external_id}?user_id=N` | Item detail + similar + user rating | None |
| POST | `/api/ratings` | Submit rating `{user_id, external_id, rating}` | None |
| POST | `/api/retrain` | Trigger immediate model retrain | None |
| GET | `/api/retrain/status` | Last retrain info | None |
