# Web Application Specifications

Comprehensive specification for the CrossRec demo web application — a hybrid movie & game recommender powered by cross-domain and single-domain ML models.

---

## Architecture Overview

```
Frontend (jQuery single-page app, served at /static/index.html)
  ├── Home page (#/) — user creation + sample user selection with CDR group tabs
  ├── Recommendations page (#/recs/{userId}) — 9 model-powered rows
  ├── Item detail page (#/item/{externalId}) — metadata + SBERT similar items
  └── Search overlay (Cmd+K) — cross-domain item search

FastAPI Backend (port 8000)
  ├── /api/users — user listing + creation (POST, no auth)
  ├── /api/user-groups — users organized by CDR group
  ├── /api/recommendations/{user_id} — 9 rec rows
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
  ├── Reverse co-occurrence (8,915 games → movie associations)
  ├── LightGCN movie-domain (19,880 × 96d, 39,402 × 96d)
  └── User ratings (in-memory dict, synced to SQLite)

SQLite Database (data/demo.db)
  ├── users — sample + custom users
  └── ratings — persistent rating storage

Background Retrain Scheduler
  └── Hourly LightGCN retrain → hot-swap embeddings
```

---

## Page Specifications

### 1. Home Page (`#/`)

**Purpose**: User creation and sample user selection.

**Components**:

| Component | Description |
|---|---|
| Hero section | CrossRec branding with tagline |
| Create user form | Name input only (no password) → creates new user → navigates to recs with popularity-based recommendations |
| Divider | "— or pick a sample user —" |
| CDR group tabs | 4 tabs: Cold Start, 1-Shot Target, Few Target, Balanced |
| User cards (up to 30 per group) | Avatar, name, taste summary, total rating count |

**User Groups (CDR-relevant classification)**:

| Group | Definition | Recommendation Strategy |
|---|---|---|
| Cold Start | 0 game ratings | EMCDR+cooc cold-start path — pure transfer from movie history |
| 1-Shot Target | Exactly 1 game rating + movies | CDR blends movie preference with minimal game signal |
| Few Target | 2-3 game ratings + movies | CDR models blend transferred preferences with sparse game history |
| Balanced | 4+ game ratings | Full pipeline: LightGCN+cooc, CDR, SBERT, popularity |

**New User Flow**:
1. User types a name → clicks "Start" (or presses Enter)
2. `POST /api/users` creates user in SQLite (no password, no auth)
3. Navigates to `#/recs/{newUserId}`
4. With zero ratings, only **popularity rows** appear (Trending Games + Trending Movies)
5. As the user rates items, SBERT and cooc rows appear on next page load

---

### 2. Recommendations Page (`#/recs/{userId}`)

**Purpose**: Main recommendation experience — up to 9 model-powered rows.

**Recommendation Rows**:

| Row | Model | Title | Refresh Speed |
|---|---|---|---|
| 1 | LightGCN + cooc | "Top Picks for You" | Batch (hourly retrain) |
| 2 | PTUPCDR or EMCDR | "Based on Your Movie Taste" | Batch |
| 3 | Cooc standalone | "Players Who Watched Your Movies Also Played" | **Instant** |
| 4 | SBERT (games) | "Games Similar in Theme" | **Real-time** (latest 5 ratings) |
| 5 | SBERT (movies) | "Movies You Might Enjoy" | **Real-time** (latest 5 ratings) |
| 6 | LightGCN movies | "Top Movie Picks" | Batch |
| 7 | Reverse cooc | "Movies Fans of Your Games Also Watched" | **Instant** |
| 8 | Popularity (games) | "Trending Games" | Static |
| 9 | Popularity (movies) | "Trending Movies" | Static |

**SBERT Real-time Behavior**:
- SBERT profile is computed from the user's **latest 5 rated items** (rating >= 4)
- Uses only recent items so recommendations respond to current taste, not diluted by old ratings
- After rating an item on the detail page, navigating back auto-reloads fresh recommendations

**Row selection logic**:
- Row 2 uses PTUPCDR if user has ≥1 game rating, EMCDR otherwise (cold-start)
- Rows 3-5 read from `store.user_ratings` on every request — instant refresh
- Row 1 uses embeddings updated hourly by background retrain scheduler
- New users (zero ratings) only see rows 8-9 (popularity)

**Visual Components per Row**:

| Component | Description |
|---|---|
| Model tag pill | Badge showing algorithm name (e.g. "LightGCN + Co-occurrence") |
| Domain lane | Red gradient (games) or blue gradient (movies) with colored left border |
| Lane watermark | Large faded emoji behind the lane (6% opacity) |
| Scrollbar | Color-matched: red for game lanes, blue for movie lanes |
| Scroll arrows | `‹` / `›` buttons appear on hover, scroll 3 cards at a time |
| Item cards | 160px wide, poster-style with domain badge, title, avg rating |

**Your Rated Items Section**:
- **Expanded by default** (collapsible via chevron toggle)
- Shows horizontal card lanes split by domain (movies/games)
- Each card shows the **user's star rating** as a golden badge (e.g. ★5)
- Same card style as recommendation rows for visual consistency

---

### 3. Item Detail Page (`#/item/{externalId}`)

**Purpose**: Full item information + SBERT content-similar items.

**URL**: Each item has its own URL path via hash routing. Browser back/forward works.

**Components**:

| Component | Description |
|---|---|
| Back button | Returns to recommendations page |
| Item poster | Max-height 300px, `object-fit: contain` (not stretched) |
| Title + domain badge | Item name with colored domain tag (MOVIE/GAME) |
| Avg rating + count | Star rating and number of ratings |
| **Star rating** | 5-star clickable rating — **positioned above the description** |
| Description | Item description text |
| Similar Games | Horizontal scroll lane (same UI as rec rows) with SBERT model tag |
| Similar Movies | Horizontal scroll lane (same UI as rec rows) with SBERT model tag |

**Rating behavior**:
- Stars update immediately on click (no toast/success message)
- Rating persists to SQLite + in-memory store via `POST /api/ratings`
- Navigating back to recs auto-reloads, reflecting the new rating in SBERT rows

**Similar items design**: Both game and movie similar sections use the same domain-lane UI as recommendation rows (watermark, colored border, scroll arrows, scrollbar colors).

---

### 4. Search (Overlay)

| Feature | Description |
|---|---|
| Trigger | Magnifying glass icon in nav bar, or Cmd+K / Ctrl+K |
| Display | Full-screen dark overlay with centered search box |
| Input | Auto-focus text input, substring match (min 2 chars) |
| Results | Up to 15 items with domain emoji, title, avg rating |
| Navigation | Click result → item detail page (overlay closes) |
| Dismiss | Click outside or press Escape |

---

## Navigation & Routing

**Hash-based routing** — all pages have unique URLs:

| Route | Page |
|---|---|
| `#/` | Home (user selection) |
| `#/recs/{userId}` | Recommendations for user |
| `#/item/{externalId}` | Item detail |

- Browser back/forward navigates between pages
- Direct URL access works (e.g. bookmark a user's recs page)

**Nav Bar** (visible on recs + detail pages, hidden on home):

| Left | Center | Right |
|---|---|---|
| Username | **CrossRec** (centered logo, click → user's recs page) | Search icon + "Switch User" button |

---

## Refresh Architecture

### Real-time (latest 5 ratings)

| Signal | Mechanism | Latency |
|---|---|---|
| SBERT (Rows 4-5) | Mean of user's **latest 5** rated item embeddings → cosine similarity | ~50ms |

### Instant (no retrain)

| Signal | Mechanism | Latency |
|---|---|---|
| Co-occurrence (Rows 3, 7) | Dict lookup summed over user's movie/game history | ~1ms |
| Popularity (Rows 8-9) | Pre-sorted by rating count, static | ~0ms |

### Batch (retrain required)

| Signal | Mechanism | Latency |
|---|---|---|
| LightGCN (Rows 1, 6) | Full GCN retrain → hot-swap embeddings | ~40s (hourly) |
| CDR (Row 2) | EMCDR/PTUPCDR retrain | ~60s |

---

## Data Flow

```
User rates item on Item Detail page
  │
  ├─→ POST /api/ratings
  │     ├─→ SQLite DB (persist)
  │     └─→ store.user_ratings (in-memory, instant)
  │
  ├─→ User navigates back to Recommendations page (auto-reloads)
  │     ├─→ Rows 4-5 (SBERT): profile from latest 5 ratings → instant update
  │     ├─→ Rows 3,7 (Cooc): re-computes from user_ratings → instant update
  │     └─→ Rows 1,6 (LightGCN): uses current embeddings → updated on hourly retrain
  │
  └─→ Background retrain (every hour)
        └─→ Re-trains LightGCN with all current user_ratings
        └─→ Hot-swaps store.lgcn_user / store.lgcn_item
```

---

## UI Design Rules

| Rule | Detail |
|---|---|
| Domain colors | **Blue** for movies, **Red** for games — applied to lane borders, badges, scrollbars |
| No drag-to-scroll | Card rows use `‹` `›` arrow buttons on hover instead |
| Scrollbar colors | Red (`#dc2626`) for game lanes, blue (`#2563eb`) for movie lanes |
| Image sizing | Detail page poster: max-height 300px, `object-fit: contain` |
| Star rating position | Above the description on item detail page |
| No toast messages | Rating submission updates stars silently, no success text |
| No "Refresh Models" button | Background retrain is automatic (hourly) |
| Rated items expanded | "Your Rated Items" section is open by default |
| Rated items show ratings | Golden badge with ★N on each rated item card |
| Similar items UI | Same horizontal lane UI on detail page as rec rows |

---

## Technical Stack

| Layer | Technology |
|---|---|
| Frontend | jQuery 3.7.1, single HTML file, Inter font |
| Styling | Inline CSS (dark theme, `#0a0a0f` background) |
| Routing | Hash-based (`window.location.hash`) |
| Backend | FastAPI + Uvicorn |
| Database | SQLite (demo.db) |
| ML Models | PyTorch, RecBole, RecBole-CDR, sentence-transformers |
| Embeddings | NumPy arrays loaded into memory at startup |
| Docker | CPU-only torch, `requirements-docker.txt` |

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/api/users` | List all users (with group labels) |
| GET | `/api/user-groups` | Users organized by CDR group (cold_start, one_shot, few_target, balanced) |
| POST | `/api/users` | Create new user `{name}` — no password required |
| GET | `/api/users/{id}` | User profile + rating history (max 50) |
| GET | `/api/recommendations/{id}` | Up to 9 recommendation rows |
| GET | `/api/items/search?q=...` | Full-text item search |
| GET | `/api/items/{external_id}?user_id=N` | Item detail + similar + user rating |
| POST | `/api/ratings` | Submit rating `{user_id, external_id, rating}` |
| POST | `/api/retrain` | Trigger immediate model retrain |
| GET | `/api/retrain/status` | Last retrain info |
