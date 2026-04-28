# System Architecture

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  Home Page    │  │ Recommendations  │  │    Item Detail        │ │
│  │  (User Picker │  │ (9 Model Rows)   │  │ (Similar Games +     │ │
│  │   + Groups)   │  │                  │  │  Similar Movies)      │ │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬────────────┘ │
└─────────┼───────────────────┼───────────────────────┼──────────────┘
          │                   │                       │
          │              HTTP REST API                │
          ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (port 8000)                       │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ User API    │  │ Recommendation│  │ Item API                 │  │
│  │ /api/users  │  │ Engine        │  │ /api/items/{id}          │  │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬────────────────┘  │
│         │                │                      │                   │
│         ▼                ▼                      ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    IN-MEMORY STORE                            │   │
│  │                                                              │   │
│  │  ┌────────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐ │   │
│  │  │ LightGCN   │ │ LightGCN   │ │ EMCDR     │ │ PTUPCDR   │ │   │
│  │  │ (Games)    │ │ (Movies)   │ │ (CDR)     │ │ (CDR)     │ │   │
│  │  │ 19K×96d    │ │ 19K×96d    │ │ 19K×64d   │ │ 19K×64d   │ │   │
│  │  └────────────┘ └────────────┘ └───────────┘ └───────────┘ │   │
│  │  ┌────────────┐ ┌────────────┐ ┌───────────┐ ┌───────────┐ │   │
│  │  │ NCF        │ │ SBERT      │ │ Cooc      │ │ Reverse   │ │   │
│  │  │ (Games)    │ │ Content    │ │ Movie→Game│ │ Cooc      │ │   │
│  │  │ 19K×64d    │ │ 50K×384d   │ │ 34K movies│ │ Game→Movie│ │   │
│  │  └────────────┘ └────────────┘ └───────────┘ └───────────┘ │   │
│  │                                                              │   │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐ │   │
│  │  │ User Ratings (dict)  │  │ Item Catalog (66K items)     │ │   │
│  │  │ Updated on each rate │  │ Movies + Games metadata      │ │   │
│  │  └──────────────────────┘  └──────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐       ┌──────────────────────────────────────┐   │
│  │ SQLite DB    │       │ Background Retrain Scheduler          │   │
│  │ (demo.db)    │       │ Every 1 hour: retrain LightGCN       │   │
│  │ Users+Ratings│       │ → hot-swap embeddings in store        │   │
│  └──────────────┘       └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE (offline)                             │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Raw Data │───▶│ Process  │───▶│ Train    │───▶│ Export       │  │
│  │ (JSONL)  │    │ (Parquet)│    │ Models   │    │ Embeddings   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│                                                                     │
│  Amazon Reviews 2023 → process_data.py → bench_*.py → export_demo_ │
│  (Movies + Games)      (k-core filter)   (8 lessons)  artifacts.py  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Architecture

### 2.1 Data Sources and Processing

```
Amazon Reviews 2023 (Raw JSONL, ~2GB)
    │
    ▼
process_data.py
    │  • Convert ratings ≥ 4 → implicit positive feedback
    │  • Item k-core: movies ≥ 20, games ≥ 10 interactions
    │  • User filtering per dataset variant
    │
    ├──▶ processed/           (L1-L2: 1M users, 5.8% overlap)
    ├──▶ processed_overlap/   (L3,L6: 19K users, 100% overlap, movies≥5)
    ├──▶ processed_sparse_loose/ (L4,L7: 14K users, 100% overlap, movies≥10)
    └──▶ movies_games.db      (SQLite: 66K items, 95K users, 2.3M ratings)
```

### 2.2 Dual Item Space

The system maintains two parallel item index spaces to handle single-domain vs cross-domain models:

```
Single-Domain (SD) Space          Cross-Domain (CD) Space
┌─────────────────────┐           ┌─────────────────────┐
│ 10,034 game items   │           │ 50,322 items        │
│ idx: 0 → 10,033     │           │ 40,288 movies       │
│                     │           │ 10,034 games        │
│ Used by:            │           │ idx: 0 → 50,321     │
│  • LightGCN (games) │           │                     │
│  • NCF              │           │ Used by:            │
│  • Cooc             │           │  • EMCDR            │
│  • Popularity       │           │  • PTUPCDR          │
└─────────────────────┘           │  • SBERT            │
                                  │  • LightGCN (movies)│
Movie-Domain Space                └─────────────────────┘
┌─────────────────────┐
│ 39,402 movie items  │
│ idx: 0 → 39,401     │
│                     │
│ Used by:            │
│  • LightGCN (movies)│
│  • Reverse cooc     │
│  • Movie popularity │
└─────────────────────┘

Linked via external_id (Amazon ASIN) — the canonical item identifier
```

### 2.3 Database Schema

```sql
-- data/demo.db (SQLite, persistent across restarts)

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id TEXT UNIQUE NOT NULL,     -- model user ID
    name TEXT NOT NULL,
    avatar TEXT DEFAULT '',
    taste_summary TEXT DEFAULT '',
    is_sample INTEGER DEFAULT 0,
    created_at REAL
);

CREATE TABLE ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    item_idx INTEGER NOT NULL,            -- DB item index
    item_id TEXT NOT NULL,                -- external_id (ASIN)
    rating REAL NOT NULL,                 -- 1.0-5.0
    created_at REAL,
    UNIQUE(user_id, item_idx)
);
```

---

## 3. Model Architecture

### 3.1 Model Portfolio

```
                        ┌───────────────────────────┐
                        │     Model Portfolio        │
                        └─────────┬─────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐
   │  Single-Domain   │  │ Cross-Domain │  │  Content-Based   │
   │  (Game-Only)     │  │ (CDR)        │  │                  │
   ├──────────────────┤  ├──────────────┤  ├──────────────────┤
   │ LightGCN (games) │  │ CMF          │  │ SBERT            │
   │ LightGCN (movies)│  │ EMCDR        │  │ SBERT-CDR        │
   │ NCF (NeuMF)      │  │ PTUPCDR      │  │                  │
   │ MF-BPR           │  │              │  │                  │
   └──────────────────┘  └──────────────┘  └──────────────────┘
              │                   │                   │
              ▼                   ▼                   ▼
   ┌──────────────────────────────────────────────────────────┐
   │              Post-Processing Layer                        │
   │  ┌─────────────────────┐  ┌────────────────────────────┐ │
   │  │ Cooc (movie→game)   │  │ Reverse Cooc (game→movie)  │ │
   │  │ Training-free        │  │ Training-free               │ │
   │  │ Instant refresh      │  │ Instant refresh             │ │
   │  └─────────────────────┘  └────────────────────────────┘ │
   └──────────────────────────────────────────────────────────┘
```

### 3.2 Model Details

| Model | Type | Embedding | Training Data | Refresh |
|-------|------|-----------|---------------|---------|
| LightGCN (games) | Graph CF | 19,880 users × 96d, 10,034 items × 96d | 58K game interactions | Hourly retrain |
| LightGCN (movies) | Graph CF | 19,880 users × 96d, 39,402 items × 96d | 390K movie interactions | Hourly retrain |
| NCF | Neural CF | 19,880 × 64d, 10,034 × 64d | 58K game interactions | Batch |
| EMCDR | CDR mapping | 19,880 × 64d, 50,322 × 64d | 390K movie + 58K game | Batch |
| PTUPCDR | CDR personalized | 19,880 × 64d, 50,322 × 64d | 390K movie + 58K game | Batch |
| SBERT | Content | 50,322 × 384d | Item text metadata | Static |
| Cooc | Behavioral | 34,596 movie→game pairs | Overlap user co-preferences | Instant |
| Reverse Cooc | Behavioral | 8,915 game→movie pairs | Overlap user co-preferences | Instant |

### 3.3 Recommendation Flow

```
User requests /api/recommendations/{user_id}
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ HybridRecommender.recommend_rows()                   │
│                                                      │
│  1. Resolve user → external_id → model indices       │
│  2. Get user's rated item set (for exclusion)        │
│  3. For each row:                                    │
│     ┌────────────────────────────────────────────┐   │
│     │ a. Compute scores: item_emb @ user_emb     │   │
│     │ b. Add cooc bonus (if applicable)           │   │
│     │ c. Mask to correct domain (game/movie)      │   │
│     │ d. Normalize scores to [0,1]                │   │
│     │ e. Exclude rated items                      │   │
│     │ f. Take top-K (15 items)                    │   │
│     │ g. Cross-row deduplication by external_id   │   │
│     │ h. Enrich with metadata (title, image, etc) │   │
│     └────────────────────────────────────────────┘   │
│  4. Return 9 rows with enriched items                │
└─────────────────────────────────────────────────────┘
```

---

## 4. Frontend Architecture

### 4.1 Page Structure

```
Next.js 14 (App Router)
│
├── / (Home)
│   ├── Hero section (CrossRec branding)
│   ├── Create user form
│   └── User group tabs (Balanced | Movie Heavy | Cold Start | Game Heavy)
│       └── 20 user cards per group
│
├── /recommendations?user={id}
│   ├── Top nav (search, refresh, logout)
│   ├── Your Rated Items (collapsible, card lanes by domain)
│   └── 9 Recommendation Rows
│       ├── 5 Game rows (purple lanes, Gamepad2 watermark)
│       │   ├── Top Picks (LightGCN+cooc)
│       │   ├── Based on Movie Taste (CDR)
│       │   ├── Also Played (cooc)
│       │   ├── Games Similar in Theme (SBERT)
│       │   └── Trending Games (popularity)
│       └── 4 Movie rows (blue lanes, Film watermark)
│           ├── Top Movie Picks (LightGCN-movies+reverse cooc)
│           ├── Fans Also Watched (reverse cooc)
│           ├── Movies You Might Enjoy (SBERT)
│           └── Trending Movies (popularity)
│
├── /item/{external_id}?user={id}
│   ├── Item metadata + star rating
│   ├── Similar Games (12, SBERT cosine)
│   └── Similar Movies (12, SBERT cosine)
│
└── /admin
    └── Admin operations
```

### 4.2 Domain Visual Language

```
Movies (Blue theme)                  Games (Purple theme)
┌────────────────────────┐           ┌────────────────────────┐
│ Background: blue-950   │           │ Background: purple-950 │
│ Border: blue-400       │           │ Border: purple-400     │
│ Label: blue-400        │           │ Label: purple-400      │
│ Badge: bg-blue-600     │           │ Badge: bg-purple-600   │
│ Icon: Film (Lucide)    │           │ Icon: Gamepad2 (Lucide)│
│ Watermark: 160px, 15%  │           │ Watermark: 160px, 15%  │
└────────────────────────┘           └────────────────────────┘
```

---

## 5. Refresh Architecture

### 5.1 Three Refresh Tiers

```
                    Latency
Tier 1: Instant     ~1ms     ┌─ Cooc (dict lookup + sum)
(no retrain)                 ├─ Reverse Cooc (same)
                             ├─ SBERT (profile mean + dot product)
                             └─ Popularity (pre-sorted)

Tier 2: Fast       ~50ms     ┌─ Re-read user_ratings dict
(in-memory update)           └─ Scores recomputed on each API call

Tier 3: Batch      ~40s      ┌─ LightGCN retrain (background thread)
(model retrain)              └─ Hot-swap embeddings in store
                                Scheduled: every 1 hour
                                Manual: POST /api/retrain
```

### 5.2 Rating → Refresh Flow

```
User rates item (e.g. rates a movie 5★)
    │
    ├──▶ SQLite DB: INSERT INTO ratings
    │
    ├──▶ In-memory store: user_ratings[ext_id].append(...)
    │
    └──▶ User navigates to /recommendations
         │
         ├── Row "Also Played" (cooc):
         │   ✓ INSTANT — new movie's cooc entries are looked up
         │
         ├── Row "Games Similar in Theme" (SBERT):
         │   ✓ INSTANT — new movie's embedding added to user profile
         │
         ├── Row "Top Picks" (LightGCN):
         │   ✗ STALE until next retrain (hourly or manual)
         │
         └── Row "Based on Movie Taste" (CDR):
             ✗ STALE until next full export
```

---

## 6. ML Pipeline Architecture

### 6.1 Experiment Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    Experiment Pipeline                             │
│                                                                   │
│  process_data.py                                                  │
│       │  Raw JSONL → Parquet (per dataset variant)                │
│       ▼                                                           │
│  load_cross_domain_split()                                        │
│       │  Parquet → CrossDomainSplit (train/val/test + ID maps)    │
│       ▼                                                           │
│  bench_*.py (one per model)                                       │
│       │  model.fit(train) → evaluate_cross_domain(model, test)    │
│       ▼                                                           │
│  save_result()                                                    │
│       │  → artifacts/results/<model>_lesson<N>.json               │
│       ▼                                                           │
│  plot_results.py                                                  │
│       │  → artifacts/plots/lesson_<N>_*.png                       │
│       ▼                                                           │
│  export_demo_artifacts.py                                         │
│       └─ → artifacts/demo/*.npy + *.json (for webapp)             │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Evaluation Protocol

```
For each model:
    │
    ├── Full-rank evaluation (primary metric)
    │   Score ALL items → Recall@10, NDCG@10
    │   Reflects real-world ranking difficulty
    │
    └── Sampled evaluation (secondary metric)
        Score 1 positive + 99 random negatives → HR@10, NDCG@10
        Easier, used for sanity checks

Split: Leave-Last-Out (LLO)
    • Per user: last game interaction → test
    • Second-to-last → validation (early stopping)
    • Rest → training

Positive threshold: rating ≥ 4 (POSITIVE_THRESHOLD = 4)
All metrics computed at K = 10 only
```

---

## 7. Deployment Architecture

### 7.1 Development Setup

```bash
# ML Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python ml/data/process_data.py          # Process raw data
PYTHONPATH=. python ml/scripts/export_demo_artifacts.py  # Train + export (~6 min)

# Backend
PYTHONPATH=. uvicorn backend.demo.main:app --port 8000

# Frontend
cd frontend && npm install && npx next dev --port 3000
```

### 7.2 Component Dependencies

```
Frontend (Next.js)
    │ HTTP calls to localhost:8000
    ▼
Backend (FastAPI)
    │ Loads numpy arrays from artifacts/demo/
    │ Reads/writes SQLite at data/demo.db
    ▼
ML Pipeline (offline)
    │ Reads processed parquet from ml/data/amazon_2023/
    │ Writes embeddings to artifacts/demo/
    ▼
Raw Data (Amazon Reviews 2023)
    └── ml/data/amazon_2023/raw/
```

### 7.3 Background Services

```
┌─────────────────────────────────────────────┐
│           Retrain Scheduler                  │
│                                             │
│  Thread: daemon, non-blocking               │
│  Interval: 3600s (1 hour)                   │
│  Action:                                    │
│    1. Read current user_ratings from store   │
│    2. Train LightGCN (20 epochs, ~40s)      │
│    3. Hot-swap store.lgcn_user/item          │
│    4. Log retrain event                      │
│                                             │
│  API triggers:                              │
│    POST /api/retrain       (immediate)      │
│    GET  /api/retrain/status (last run info) │
└─────────────────────────────────────────────┘
```

---

## 8. Directory Structure

```
NewCrossDomainRecommenders/
│
├── ml/                              # ML pipeline
│   ├── data/
│   │   ├── amazon_2023/
│   │   │   ├── raw/                 # Raw JSONL (gitignored)
│   │   │   ├── processed_overlap/   # 19K users, 100% overlap
│   │   │   ├── processed_sparse_loose/ # 14K users, movies≥10
│   │   │   └── movies_games.db      # Full SQLite catalog (419MB)
│   │   ├── process_data.py
│   │   ├── data_splitter.py
│   │   ├── dataset.py
│   │   └── item_dedup.py
│   ├── models/
│   │   ├── base_recommender.py      # Abstract base
│   │   ├── _cdr_base.py             # RecBole-CDR shared setup
│   │   ├── lightgcn.py              # LightGCN (PyTorch)
│   │   ├── ncf.py                   # NeuMF (RecBole)
│   │   ├── matrix_factorization_bpr.py  # BPR (numpy)
│   │   ├── cmf.py                   # CMF (RecBole-CDR)
│   │   ├── emcdr.py                 # EMCDR (RecBole-CDR)
│   │   ├── ptupcdr.py               # PTUPCDR (RecBole-CDR)
│   │   ├── sbert_model.py           # SBERT (sentence-transformers)
│   │   ├── id_utils.py              # ID normalization
│   │   └── recbole_cdr/             # Vendored RecBole-CDR
│   ├── evaluation/
│   │   ├── evaluator.py             # Full-rank + sampled eval
│   │   └── metrics.py               # Recall, NDCG, HR @10
│   └── scripts/
│       ├── benchmarks/
│       │   ├── benchmark_common.py   # Shared eval framework
│       │   ├── bench_lightgcn.py
│       │   ├── bench_ncf.py
│       │   ├── bench_mf_bpr.py
│       │   ├── bench_cmf.py
│       │   ├── bench_emcdr.py
│       │   ├── bench_ptupcdr.py
│       │   ├── bench_sbert.py
│       │   ├── bench_sbert_cdr.py
│       │   ├── bench_cooc_l3.py
│       │   ├── bench_user_split_coldstart.py
│       │   └── cooc_rerank.py
│       ├── export_demo_artifacts.py
│       ├── plot_results.py
│       └── run_parallel_benchmarks.py
│
├── backend/                         # Web API
│   └── demo/
│       ├── main.py                  # FastAPI app + routes
│       ├── store.py                 # In-memory embedding store
│       ├── recommender.py           # 9-row hybrid engine
│       ├── database.py              # SQLite persistence
│       ├── retrain.py               # Background LightGCN retrain
│       └── graph.py                 # Explanation graphs
│
├── frontend/                        # Web UI
│   ├── app/
│   │   ├── page.tsx                 # Home (user groups)
│   │   ├── recommendations/page.tsx # 9 rec rows
│   │   └── item/[id]/page.tsx       # Item detail + similar
│   ├── lib/api.ts                   # Backend API client
│   └── types/index.ts               # TypeScript interfaces
│
├── artifacts/
│   ├── demo/                        # Exported model embeddings
│   ├── results/                     # Benchmark JSONs
│   └── plots/                       # Lesson comparison PNGs
│
├── summaries/                       # Per-lesson experiment summaries
├── report_figures/                  # Generated diagrams
│
├── LESSON_PLAN.md                   # 8-lesson experiment design
├── WEBAPP_SPECS.md                  # Frontend/backend specs
├── SYSTEM_ARCHITECTURE.md           # This document
├── CLAUDE.md                        # Development guidelines
├── requirements.txt                 # Python dependencies
├── generate_report.py               # Word report generator
└── generate_diagrams.py             # Matplotlib diagram generator
```

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual item space (SD + CD)** | Single-domain models (LightGCN, NCF) need a compact game-only space for efficient training. CDR models (EMCDR, PTUPCDR) need unified movie+game space for cross-domain transfer. Both are served from the same store. |
| **In-memory store vs database** | Model embeddings are numpy arrays — dot product scoring is orders of magnitude faster in memory than any DB query. SQLite is only used for rating persistence across restarts. |
| **Cooc as post-processing** | Co-occurrence is training-free, model-agnostic, and instant. Applying it as a score bonus (rather than a separate model) lets it improve every model universally (Lesson 8 finding). |
| **Reverse cooc for movies** | Rather than training a full movie→game CDR model in reverse, the game→movie cooc matrix provides meaningful movie recommendations with zero training cost. |
| **Hourly retrain (not real-time)** | Full LightGCN retrain takes ~40s on 58K interactions. Real-time per-rating retraining is impractical. Cooc/SBERT fill the freshness gap between retrains. |
| **external_id as canonical key** | Three index spaces (DB, SD, CD) create mapping complexity. Using the Amazon ASIN as the universal identifier eliminates off-by-one errors in item navigation. |
| **User groups on home page** | The four group tabs (Cold Start, 1-Shot, Few, Balanced) let users see how the simplified two-lane routing rule (0 games → EMCDR+cooc; ≥1 game → LightGCN+cooc) plays out across history depths — the experimental findings from Lessons 6 and 8 made directly interactive. |
| **SBERT for both domains** | SBERT operates in a shared text space — "Halo" the game and "Halo" the movie are naturally close. No domain-specific training needed for cross-domain content matching. |
| **9 rows (5 game + 4 movie)** | Each row demonstrates a different recommendation paradigm. Together they cover collaborative (LightGCN), cross-domain (CDR), content (SBERT), behavioral (cooc), and popularity — the full spectrum of our research. |
