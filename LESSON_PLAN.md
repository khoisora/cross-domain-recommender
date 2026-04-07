# CDR Experiment Lesson Plan — Movies → Games

> **Purpose**: Structured experiment plan for building benchmark infrastructure and code, lesson by lesson, in a clean new repo. **Dataset**: Amazon 2023 reviews, movie↔game domain pair. **Fixed positive threshold**: rating ≥ 4 (consistent across all lessons). **Fixed metrics**: Recall@10, NDCG@10 (full-rank) + sampled HR@10 and NDCG@10 (1 pos + 99 neg). All @10 only. **Split**: Per-user leave-last-out (LLO) on target-domain (game) interactions, unless stated otherwise.

---

## Model portfolio

### Single-domain (game-only training)

Model

Family

Base mechanism

Old repo reference

MF-BPR

Matrix factorization

BPR pairwise loss on implicit game interactions

`ml/models/matrix_factorization_bpr.py`

NCF (NeuMF)

Neural

GMF + MLP fusion on game interactions

`ml/models/ncf.py`

LightGCN

Graph

GCN on game bipartite graph

`ml/models/lightgcn.py`

### Cross-domain (movie → game transfer)

Model

Family

Transfer mechanism

Cold-start capable?

Old repo reference

CMF

Matrix factorization

Joint factorization, shared user factors across movies + games

Yes — movie edges keep user factors alive

`ml/models/cmf.py`

EMCDR

Mapping

Separate MF per domain → global MLP maps source user embedding → target space

Yes — mapping works from movie embedding alone

`ml/models/emcdr.py`

PTUPCDR

Personalized mapping

MF base + per-user hypernetwork (MoE) maps movie preference → game embedding; few-shot blend`1/(1+k)`

Yes — pure movie transfer when k=0

`ml/models/ptupcdr.py`

### Content / Semantic

Model

Family

Mechanism

Old repo reference

SBERT

Content (single-domain)

User embedding = mean of game item SBERT vectors; rank by cosine

`ml/models/sbert_model.py`

SBERT-CDR

Content (cross-domain)

User embedding = mean of**movie** item SBERT vectors; rank games by cosine in shared text space

`ml/scripts/benchmarks/bench_sbert_cdr.py`

### Baselines

Model

Mechanism

MF-Explicit

SVD, squared error on raw star ratings (Lesson 1 only — shows why explicit objective fails at ranking)

Popularity

Rank by global game popularity (Lesson 6 — shows that first game choice is heavily popularity-driven)

### Model presence per lesson

Model

L1

L2

L3

L4

L5

L6 (cold)

L7

MF-Explicit

✓

—

—

—

—

—

—

MF-BPR

✓

✓

✓

✓

—

✓ (fails)

—

NCF

—

✓

✓

✓

—

✓ (fails)

—

LightGCN

—

✓

✓

✓

✓

✓ (fails)

✓

CMF

—

✓

✓

✓

✓

✓ (wins)

—

EMCDR

—

✓

✓

✓

✓

✓ (wins)

—

PTUPCDR

—

✓

✓

✓

✓

✓ (wins)

✓

SBERT

—

—

—

—

—

—

✓

SBERT-CDR

—

—

—

—

—

—

✓ (wins on niche)

Popularity

—

—

—

—

—

✓

—

**Lesson 6 narrative**: All three single-domain models (MF-BPR, NCF, LightGCN) fail on cold users — zero game edges → scores collapse to bias/popularity. All three CDR models (CMF, EMCDR, PTUPCDR) win — movie signal transfers through shared factors or learned mappings. Popularity is a surprisingly strong baseline at cold-start. This justifies a routing rule: no game history → CDR path.

---

## Repo & workflow rules

**New repo**: `~/work/NewCrossDomainRecommenders`

-   All new experiment code goes here. Do not modify the old repo.
-   The old repo at `~/work/MoviesGamesRecommender` is the **reference implementation**. Always read it before writing anything — copy and adapt logic, do not rewrite from scratch.

**Phased build**: Implement lessons in order. Each lesson adds only what it needs. Do not port the entire old codebase upfront.

**Phase 0 (bootstrap, before any lesson)**: Set up the base infrastructure that every lesson depends on. See the Phase 0 section below.

---

## Phase 0 — Base infrastructure (do this first)

Port the minimum shared code from `~/work/MoviesGamesRecommender` needed to run any lesson. Everything else gets added per-lesson.

### What to port / create

**1. Data processing**

-   Port `ml/data/process_data.py` from the old repo, but **only the core pipeline**: raw JSONL → parquet for the movie_game pair. **Convert to implicit first**: keep only ratings ≥ 4 (POSITIVE_THRESHOLD) before any k-core or sampling. Item k-core: movies ≥ 20, games ≥ 10. User k-core and user sampling are configurable via CLI args (`--min-user-interactions`, `--sample-users`). Do **not** filter for overlap users — that restriction is introduced in Lesson 3.
-   **Lesson 1–2 default**: no user k-core (`--min-user-interactions 0`), randomly sampled to ~1M users (`--sample-users 1000000`).
-   The initial processed output is simply `processed/` (movies + games + ratings parquets). All bench scripts in Lessons 1–2 use this same dataset.
-   Register only one domain pair in `_DOMAIN_PAIR_PATHS` for now: `"movie_game"` pointing at `processed/`.
-   **Run the data pipeline as part of Phase 0**: after writing `process_data.py`, execute it to generate the parquet files in `processed/`. Verify the output exists and matches expected counts from `dataset_metadata.json` before proceeding.

**2. Data loading & splitting**

-   Port the parquet reading logic from `ml/scripts/benchmarks/benchmark_common.py`: `load_cross_domain_split()`, `leave_last_out_split()`, the `CrossDomainSplit` dataclass.
-   `load_cross_domain_split()` for `"movie_game"` should just load the basic processed parquet and apply LLO — no cohort filtering at this stage.

**3. Evaluation**

-   Port `ml/evaluation/metrics.py` — Recall@K, NDCG@K, HitRate@K. Expose only @10.
-   Port `ml/evaluation/evaluator.py` — full-rank eval and sampled@99 eval. Keep both modes; `sampled=True` flag.
-   Port `evaluate_cross_domain()` from `benchmark_common.py`. No subgroup logic yet — subgroup analysis is introduced in Lesson 4.

**4. Benchmark common**

-   Create `benchmark_common.py` modelled on the old one: `save_result()` writing JSON to `artifacts/<domain-pair>/results/`, `setup_logging()`, CLI arg helpers (`--domain-pair`, `--target`).
-   Positive threshold constant: `POSITIVE_THRESHOLD = 4`. Import it everywhere; never hardcode.

**5. Base model interface**

-   Port `ml/models/base_recommender.py` — `BaseRecommender` and `BasePyTorchRecommender` abstract classes.
-   Port `ml/models/id_utils.py` — user/item ID mapping utilities.

**6. Visualization**

-   Create `ml/scripts/plot_results.py` — the single script all lessons use to generate charts. It reads the `artifacts/<domain-pair>/results/*.json` files produced by `save_result()` and renders comparison plots.
    
-   **Chart types to implement**:
    
    -   **Bar chart**: side-by-side bars per model for Recall@10 and NDCG@10 (full-rank and sampled, 2×2 or 2 subplots). One chart per lesson/domain-pair.
    -   **Subgroup bar chart**: grouped bars per subgroup (x-axis) with one bar per model (colors). Only rendered when subgroup data is present in the result JSON.
-   **Data characteristic annotation**: every chart must display a metadata box (e.g. as a figure subtitle or text annotation in the bottom-left corner) showing:
    
    -   Domain pair name
    -   Cohort filter condition (read from `"cohort_filter"` field in the result JSON)
    -   n_users, n_movie_interactions, n_game_interactions — also from the result JSON
    -   Split type (e.g. "LLO on games")
-   **`save_result()` contract** (update in `benchmark_common.py`): the JSON written must include a top-level `"dataset_info"` block:
    
    ```json
    "dataset_info": {
      "domain_pair": "movie_game",
      "cohort_filter": "k-core ≥ 10 (both movies and games)",
      "n_users": 7049,
      "n_movie_interactions": 195000,
      "n_game_interactions": 43000,
      "split": "leave-last-out on games"
    }
    ```
    
    **In Phase 0 / Lessons 1–2**, the `cohort_filter` value is read dynamically from `dataset_metadata.json` (written by `process_data.py`). Every bench script must pass this block when calling `save_result()`. `load_cross_domain_split()` should return it as part of `CrossDomainSplit` so bench scripts don't have to construct it manually. **Domain-specific cohort logic (including overlap-user filtering) is introduced in Lesson 3** — different cohort variants will have different `cohort_filter` strings.
    
-   **CLI**: `python plot_results.py --domain-pair movie_game --lesson 1` reads all result JSONs for that lesson tag and writes a PNG to `artifacts/<domain-pair>/plots/lesson_<N>_<timestamp>.png`.
    
-   Each bench script should accept a `--lesson` tag (integer) and pass it through to `save_result()` so plots can be scoped per lesson.
    

**7. Repo structure**

```
NewCrossDomainRecommenders/
  ml/
    data/
      process_data.py        # basic pipeline only, no cohort logic yet
      data_splitter.py
      dataset.py
    models/
      base_recommender.py
      id_utils.py
    evaluation/
      metrics.py
      evaluator.py
    scripts/
      benchmarks/
        benchmark_common.py  # one domain pair registered: "movie_game"
      plot_results.py         # shared visualization script
  artifacts/
    <domain-pair>/
      results/               # JSON per model run
      plots/                 # PNG per lesson
  requirements.txt
```

**8. SQLite dump**

-   After the data pipeline produces parquets, dump them into a SQLite database at `ml/data/amazon_2023/movies_games.db` using `ml/scripts/dump_to_sqlite.py`. This provides an easy-to-query view of the processed data for ad-hoc analysis and the backend API.
-   Run: `PYTHONPATH=. python ml/scripts/dump_to_sqlite.py`

**Verification**: After Phase 0, running `load_cross_domain_split("movie_game")` in a Python shell should return a valid `CrossDomainSplit` object with non-zero user/item counts. Exact numbers will depend on what the basic k-core filter produces — check against the old repo's `dataset_metadata.json` for rough expected counts.

---

## Lesson 1 — Explicit rating prediction vs implicit ranking

**Claim**: Models trained to minimize RMSE on star ratings optimize a different objective than top-10 ranking. BPR-trained implicit models outperform explicit MF on Recall@10 even at the same model capacity.

**Data**: `movie_game` (no user k-core filter, randomly sampled to ~1M users). **Split**: Standard LLO on games.

**Models to port from old repo**:

-   `ml/models/matrix_factorization.py` → `bench_mf_explicit.py` (explicit SVD, squared error on raw ratings)
-   `ml/models/matrix_factorization_bpr.py` → `bench_mf_bpr.py` (rating ≥ 4 = positive, pairwise BPR loss)

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. One table, two rows. **Plot**: `python plot_results.py --domain-pair movie_game --lesson 1` → bar chart, two models, annotated with dataset info.

**→ Lesson 2**: BPR is now the baseline family. Add graph and CDR models on the same protocol to see if architecture or cross-domain signal improve over BPR.

---

## Lesson 2 — Low overlap kills CDR: single-domain graph wins on mixed population

**Claim**: On a naturally mixed population with very low user overlap (~5.8%), CDR models underperform single-domain models. Cross-domain transfer needs shared users to learn from — without them, mapping functions train on noise.

**Data**: `movie_game` (same as Lesson 1: no user k-core, ~1M randomly sampled users). **Split**: Standard LLO on games.

**Models to port**:

-   `ml/models/lightgcn.py` → `bench_lightgcn.py` (game-only training)
-   `ml/models/ncf.py` → `bench_ncf.py` (game-only training)
-   `ml/models/cmf.py` → `bench_cmf.py`
-   `ml/models/emcdr.py` → `bench_emcdr.py`
-   `ml/models/ptupcdr.py` → `bench_ptupcdr.py`
-   Also port `ml/models/_cdr_base.py` (shared CDR training loop used by EMCDR / PTUPCDR)

All CDR scripts use `--domain-pair movie_game`.

Include MF-BPR from Lesson 1 as carry-forward baseline (no re-port needed).

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. One table, all models + % gap vs LightGCN. **Plot**: `--lesson 2` → bar chart, all models, LightGCN highlighted as reference bar, annotated with dataset info (including overlap %).

**→ Lesson 3**: Lesson 2 shows CDR fails at 5.8% overlap. Lesson 3 filters to 100% overlap users and demonstrates that CDR scores recover — proving overlap % is the key variable.

---

## Lesson 3 — Overlap filtering rescues CDR: 100% overlap users

**Claim**: Filtering to 100% overlap users (active in both domains) dramatically improves CDR performance compared to the low-overlap Lesson 2 population. This proves overlap % is the key variable for cross-domain transfer.

**Data**: `movie_game` — user k-core ≥ 10 total interactions (same as Phase 0), then filter to overlap users with ≥ 5 movie ratings AND ≥ 1 game rating. This becomes the default dataset for Lesson 3 — no separate cohort variants, no strict filter. All users are guaranteed to have activity in both domains (100% overlap). Stored in `processed_overlap/`. **Split**: Standard LLO on games.

**Models**: MF-BPR, NCF, LightGCN, CMF, EMCDR, PTUPCDR (all already ported from Lesson 2).

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. Compare against Lesson 2 results — CDR models should show large score increases while single-domain models may stay flat or decline (fewer users, less game data). **Plot**: `--lesson 3` → bar chart with Lesson 2 vs Lesson 3 comparison, annotated with overlap % change (5.8% → 100%).

**→ Lesson 4**: Lesson 3 controls who is in the data. Lesson 4 controls how much target-domain data those users have, stressing CDR toward the regime it was designed for.

---

## Lesson 4 — Source-rich / target-sparse: CDR closes the gap

**Claim**: CDR's relative advantage improves when users have rich movie history but sparse game history. Tightening the movie requirement from ≥ 5 to ≥ 10 gives CDR models richer source embeddings to transfer from.

**New in this lesson**: Introduce subgroup analysis in `benchmark_common.py`. Subgroup logic (cold-start groups, transfer groups, balance groups) is added to `load_cross_domain_split()` here — it does not exist in L1–L3.

**Data**: `movie_game` — k-core ≥ 10 total interactions, then overlap filter: movies ≥ 10, games ≥ 1. This is the default dataset for Lesson 4 — no separate cohort variants. Stored in `processed_sparse_loose/`. **Split**: Standard LLO on games.

**Models**: MF-BPR, NCF, LightGCN, CMF, EMCDR, PTUPCDR (all already ported).

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. Compare against Lesson 3 results. **Plot**: `--lesson 4` → bar chart with Lesson 3 vs Lesson 4 comparison.

**→ Lesson 5**: Lessons 2–4 vary the user population. Lesson 5 varies the item catalog — filtering out weakly transferable items.

---

## Lesson 5 — Catalog sharpening: genre filter + overlap-user item filter

**Claim**: Removing genre-mismatched and overlap-user-irrelevant items reduces embedding noise and should improve CDR relative to single-domain.

**Data**: Start from Lesson 4 dataset (movies ≥ 10, games ≥ 1). Apply three interventions as ablations:

1.  **Genre whitelist**: drop movie-only genres with no game analog (Documentary, Exercise DVDs, Musicals, Classical).
2.  **Overlap-user item filter**: drop items never rated by any overlap user.
3.  **Movie popularity filter (>=50 ratings)**: drop long-tail movie items with <50 ratings. Analysis shows this reduces items from 52K→10K, increases interaction matrix density 3.1×, while retaining 67% of overlap user movie interactions and 73% of overlap users with >=5 ratings. Denser source embeddings → cleaner transfer mapping for EMCDR/PTUPCDR.

Register filtered variant as `movie_game_filtered`. Reference `processed_transfer_loose_filtered/` in the old repo for the existing implementation in `process_data.py`.

**Split**: Standard LLO on games. **Models**: LightGCN, CMF, EMCDR, PTUPCDR.

**Report**: Five-column table: Unfiltered (from Lesson 4) | Genre-only | Item-filter-only | Popularity-only | All three. Rows: LightGCN and PTUPCDR at minimum, other models optional. If a null result appears (filter did not help), report it — do not omit. **Plot**: `--lesson 5` → bar chart with five filter-variant groups on x-axis. Annotation box per group shows the active filters and resulting n_items (movies and games separately) so catalog shrinkage is visible.

**→ Lesson 6**: Lessons 2–5 all use LLO so even cold users have ≥ 1 game in training. Lesson 6 removes all game edges for a held-out user set — the true zero-shot regime.

---

## Lesson 6 — User-split cold-start protocol

**Claim**: When all game interactions are hidden for a subset of users during training, CDR outperforms single-domain models by 2–5× on that subset. This justifies a routing rule: no game history → CDR path.

**Data**: `movie_game` (from Lesson 3: overlap users, movies ≥ 5, games ≥ 1). User split: 80% warm / 20% cold. **Split**: Port `load_user_split_cold_start_split()` from `ml/scripts/benchmarks/bench_user_split_coldstart.py` in the old repo → new `bench_user_split_coldstart.py`.

**Models — single-domain (expected to fail on cold users)**:

-   MF-BPR — no game edges → scores collapse to global bias
-   NCF — no game edges → random-like scores
-   LightGCN — no game edges → near-popularity behavior

**Models — cross-domain (expected to win on cold users)**:

-   CMF — movie edges keep user factors alive in the shared factorization
-   EMCDR — mapping(movie_embedding) → game space, works without any game history
-   PTUPCDR — MoE hypernetwork maps movie preference → game embedding; blend weight = `1/(1+0)` = pure movie transfer

**Baselines**:

-   Popularity — rank by global game popularity (surprisingly strong at cold-start: first game choice is often a well-known title)

**Report**: Cold-user evaluation only. Full-rank Recall@10 + NDCG@10 AND sampled HR@10 + NDCG@10 (both, because full-rank understates CDR advantage here). **Plot**: `--lesson 6` → side-by-side bar chart, cold users only. Color-code bars by type (single-domain = grey, cross-domain = blue, baseline = orange). Annotation box shows warm/cold split ratio and n_cold_users. Include a second small panel comparing full-rank vs sampled NDCG@10 for the same models, to illustrate why both protocols are needed.

**→ Lesson 7**: Lessons 1–6 are purely collaborative. Lesson 7 adds item text (SBERT) as a content bridge for cases where collaborative overlap is too thin.

---

## Lesson 7 — Content-aware CDR (SBERT)

**Claim**: When collaborative CDR is noisy or user overlap is low, SBERT item embeddings can bridge movie and game content in a shared semantic space — particularly for unpopular target items.

**Data**: `movie_game` (from Lesson 4, or `movie_game_filtered` from Lesson 5 if filtering helped). **Split**: Standard LLO on games.

**Models to port**:

-   `ml/models/sbert_model.py` → `bench_sbert.py` (in-domain SBERT: user vec = mean of game item embeddings)
-   `ml/scripts/benchmarks/bench_sbert_cdr.py` → `bench_sbert_cdr.py` (SBERT-CDR: user vec = mean of **movie** item embeddings, rank games by cosine in shared text space)
-   Compare against LightGCN and PTUPCDR (already ported).

**Report**: Overall Recall@10 / NDCG@10 + subgroup focus on `high_source_unpopular_low_target` and `one_shot_unpopular_target_user`. If SBERT-CDR does not beat collaborative CDR overall, report it and explain (text similarity ≠ behavioral preference). **Plot**: `--lesson 7` → two-panel chart: left = overall bar chart (all four models), right = subgroup bar chart focused on the two unpopular slices. Annotation box shows dataset info and SBERT embedding dimensions.

**→ Lesson 8**: Lessons 1–7 evaluate models in isolation. Lesson 8 asks: can a lightweight, training-free post-processing step (movie→game co-occurrence reranking) improve every model across every regime?

---

## Lesson 8 — Co-occurrence reranking as universal post-processing

**Claim**: Movie→game behavioral co-occurrence (`cooc_rerank.py`) improves every model in every regime — LLO and cold-start — because it adds item-level cross-domain signal that embedding-based methods compress away. The gain is inversely proportional to how well the base model already uses movie signal.

**What changes vs Lesson 7**: Lesson 8 does not add a new model. It wraps all existing models with a test-time reranking layer and measures the delta. The training pipeline is unchanged.

**Data**: Two regimes tested:
- **LLO (L3 dataset)**: `movie_game_overlap` (100% overlap users, movies ≥ 5, games ≥ 1). All users have movie history → cooc has signal for everyone.
- **Cold-start (L6 split)**: `movie_game` user-split (80/20 warm/cold). Cold users have zero game training edges → cooc is the only cross-domain signal available.

**Cooc mechanism** (`cooc_rerank.py`):
- Build `cooc[movie_id][game_id]` = log(1 + count of overlap users who liked both), from training data only (no leakage).
- At inference: for user U, boost game scores by `lam × Σ cooc[m][game]` over U's positive movie history.
- Gate with `max_target_train=0` on cold-start paths to avoid applying to warm users.
- `lam=0.05` default; score mode `raw` (log-count).

**Models**: All from Lessons 2–7. Run base + cooc variant for each.

**Bench script**: `ml/scripts/benchmarks/bench_cooc_l3.py` (LLO) and `ml/scripts/benchmarks/bench_user_split_coldstart.py` (cold-start, already includes cooc variants).

**Report**: Two tables — LLO and cold-start — showing base Recall@10, cooc Recall@10, and % delta per model. Highlight the pattern: gain magnitude correlates with how broken the base model is at using movie signal.

**Key findings**:

| Regime | Model | Base | +cooc | Δ |
|---|---|---|---|---|
| LLO | MF-BPR | 0.0445 | 0.0520 | +17% |
| LLO | NCF | 0.0255 | 0.0370 | +45% |
| LLO | LightGCN | 0.0595 | 0.0625 | +5% |
| LLO | CMF (fixed) | 0.0395 | 0.0380 | -4% (cooc hurts — CDR already captures signal) |
| LLO | PTUPCDR | 0.0315 | 0.0355 | +13% |
| LLO | EMCDR | 0.0225 | 0.0335 | +49% |
| Cold | LightGCN | 0.0067 | 0.0261 | +289% |
| Cold | EMCDR | 0.0334 | 0.0341 | +2% |
| Cold | CMF | 0.0007 | 0.0321 | +45× |
| Cold | Popularity | 0.0381 | 0.0387 | +1.6% (noise) |

**Why cooc helps each model type**:
- **Game-only models** (MF-BPR, NCF, LightGCN): cooc adds an entirely missing dimension — movie history is invisible to them; cooc injects it all at inference time.
- **Joint factorization (properly tuned CMF)**: once fixed (alpha=0.05, lr=0.0005), CMF already captures cross-domain signal well — cooc slightly hurts (-4%) by adding redundant/conflicting signal. Broken CMF (alpha=0.3, lr=0.01) gained +70% from cooc, which was masking the misconfiguration.
- **Global mapping CDR** (EMCDR): MLP learns population-average movie→game transfer; cooc is personalized per user/item pair — complementary granularity.
- **Personalized mapping CDR** (PTUPCDR): MoE captures user clusters, not item-level co-occurrence; cooc fills that gap.
- **Popularity**: already a strong cold-start prior; popular games dominate co-occurrence counts, so the two signals are highly correlated — no new information.

**Updated routing rule**: Cooc is added as universal post-processing to all routing paths. Cold-start preference updated: EMCDR+cooc preferred over PTUPCDR+cooc (EMCDR's global MLP works without game edges; PTUPCDR's few-shot blend requires game history to activate). Niche path updated: SBERT-CDR+cooc replaces SBERT-CDR alone.

**Plot**: `--lesson 8` → two-panel bar chart: left = LLO base vs cooc per model, right = cold-start base vs cooc per model. Bars paired (base + cooc side by side per model). Annotation box shows lam value and cooc edge count.

---

## Summary: the decision rule all lessons justify

> **Cooc as universal post-processing**: movie→game co-occurrence reranking (`cooc_rerank.py`) improves every model in every regime (L3 LLO: +9% to +2850%; L6 cold-start: +2% to +45×). Apply it to all paths below. The only exception is Popularity, where cooc adds noise (+1.6%). Gate with `max_target_train=0` for cold-start paths so warm users are unaffected.

User state

Recommended model

Justified by

0 games, rich movies

**EMCDR + cooc** (popularity blend as floor)

L6: EMCDR most robust at true cold-start (global MLP mapping works without game edges); PTUPCDR needs game history to activate few-shot blend. L8: cooc adds small additional boost (+2%)

1–2 games, rich movies

**PTUPCDR + cooc**

L4, L6: few-shot blend activates once game edges exist. L8: cooc adds +13% on LLO

3–9 games

**LightGCN + cooc**

L2, L4: best LLO base model. L8: cooc +5%, fills movie→game gap graph alone misses

10+ games

**LightGCN + cooc**

L2: LightGCN dominates. L8: cooc consistently helps even with rich game history

Niche / unpopular games

**SBERT-CDR + cooc**

L7, L6 cooc: SBERT-CDR alone weak (Recall=0.002); cooc provides behavioral anchor that rescues it (→0.024); semantic + co-occurrence signals are complementary for niche items

---

## Implementation rules for Claude Code

### Git branching

-   **Create a new branch for each lesson/phase**: `phase-0`, `lesson-1`, `lesson-2`, etc.
-   **When a lesson/phase is complete** (bench scripts run, plots saved), merge the branch to `main` with `--no-ff` (no fast-forward) to preserve a merge commit, then create the next lesson branch from `main`.
-   **Use the current branch name to know which lesson you are working on.** Do not skip ahead.

### Conversation workflow

-   **One lesson per conversation.** Each new conversation should check which lesson/phase is current (via branch name or last merged branch) and implement only that one. Do not execute all lessons in a single conversation.
-   **Before starting a new lesson**, verify the previous lesson is fully merged to `main` with passing bench scripts and saved plots.
-   **Each lesson must end with**: (1) bench scripts run successfully, (2) result JSONs saved, (3) plot PNGs generated, (4) summary write-up saved (see below), (5) branch merged to `main`.

### Core workflow

-   **Always read the old repo first** (`~/work/MoviesGamesRecommender`) before writing any model, loader, or eval logic. Port and adapt — don't rewrite from scratch. Fix bugs in old repo code if encountered during porting. The old repo may contain messy or overly complex logic — only port what is relevant to the current phase/lesson. Strip irrelevant paths, dead code, and over-engineered abstractions. Keep ported code simple, minimal, and well-commented.
-   **One file per model, one file per bench script.** Mirror the old repo's naming (`bench_<model>.py`, `ml/models/<model>.py`).
-   **Do not start Lesson N+1 until Lesson N's bench scripts run end-to-end and the plot PNG is saved.**
-   **RecBole-CDR** is vendored at `ml/models/recbole_cdr/` from the start. Do not pip install it; assume it's always available.

### Hardware

-   **Use GPU (CUDA/MPS) when available**, fall back to CPU. PyTorch models should auto-detect device. Use whichever is faster.
-   **Hyperparameters**: start with the old repo's hyperparameters. If a model trains slowly (>10 min for a single run), tune down epochs/embedding size to keep iteration fast. Document any changes.

### Code quality & libraries

-   **Use only publicly available libraries.** For example: pandas, numpy, PyTorch, scikit-learn, matplotlib, seaborn. Refrain entirely from self-implementation of algorithms, utilities, or helpers — always prefer existing well-maintained packages.
-   **Keep code logic simple and minimal.** Prioritize readability and directness over cleverness. Minimize lines of code. If a model/loader/evaluator is getting long (>200 lines), check if it can be simplified by using library functions or removing redundant logic.
-   **Avoid custom classes and wrappers where standard library alternatives exist.** For example, use pandas DataFrames directly rather than custom data containers; use sklearn pipelines rather than custom preprocessing chains.
-   **Add clear comments for non-obvious logic.** Every file should have a module docstring explaining its role in the pipeline. Add inline comments for: design decisions (why this approach?), data flow (what goes in/out), non-obvious thresholds or constants, and any logic that would take >5 seconds to understand when re-reading. Do not comment trivial code.

### Data and evaluation

-   **All metrics are @10 only.** No @5, @20, @50 in any output or report.
-   **Single evaluation entry point**: all bench scripts call `evaluate_cross_domain()` from `benchmark_common.py`. Do not inline eval logic in bench scripts.
-   **Positive threshold**: `POSITIVE_THRESHOLD = 4`, defined once in `benchmark_common.py`, imported everywhere.
-   **New cohort variants**: add the new `--domain-pair` key to `_DOMAIN_PAIR_PATHS` in `benchmark_common.py`. Do not hardcode data paths in bench scripts.

### Artifacts & outputs

-   **Artifacts**: every bench script writes a JSON result via `save_result()` to `artifacts/<domain-pair>/results/<model>_lesson<N>.json`. The filename must include the lesson number so results from different lessons don't overwrite each other.
-   **`save_result()` must always include a `"dataset_info"` block** (see Phase 0 spec). If a bench script does not have all fields, it must at minimum include `domain_pair`, `n_users`, and `cohort_filter`.
-   **Plotting**: after all bench scripts for a lesson finish, always run `python plot_results.py --domain-pair <pair> --lesson <N>`. The plot is the deliverable, not just the JSON.
-   **Annotation box content**: every plot must show — domain pair, cohort filter string, n_users, n_movie_interactions, n_game_interactions. These come from `dataset_info` in the JSON; do not hardcode them in the plot script.

### Phase/lesson summaries

After completing each phase or lesson, write a detailed summary markdown file. Structure:

```
summaries/
  phase-0/
    summary.md
  lesson-1/
    summary.md
  lesson-2/
    summary.md
  ...
```

Each `summary.md` must include:

1.  **What changed vs previous lesson** — a short section at the top listing (a) what variables changed from the previous lesson and why, and (b) what was kept constant. This makes the experimental design legible across lessons.
2.  **File changes** — list of files added/modified with a one-line description of what changed and why.
3.  **Dataset characteristics** — table showing cohort filter, n_users, n_items (per domain), n_interactions (per domain), overlap users, sparsity.
4.  **Benchmark results** — table(s) of Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10 across all models. Include subgroup breakdowns if applicable.
5.  **Benchmark plots** — embed or reference the plot PNGs generated for this lesson (relative path to `artifacts/`).
6.  **Key takeaways** — what was learned, what surprised, what matched or contradicted the lesson's original claim/plan. Be honest about null results.