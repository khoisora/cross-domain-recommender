# CDR Experiment Lesson Plan — Movies → Games

> **Purpose**: Structured experiment plan for building benchmark infrastructure and code, lesson by lesson, in a clean new repo.
> **Dataset**: Amazon 2023 reviews, movie↔game domain pair.
> **Fixed positive threshold**: rating ≥ 4 (consistent across all lessons).
> **Fixed metrics**: Recall@10, NDCG@10 (full-rank) + sampled HR@10 and NDCG@10 (1 pos + 99 neg). All @10 only.
> **Split**: Per-user leave-last-out (LLO) on target-domain (game) interactions, unless stated otherwise.

---

## Model portfolio

### Single-domain (game-only training)

| Model | Family | Base mechanism | Old repo reference |
|---|---|---|---|
| MF-BPR | Matrix factorization | BPR pairwise loss on implicit game interactions | `ml/models/matrix_factorization_bpr.py` |
| NCF (NeuMF) | Neural | GMF + MLP fusion on game interactions | `ml/models/ncf.py` |
| LightGCN | Graph | GCN on game bipartite graph | `ml/models/lightgcn.py` |

### Cross-domain (movie → game transfer)

| Model | Family | Transfer mechanism | Cold-start capable? | Old repo reference |
|---|---|---|---|---|
| CMF | Matrix factorization | Joint factorization, shared user factors across movies + games | Yes — movie edges keep user factors alive | `ml/models/cmf.py` |
| EMCDR | Mapping | Separate MF per domain → global MLP maps source user embedding → target space | Yes — mapping works from movie embedding alone | `ml/models/emcdr.py` |
| PTUPCDR | Personalized mapping | MF base + per-user hypernetwork (MoE) maps movie preference → game embedding; few-shot blend `1/(1+k)` | Yes — pure movie transfer when k=0 | `ml/models/ptupcdr.py` |

### Content / Semantic

| Model | Family | Mechanism | Old repo reference |
|---|---|---|---|
| SBERT | Content (single-domain) | User embedding = mean of game item SBERT vectors; rank by cosine | `ml/models/sbert_model.py` |
| SBERT-CDR | Content (cross-domain) | User embedding = mean of **movie** item SBERT vectors; rank games by cosine in shared text space | `ml/scripts/benchmarks/bench_sbert_cdr.py` |

### Baselines

| Model | Mechanism |
|---|---|
| MF-Explicit | SVD, squared error on raw star ratings (Lesson 1 only — shows why explicit objective fails at ranking) |
| Popularity | Rank by global game popularity (Lesson 6 — shows that first game choice is heavily popularity-driven) |

### Model presence per lesson

| Model | L1 | L2 | L3 | L4 | L5 | L6 (cold) | L7 |
|---|---|---|---|---|---|---|---|
| MF-Explicit | ✓ | — | — | — | — | — | — |
| MF-BPR | ✓ | ✓ | ✓ | ✓ | — | ✓ (fails) | — |
| NCF | — | ✓ | ✓ | ✓ | — | ✓ (fails) | — |
| LightGCN | — | ✓ | ✓ | ✓ | ✓ | ✓ (fails) | ✓ |
| CMF | — | ✓ | ✓ | ✓ | ✓ | ✓ (wins) | — |
| EMCDR | — | ✓ | ✓ | ✓ | ✓ | ✓ (wins) | — |
| PTUPCDR | — | ✓ | ✓ | ✓ | ✓ | ✓ (wins) | ✓ |
| SBERT | — | — | — | — | — | — | ✓ |
| SBERT-CDR | — | — | — | — | — | — | ✓ (wins on niche) |
| Popularity | — | — | — | — | — | ✓ | — |

**Lesson 6 narrative**: All three single-domain models (MF-BPR, NCF, LightGCN) fail on cold users — zero game edges → scores collapse to bias/popularity. All three CDR models (CMF, EMCDR, PTUPCDR) win — movie signal transfers through shared factors or learned mappings. Popularity is a surprisingly strong baseline at cold-start. This justifies a routing rule: no game history → CDR path.

---

## Repo & workflow rules

**New repo**: `~/work/NewCrossDomainRecommenders`
- All new experiment code goes here. Do not modify the old repo.
- The old repo at `~/work/MoviesGamesRecommender` is the **reference implementation**. Always read it before writing anything — copy and adapt logic, do not rewrite from scratch.

**Phased build**: Implement lessons in order. Each lesson adds only what it needs. Do not port the entire old codebase upfront.

**Phase 0 (bootstrap, before any lesson)**: Set up the base infrastructure that every lesson depends on. See the Phase 0 section below.

---

## Phase 0 — Base infrastructure (do this first)

Port the minimum shared code from `~/work/MoviesGamesRecommender` needed to run any lesson. Everything else gets added per-lesson.

### What to port / create

**1. Data processing**
- Port `ml/data/process_data.py` from the old repo, but **only the core pipeline**: raw JSONL → parquet for the movie_game pair, with a **single, uniform k-core filter**: users with ≥ 10 interactions **in each domain** (movies AND games). Do not port any transfer/cohort-specific logic yet — that gets added lesson by lesson.
- The initial processed output is simply `processed/` (movies + games + ratings parquets). All bench scripts in Lessons 1–2 use this same filtered dataset with no further domain-specific cohort variants.
- Register only one domain pair in `_DOMAIN_PAIR_PATHS` for now: `"movie_game"` pointing at `processed/`.

**2. Data loading & splitting**
- Port the parquet reading logic from `ml/scripts/benchmarks/benchmark_common.py`: `load_cross_domain_split()`, `leave_last_out_split()`, the `CrossDomainSplit` dataclass.
- `load_cross_domain_split()` for `"movie_game"` should just load the basic processed parquet and apply LLO — no cohort filtering at this stage.

**3. Evaluation**
- Port `ml/evaluation/metrics.py` — Recall@K, NDCG@K, HitRate@K. Expose only @10.
- Port `ml/evaluation/evaluator.py` — full-rank eval and sampled@99 eval. Keep both modes; `sampled=True` flag.
- Port `evaluate_cross_domain()` from `benchmark_common.py`. Include subgroup logic — but subgroups only activate when the cohort actually contains those users; they silently return empty if no users qualify.

**4. Benchmark common**
- Create `benchmark_common.py` modelled on the old one: `save_result()` writing JSON to `artifacts/<domain-pair>/results/`, `setup_logging()`, CLI arg helpers (`--domain-pair`, `--target`).
- Positive threshold constant: `POSITIVE_THRESHOLD = 4`. Import it everywhere; never hardcode.

**5. Base model interface**
- Port `ml/models/base_recommender.py` — `BaseRecommender` and `BasePyTorchRecommender` abstract classes.
- Port `ml/models/id_utils.py` — user/item ID mapping utilities.

**6. Visualization**
- Create `ml/scripts/plot_results.py` — the single script all lessons use to generate charts. It reads the `artifacts/<domain-pair>/results/*.json` files produced by `save_result()` and renders comparison plots.
- **Chart types to implement**:
  - **Bar chart**: side-by-side bars per model for Recall@10 and NDCG@10 (full-rank and sampled, 2×2 or 2 subplots). One chart per lesson/domain-pair.
  - **Subgroup bar chart**: grouped bars per subgroup (x-axis) with one bar per model (colors). Only rendered when subgroup data is present in the result JSON.
- **Data characteristic annotation**: every chart must display a metadata box (e.g. as a figure subtitle or text annotation in the bottom-left corner) showing:
  - Domain pair name
  - Cohort filter condition (read from `"cohort_filter"` field in the result JSON)
  - n_users, n_movie_interactions, n_game_interactions — also from the result JSON
  - Split type (e.g. "LLO on games")
- **`save_result()` contract** (update in `benchmark_common.py`): the JSON written must include a top-level `"dataset_info"` block:
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
  **In Phase 0**, all scripts use the same `cohort_filter` value: `"k-core ≥ 10 (both movies and games)"`. Every bench script must pass this block when calling `save_result()`. `load_cross_domain_split()` should return it as part of `CrossDomainSplit` so bench scripts don't have to construct it manually. **Domain-specific cohort logic is introduced in Lesson 3** — different cohort variants will have different `cohort_filter` strings.
- **CLI**: `python plot_results.py --domain-pair movie_game --lesson 1` reads all result JSONs for that lesson tag and writes a PNG to `artifacts/<domain-pair>/plots/lesson_<N>_<timestamp>.png`.
- Each bench script should accept a `--lesson` tag (integer) and pass it through to `save_result()` so plots can be scoped per lesson.

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

**Verification**: After Phase 0, running `load_cross_domain_split("movie_game")` in a Python shell should return a valid `CrossDomainSplit` object with non-zero user/item counts. Exact numbers will depend on what the basic k-core filter produces — check against the old repo's `dataset_metadata.json` for rough expected counts.

---

## Lesson 1 — Explicit rating prediction vs implicit ranking

**Claim**: Models trained to minimize RMSE on star ratings optimize a different objective than top-10 ranking. BPR-trained implicit models outperform explicit MF on Recall@10 even at the same model capacity.

**Data**: `movie_game` (basic processed from Phase 0: k-core ≥ 10 both movies and games, no further filtering).
**Split**: Standard LLO on games.

**Models to port from old repo**:
- `ml/models/matrix_factorization.py` → `bench_mf_explicit.py` (explicit SVD, squared error on raw ratings)
- `ml/models/matrix_factorization_bpr.py` → `bench_mf_bpr.py` (rating ≥ 4 = positive, pairwise BPR loss)

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. One table, two rows.
**Plot**: `python plot_results.py --domain-pair movie_game --lesson 1` → bar chart, two models, annotated with dataset info.

**→ Lesson 2**: BPR is now the baseline family. Add graph and CDR models on the same protocol to see if architecture or cross-domain signal improve over BPR.

---

## Lesson 2 — Single-domain graph beats CDR on the standard benchmark

**Claim**: On standard LLO with a mixed user population, a well-tuned single-domain graph model (LightGCN) outperforms all collaborative CDR models.

**Data**: `movie_game` (same as Lesson 1: k-core ≥ 10 both domains, no further filtering).
**Split**: Standard LLO on games.

**Models to port**:
- `ml/models/lightgcn.py` → `bench_lightgcn.py` (game-only training)
- `ml/models/ncf.py` → `bench_ncf.py` (game-only training)
- `ml/models/cmf.py` → `bench_cmf.py`
- `ml/models/emcdr.py` → `bench_emcdr.py`
- `ml/models/ptupcdr.py` → `bench_ptupcdr.py`
- Also port `ml/models/_cdr_base.py` (shared CDR training loop used by EMCDR / PTUPCDR)

All CDR scripts use `--domain-pair movie_game`.

Include MF-BPR from Lesson 1 as carry-forward baseline (no re-port needed).

**Report**: Recall@10, NDCG@10, sampled HR@10, sampled NDCG@10. One table, all models + % gap vs LightGCN.
**Plot**: `--lesson 2` → bar chart, all models, LightGCN highlighted as reference bar, annotated with dataset info.

**→ Lesson 3**: The Lesson 2 population mixes users with very different amounts of game history. Lesson 3 introduces the concept of an overlap cohort and adds cohort-filtering logic to `process_data.py`.

---

## Lesson 3 — Cohort definition: restricting to real overlap users

**Claim**: Many users have signal in only one domain. The overlap cohort definition materially changes results — CDR and single-domain models must be compared on the same population.

**Data**: **This lesson introduces cohort filtering logic to `process_data.py` for the first time.** Add a `--cohort` argument (or equivalent) that supports:
- `default`: (same as Phase 0) k-core ≥ 10 both movies and games
- `loose`: every user has ≥ 5 movie ratings AND ≥ 1 game rating (much wider overlap, many more users)
- `strict`: every user has ≥ 10 movie ratings AND ≥ 5 game ratings (tighter, more balanced)

Reference the cohort filtering logic in `ml/data/process_data.py` of the old repo — it lives there as the `transfer_loose` / `transfer_strict` build paths. Adapt it; do not copy blindly.

Register the three variants in `_DOMAIN_PAIR_PATHS`:
- `movie_game` → `processed/` (existing, k-core ≥ 10)
- `movie_game_loose` → `processed_loose/` (new)
- `movie_game_strict` → `processed_strict/` (new)

Compare all three in one report.

**Split**: Standard LLO on games.
**Models**: MF-BPR, NCF, LightGCN, CMF, EMCDR, PTUPCDR (all already ported).

**Report**: Three-column comparison table (Default [k≥10] | Loose [k≥5 movie, ≥1 game] | Strict [k≥10 movie, ≥5 game]), same four metrics. Include n_users and n_interactions as a header row so the cohort size changes are visible.
**Plot**: `--lesson 3` → grouped bar chart, x-axis = cohort variant (3 groups), bars = models. Each group's annotation box shows the cohort filter condition and n_users so population differences are visually obvious.

**→ Lesson 4**: Lesson 3 controls who is in the data. Lesson 4 controls how much target-domain data those users have, stressing CDR toward the regime it was designed for.

---

## Lesson 4 — Source-rich / target-sparse cohorts + subgroup slices

**Claim**: CDR's relative advantage improves when users have rich movie history but very few games. Subgroup analysis reveals which user regimes drive the headline metric.

**Data**: Extend `process_data.py` with two new target-sparse cohort variants (build on top of the `loose` cohort definition from Lesson 3):
- **Sparse-loose**: movies ≥ 10, games ≥ 1 → `movie_game_sparse_loose`
- **Sparse-strict**: movies ≥ 10, 1 ≤ games ≤ 3 → `movie_game_sparse_strict`

These add an **additional constraint on game count** on top of the "loose" movie/game overlap definition. Reference the existing `processed_transfer_loose_filtered/` in the old repo to see how this was done previously. Register both in `_DOMAIN_PAIR_PATHS`.

**Split**: Standard LLO on games.
**Models**: MF-BPR, NCF, LightGCN, CMF, EMCDR, PTUPCDR.

**Report**:
- **Table A** — Recall@10 / NDCG@10 per cohort (Sparse-loose, Sparse-strict), one column comparing to Lesson 3 for context.
- **Table B** — Subgroup breakdown. Subgroup definitions (port from `evaluate_cross_domain()` in old `benchmark_common.py`):

| Subgroup | Definition |
|---|---|
| `super_cold_users` | 0 games in train after LLO, ≥ 10 movies |
| `one_shot_target_user` | 1 game in train, ≥ 10 movies, train game is popular |
| `one_shot_unpopular_target_user` | 1 game in train, ≥ 10 movies, train game is unpopular |
| `high_source_low_target` | ≤ 3 games total, ≥ 15 movies, popular train game |
| `high_source_unpopular_low_target` | ≤ 3 games total, ≥ 15 movies, unpopular train game |
| `movie_heavy` | > 3 games total, movies > 2× games |
| `game_heavy` | > 3 games total, games > 2× movies |
| `balanced` | > 3 games total, neither 2× skew |

Reference `ml/scripts/benchmarks/benchmark_common.py` in the old repo for the exact subgroup logic.
**Plot**: `--lesson 4` → two charts:
  - Chart A: grouped bars per cohort (Sparse-loose, Sparse-strict), bars = models, annotated with filter conditions + n_users.
  - Chart B: subgroup heatmap or grouped bar chart — x-axis = subgroup name, bars = models, with a small text note per subgroup showing its n_users. Skip subgroups with fewer than 10 users.

**→ Lesson 5**: Lessons 2–4 vary the user population. Lesson 5 varies the item catalog — filtering out weakly transferable items.

---

## Lesson 5 — Catalog sharpening: genre filter + overlap-user item filter

**Claim**: Removing genre-mismatched and overlap-user-irrelevant items reduces embedding noise and should improve CDR relative to single-domain.

**Data**: Start from Lesson 4 Sparse-loose cohort. Apply two interventions as ablations:
1. **Genre whitelist**: drop movie-only genres with no game analog (Documentary, Exercise DVDs, Musicals, Classical).
2. **Overlap-user item filter**: drop items never rated by any overlap user.

Register filtered variant as `movie_game_sparse_loose_filtered`. Reference `processed_transfer_loose_filtered/` in the old repo for the existing implementation in `process_data.py`.

**Split**: Standard LLO on games.
**Models**: LightGCN, CMF, EMCDR, PTUPCDR.

**Report**: Four-column table: Unfiltered (from Lesson 4) | Genre-only | Item-filter-only | Both. Rows: LightGCN and PTUPCDR at minimum, other models optional. If a null result appears (filter did not help), report it — do not omit.
**Plot**: `--lesson 5` → bar chart with four filter-variant groups on x-axis. Annotation box per group shows the active filters and resulting n_items (movies and games separately) so catalog shrinkage is visible.

**→ Lesson 6**: Lessons 2–5 all use LLO so even cold users have ≥ 1 game in training. Lesson 6 removes all game edges for a held-out user set — the true zero-shot regime.

---

## Lesson 6 — User-split cold-start protocol

**Claim**: When all game interactions are hidden for a subset of users during training, CDR outperforms single-domain models by 2–5× on that subset. This justifies a routing rule: no game history → CDR path.

**Data**: `movie_game_loose` (from Lesson 3). User split: 80% warm / 20% cold.
**Split**: Port `load_user_split_cold_start_split()` from `ml/scripts/benchmarks/bench_user_split_coldstart.py` in the old repo → new `bench_user_split_coldstart.py`.

**Models — single-domain (expected to fail on cold users)**:
- MF-BPR — no game edges → scores collapse to global bias
- NCF — no game edges → random-like scores
- LightGCN — no game edges → near-popularity behavior

**Models — cross-domain (expected to win on cold users)**:
- CMF — movie edges keep user factors alive in the shared factorization
- EMCDR — mapping(movie_embedding) → game space, works without any game history
- PTUPCDR — MoE hypernetwork maps movie preference → game embedding; blend weight = `1/(1+0)` = pure movie transfer

**Baselines**:
- Popularity — rank by global game popularity (surprisingly strong at cold-start: first game choice is often a well-known title)

**Also run**: LightGCN + movie→game co-occurrence rerank, gated to cold users only (`--cooc-max-target-train 0`). Port `ml/scripts/benchmarks/cooc_rerank.py` from old repo. This is a test-time-only patch that injects movie→game co-preference signal without retraining.

**Report**: Cold-user evaluation only. Full-rank Recall@10 + NDCG@10 AND sampled HR@10 + NDCG@10 (both, because full-rank understates CDR advantage here).
**Plot**: `--lesson 6` → side-by-side bar chart, cold users only. Color-code bars by type (single-domain = grey, cross-domain = blue, baseline = orange). Annotation box shows warm/cold split ratio and n_cold_users. Include a second small panel comparing full-rank vs sampled NDCG@10 for the same models, to illustrate why both protocols are needed.

**→ Lesson 7**: Lessons 1–6 are purely collaborative. Lesson 7 adds item text (SBERT) as a content bridge for cases where collaborative overlap is too thin.

---

## Lesson 7 — Content-aware CDR (SBERT)

**Claim**: When collaborative CDR is noisy or user overlap is low, SBERT item embeddings can bridge movie and game content in a shared semantic space — particularly for unpopular target items.

**Data**: `movie_game_loose` (or `movie_game_sparse_loose_filtered` from Lesson 5 if filtering helped).
**Split**: Standard LLO on games.

**Models to port**:
- `ml/models/sbert_model.py` → `bench_sbert.py` (in-domain SBERT: user vec = mean of game item embeddings)
- `ml/scripts/benchmarks/bench_sbert_cdr.py` → `bench_sbert_cdr.py` (SBERT-CDR: user vec = mean of **movie** item embeddings, rank games by cosine in shared text space)
- Compare against LightGCN and PTUPCDR (already ported).

**Report**: Overall Recall@10 / NDCG@10 + subgroup focus on `high_source_unpopular_low_target` and `one_shot_unpopular_target_user`. If SBERT-CDR does not beat collaborative CDR overall, report it and explain (text similarity ≠ behavioral preference).
**Plot**: `--lesson 7` → two-panel chart: left = overall bar chart (all four models), right = subgroup bar chart focused on the two unpopular slices. Annotation box shows dataset info and SBERT embedding dimensions.

---

## Summary: the decision rule all lessons justify

| User state | Recommended model | Justified by |
|---|---|---|
| 0 games, rich movies | EMCDR or PTUPCDR + popularity blend | Lesson 6 |
| 1–2 games, rich movies | PTUPCDR (few-shot blend) | Lessons 4, 6 |
| 3–9 games | LightGCN (game-only) | Lessons 2, 4 |
| 10+ games | LightGCN | Lesson 2 |
| Niche / unpopular games | SBERT-CDR as fallback | Lesson 7 |

---

## Implementation rules for Claude Code

### Git branching

- **Create a new branch for each lesson/phase**: `phase-0`, `lesson-1`, `lesson-2`, etc.
- **When a lesson/phase is complete** (bench scripts run, plots saved), merge the branch to `main` and create the next lesson branch from `main`.
- **Use the current branch name to know which lesson you are working on.** Do not skip ahead.

### Conversation workflow

- **One lesson per conversation.** Each new conversation should check which lesson/phase is current (via branch name or last merged branch) and implement only that one. Do not execute all lessons in a single conversation.
- **Before starting a new lesson**, verify the previous lesson is fully merged to `main` with passing bench scripts and saved plots.
- **Each lesson must end with**: (1) bench scripts run successfully, (2) result JSONs saved, (3) plot PNGs generated, (4) branch merged to `main`.

### Core workflow

- **Always read the old repo first** (`~/work/MoviesGamesRecommender`) before writing any model, loader, or eval logic. Port and adapt — don't rewrite from scratch. Fix bugs in old repo code if encountered during porting.
- **One file per model, one file per bench script.** Mirror the old repo's naming (`bench_<model>.py`, `ml/models/<model>.py`).
- **Do not start Lesson N+1 until Lesson N's bench scripts run end-to-end and the plot PNG is saved.**
- **RecBole-CDR** is vendored at `ml/models/recbole_cdr/` from the start. Do not pip install it; assume it's always available.

### Hardware

- **Use GPU (CUDA/MPS) when available**, fall back to CPU. PyTorch models should auto-detect device. Use whichever is faster.
- **Hyperparameters**: start with the old repo's hyperparameters. If a model trains slowly (>10 min for a single run), tune down epochs/embedding size to keep iteration fast. Document any changes.

### Code quality & libraries

- **Use only publicly available libraries.** For example: pandas, numpy, PyTorch, scikit-learn, matplotlib, seaborn. Refrain entirely from self-implementation of algorithms, utilities, or helpers — always prefer existing well-maintained packages.
- **Keep code logic simple and minimal.** Prioritize readability and directness over cleverness. Minimize lines of code. If a model/loader/evaluator is getting long (>200 lines), check if it can be simplified by using library functions or removing redundant logic.
- **Avoid custom classes and wrappers where standard library alternatives exist.** For example, use pandas DataFrames directly rather than custom data containers; use sklearn pipelines rather than custom preprocessing chains.

### Data and evaluation

- **All metrics are @10 only.** No @5, @20, @50 in any output or report.
- **Single evaluation entry point**: all bench scripts call `evaluate_cross_domain()` from `benchmark_common.py`. Do not inline eval logic in bench scripts.
- **Positive threshold**: `POSITIVE_THRESHOLD = 4`, defined once in `benchmark_common.py`, imported everywhere.
- **New cohort variants**: add the new `--domain-pair` key to `_DOMAIN_PAIR_PATHS` in `benchmark_common.py`. Do not hardcode data paths in bench scripts.

### Artifacts & outputs

- **Artifacts**: every bench script writes a JSON result via `save_result()` to `artifacts/<domain-pair>/results/<model>_lesson<N>.json`. The filename must include the lesson number so results from different lessons don't overwrite each other.
- **`save_result()` must always include a `"dataset_info"` block** (see Phase 0 spec). If a bench script does not have all fields, it must at minimum include `domain_pair`, `n_users`, and `cohort_filter`.
- **Plotting**: after all bench scripts for a lesson finish, always run `python plot_results.py --domain-pair <pair> --lesson <N>`. The plot is the deliverable, not just the JSON.
- **Annotation box content**: every plot must show — domain pair, cohort filter string, n_users, n_movie_interactions, n_game_interactions. These come from `dataset_info` in the JSON; do not hardcode them in the plot script.
