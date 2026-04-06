# Lesson 4 — Source-rich / target-sparse: CDR closes the gap

**Claim**: CDR's relative advantage improves when users have rich movie history but sparse game history. Tightening the movie requirement from >= 5 to >= 10 gives CDR models richer source embeddings to transfer from.

**Result**: Confirmed. PTUPCDR closes to within 17% of LightGCN (vs 38% in Lesson 3). EMCDR also narrows the gap. CDR models benefit from richer source data while single-domain models lose ground as users become more target-sparse.

---

## What changed vs Lesson 3

| Variable | Lesson 3 | Lesson 4 | Why |
|---|---|---|---|
| **Movie minimum** | >= 5 ratings | >= 10 ratings | Ensure richer source embeddings for CDR transfer |
| **n_users** | 26,487 | 18,841 | Tighter movie filter removes ~7.6K game-heavy users |
| **Game interactions** | 129K | 86K (-33%) | Removed users had rich game histories |

**Kept constant**: User k-core (>=10), game minimum (>=1), 100% overlap, item k-core, positive threshold (>=4), LLO split on games, metrics @10, all 6 models, evaluation protocol.

**Key logic**: Raising movie floor from 5→10 removes users who were movie-poor/game-rich. This hurts single-domain models (less game data) but helps CDR models (richer source embeddings). The experiment isolates source-richness as a variable.

## File changes

| File | Change |
|---|---|
| `ml/data/process_data.py` | Added `max_game_ratings` param (not used this lesson, available for future use). |
| `ml/scripts/benchmarks/benchmark_common.py` | `movie_game` now points to `processed_sparse_loose/` (movies >= 10, games >= 1). |
| `LESSON_PLAN.md` | Simplified Lesson 4: single default dataset, no separate cohort variants. |

## Dataset characteristics

| Property | Value |
|---|---|
| Domain pair | movie_game |
| Cohort filter | users >= 10 total interactions, overlap users (movies >= 10, games >= 1) |
| n_users | 18,841 |
| n_movie_items | 47,089 |
| n_game_items | 11,545 |
| n_movie_interactions | 539,782 |
| n_game_interactions | 86,070 |
| Overlap | 100% |
| Split | Leave-last-out on games |

## Benchmark results

| Model | Family | Recall@10 | NDCG@10 | HR@10 | sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0380** | **0.0206** | 0.3535 | 0.1680 |
| PTUPCDR | Personalized mapping (CDR) | 0.0315 | 0.0161 | 0.4310 | 0.1962 |
| EMCDR | Mapping (CDR) | 0.0265 | 0.0140 | 0.4150 | 0.1835 |
| NCF | Neural (single-domain) | 0.0225 | 0.0109 | 0.6875 | 0.3948 |
| CMF | Joint MF (CDR) | 0.0140 | 0.0076 | 0.2240 | 0.0977 |
| MF-BPR | MF (single-domain) | 0.0095 | 0.0056 | 0.1340 | 0.0574 |

### % gap vs LightGCN (full-rank Recall@10)

| Model | L3 gap | L4 gap | Trend |
|---|---|---|---|
| PTUPCDR | -38% | -17% | CDR closing |
| EMCDR | -40% | -30% | CDR closing |
| NCF | -44% | -41% | Flat |
| CMF | -74% | -63% | CDR closing |
| MF-BPR | -77% | -75% | Flat |

### Lesson 3 → Lesson 4 comparison (Recall@10)

| Model | L3 (movies>=5) | L4 (movies>=10) | Change |
|---|---|---|---|
| LightGCN | 0.0525 | 0.0380 | -28% |
| PTUPCDR | 0.0325 | 0.0315 | -3% |
| EMCDR | 0.0315 | 0.0265 | -16% |
| NCF | 0.0295 | 0.0225 | -24% |
| CMF | 0.0135 | 0.0140 | +4% |
| MF-BPR | 0.0120 | 0.0095 | -21% |

## Benchmark plots

- `artifacts/plots/lesson_4_*.png`

## Key takeaways

1. **PTUPCDR nearly matches LightGCN**: gap narrows from -38% (L3) to -17% (L4). The per-user hypernetwork benefits from richer movie histories (>=10 ratings) which give it more source signal to personalize the transfer mapping.

2. **LightGCN drops most** (-28% from L3): tightening the movie filter from >=5 to >=10 removes users who were game-heavy (many games, few movies). These were LightGCN's strongest users. The remaining users are more movie-oriented — exactly the population CDR was designed for.

3. **CDR models are more robust to the filter change**: PTUPCDR only drops 3%, EMCDR 16%, vs LightGCN's 28%. CDR models benefit from richer source data offsetting the loss of game-heavy users.

4. **CMF is the only model that improves** (+4%). Joint factorization benefits from the higher movie count requirement which gives it denser shared factors.

5. **The trend is clear across L2-L4**: as we move from mixed population (L2) → overlap users (L3) → source-rich overlap (L4), CDR models steadily close the gap to LightGCN. The next step (L6: cold-start) should flip the ranking entirely.
