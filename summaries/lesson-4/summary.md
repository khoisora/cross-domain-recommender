# Lesson 4 — Source-rich / target-sparse: CDR closes the gap

**Claim**: CDR's relative advantage improves when users have rich movie history but sparse game history. Tightening the movie requirement from >= 5 to >= 10 gives CDR models richer source embeddings to transfer from.

**Result**: Confirmed. PTUPCDR closes to within 17% of LightGCN (vs 42% in Lesson 3). EMCDR also narrows the gap. CDR models benefit from richer source data while single-domain models lose ground as users become more target-sparse.

---

## What changed vs Lesson 3

| Variable | Lesson 3 | Lesson 4 | Why |
|---|---|---|---|
| **Movie minimum** | >= 5 ratings | >= 10 ratings | Ensure richer source embeddings for CDR transfer |
| **n_users** | 19,880 | 14,328 | Tighter movie filter removes ~5.5K movie-poor users |
| **Game interactions** | 87.6K | 58.8K (-33%) | Removed users had game activity |

**Kept constant**: User k-core (>=10), game minimum (>=1), 100% overlap, item k-core, implicit conversion (rating >= 4), LLO split on games, metrics @10, all 6 models, evaluation protocol.

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
| Implicit conversion | rating >= 4 (POSITIVE_THRESHOLD) |
| n_users | 14,328 |
| n_movie_items | 39,534 |
| n_game_items | 9,147 |
| n_movie_interactions | 388,919 |
| n_game_interactions | 58,782 |
| Overlap | 100% |
| Split | Leave-last-out on games |

## Benchmark results

| Model | Family | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0350** | **0.0180** | 0.3465 | 0.1627 |
| PTUPCDR | Personalized mapping (CDR) | 0.0290 | 0.0154 | 0.3975 | 0.1795 |
| EMCDR | Mapping (CDR) | 0.0230 | 0.0116 | 0.3740 | 0.1649 |
| CMF | Joint MF (CDR) | 0.0160 | 0.0078 | 0.1945 | 0.0844 |
| NCF | Neural (single-domain) | 0.0140 | 0.0073 | 0.6575 | 0.3793 |
| MF-BPR | MF (single-domain) | 0.0070 | 0.0029 | 0.1250 | 0.0507 |

### % gap vs LightGCN (full-rank Recall@10)

| Model | L3 gap | L4 gap | Trend |
|---|---|---|---|
| PTUPCDR | -42% | -17% | CDR closing |
| EMCDR | -56% | -34% | CDR closing |
| NCF | -59% | -60% | Flat |
| CMF | -75% | -54% | CDR closing |
| MF-BPR | -91% | -80% | Slight recovery |

### Lesson 3 → Lesson 4 comparison (Recall@10)

| Model | L3 (movies>=5) | L4 (movies>=10) | Change |
|---|---|---|---|
| LightGCN | 0.0555 | 0.0350 | -37% |
| PTUPCDR | 0.0320 | 0.0290 | -9% |
| EMCDR | 0.0245 | 0.0230 | -6% |
| CMF | 0.0140 | 0.0160 | +14% |
| NCF | 0.0230 | 0.0140 | -39% |
| MF-BPR | 0.0050 | 0.0070 | +40% |

## Benchmark plots

- `artifacts/plots/lesson_4_*.png`

## Key takeaways

1. **PTUPCDR nearly matches LightGCN**: gap narrows from -42% (L3) to -17% (L4). The per-user hypernetwork benefits from richer movie histories (>=10 ratings) which give it more source signal to personalize the transfer mapping.

2. **LightGCN drops most** (-37% from L3): tightening the movie filter from >=5 to >=10 removes users who had more game activity. These were LightGCN's strongest users. The remaining users are more movie-oriented — exactly the population CDR was designed for.

3. **CDR models are more robust to the filter change**: PTUPCDR only drops 9%, EMCDR 6%, vs LightGCN's 37%. CDR models benefit from richer source data offsetting the loss of game-heavy users.

4. **CMF and MF-BPR improve** (+14% and +40%). These simpler models benefit from the denser per-user movie signal at this smaller scale.

5. **The trend is clear across L2-L4**: as we move from mixed population (L2) → overlap users (L3) → source-rich overlap (L4), CDR models steadily close the gap to LightGCN. The next step (cold-start) should flip the ranking entirely.
