# Lesson 4 — Source-rich / target-sparse cohorts: CDR overtakes LightGCN

**Claim**: CDR's relative advantage improves when users have rich movie history but very few games. Subgroup analysis reveals which user regimes drive the headline metric.

**Result**: Confirmed. On the strictest cohort (movies >= 10, 1-3 games), EMCDR leads all models at Recall@10=0.0130 — 2.6x higher than LightGCN (0.0050). This is the first lesson where a CDR model beats the best single-domain model on full-rank metrics.

---

## File changes

| File | Change |
|---|---|
| `ml/data/process_data.py` | Added `max_game_ratings` param for upper-bounding game interactions (sparse-strict cohort). |
| `ml/scripts/benchmarks/benchmark_common.py` | Registered `movie_game_sparse_loose` and `movie_game_sparse_strict` in `_DOMAIN_PAIR_PATHS`. |

## Dataset characteristics

| Property | Sparse-loose | Sparse-strict |
|---|---|---|
| Cohort filter | k-core >= 10, movies >= 10, games >= 1 | k-core >= 10, movies >= 10, 1 <= games <= 3 |
| n_users | 18,841 | 12,891 |
| n_movie_items | 47,089 | 44,131 |
| n_game_items | 11,545 | 6,464 |
| n_movie_interactions | 539,782 | 337,531 |
| n_game_interactions | 86,070 | 19,679 |
| Overlap | 100% | 100% |
| Split | Leave-last-out on games | Leave-last-out on games |

## Benchmark results

### Sparse-loose (movies >= 10, games >= 1)

| Model | Family | Recall@10 | NDCG@10 | HR@10 | sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0440** | **0.0247** | 0.3585 | 0.1699 |
| PTUPCDR | Personalized mapping (CDR) | 0.0295 | 0.0159 | 0.4215 | 0.1913 |
| NCF | Neural (single-domain) | 0.0275 | 0.0137 | 0.6970 | 0.3967 |
| EMCDR | Mapping (CDR) | 0.0240 | 0.0127 | 0.4140 | 0.1849 |
| CMF | Joint MF (CDR) | 0.0175 | 0.0097 | 0.2050 | 0.0895 |
| MF-BPR | MF (single-domain) | 0.0095 | 0.0056 | 0.1340 | 0.0574 |

### Sparse-strict (movies >= 10, 1 <= games <= 3)

| Model | Family | Recall@10 | NDCG@10 | HR@10 | sampled NDCG@10 |
|---|---|---|---|---|---|
| **EMCDR** | Mapping (CDR) | **0.0130** | **0.0057** | 0.2040 | 0.0846 |
| NCF | Neural (single-domain) | 0.0085 | 0.0037 | **0.7680** | **0.4682** |
| PTUPCDR | Personalized mapping (CDR) | 0.0055 | 0.0030 | 0.1735 | 0.0724 |
| LightGCN | Graph (single-domain) | 0.0050 | 0.0025 | 0.1780 | 0.0688 |
| MF-BPR | MF (single-domain) | 0.0035 | 0.0018 | 0.1010 | 0.0402 |
| CMF | Joint MF (CDR) | 0.0025 | 0.0015 | 0.1645 | 0.0632 |

### Cross-cohort comparison (Recall@10)

| Model | L3 (movies>=5, games>=1) | Sparse-loose (movies>=10, games>=1) | Sparse-strict (movies>=10, games 1-3) |
|---|---|---|---|
| LightGCN | 0.0525 | 0.0440 (-16%) | 0.0050 (-90%) |
| PTUPCDR | 0.0325 | 0.0295 (-9%) | 0.0055 (-83%) |
| EMCDR | 0.0315 | 0.0240 (-24%) | 0.0130 (-59%) |
| NCF | 0.0295 | 0.0275 (-7%) | 0.0085 (-71%) |
| CMF | 0.0135 | 0.0175 (+30%) | 0.0025 (-81%) |
| MF-BPR | 0.0120 | 0.0095 (-21%) | 0.0035 (-71%) |

## Key takeaways

1. **EMCDR wins on sparse-strict**: At 0.0130 Recall@10, EMCDR is 2.6x better than LightGCN (0.0050). With only 1-3 game interactions per user, graph structure has almost nothing to work with. The global MLP mapping from movie embeddings to game space is the winning strategy when target data is extremely sparse.

2. **LightGCN degrades fastest**: From L3 to sparse-strict, LightGCN drops 90% (0.053 → 0.005). Its advantage depends entirely on dense game interaction graphs. When users have very few games, the GCN layers propagate almost no signal.

3. **EMCDR degrades least**: Only 59% drop from L3 to sparse-strict vs 90% for LightGCN. The movie→game mapping transfers movie preferences regardless of how many game interactions exist. This is exactly the CDR value proposition.

4. **PTUPCDR underperforms EMCDR on sparse-strict** (0.006 vs 0.013). The personalized MoE hypernetwork needs some target-domain data to calibrate per-user mappings. With 1-3 games, the few-shot blend weight `1/(1+k)` doesn't help enough. EMCDR's simpler global mapping is more robust at extreme sparsity.

5. **CMF shows an anomaly**: It improves from L3 to sparse-loose (+30%) but collapses on sparse-strict. Joint factorization benefits from the higher movie-count requirement (movies >= 10 vs 5) but fails when game data is capped.

6. **NCF's sampled metric dominance continues**: HR@10=0.768 on sparse-strict while Recall@10 is only 0.009. The 1+99 sampled protocol remains misleading — NCF discriminates against random negatives but can't rank against the full catalog.

## Benchmark plots

- Sparse-loose: `artifacts_sparse_loose/plots/lesson_4_*.png`
- Sparse-strict: `artifacts_sparse_strict/plots/lesson_4_*.png`
