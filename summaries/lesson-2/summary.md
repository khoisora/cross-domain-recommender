# Lesson 2 — Single-domain graph beats CDR on the standard benchmark

**Claim**: On standard LLO with a mixed user population, a well-tuned single-domain graph model (LightGCN) outperforms all collaborative CDR models.

**Result**: Confirmed. LightGCN leads on full-rank Recall@10 and NDCG@10.

---

## File changes

| File | Change |
|---|---|
| `ml/models/lightgcn.py` | Ported from old repo. Fixed early stopping: val-only patience when `val_metric_fn` provided (non-val epochs no longer increment patience via loss fallback). |
| `ml/models/ncf.py` | Ported NCF (NeuMF) model from old repo. |
| `ml/models/cmf.py` | Ported CMF model from old repo. |
| `ml/models/emcdr.py` | Ported EMCDR model from old repo. |
| `ml/models/ptupcdr.py` | Ported PTUPCDR model from old repo. |
| `ml/models/_cdr_base.py` | Ported shared CDR training loop (EMCDR/PTUPCDR). |
| `ml/scripts/benchmarks/bench_lightgcn.py` | Fixed config to match old repo: layers 2→3, reg 0.01→0.001, added val-based early stopping (patience=5, val_every=3). |
| `ml/scripts/benchmarks/bench_ncf.py` | New bench script for NCF. |
| `ml/scripts/benchmarks/bench_cmf.py` | New bench script for CMF. |
| `ml/scripts/benchmarks/bench_emcdr.py` | New bench script for EMCDR. |
| `ml/scripts/benchmarks/bench_ptupcdr.py` | New bench script for PTUPCDR. |
| `ml/scripts/benchmarks/benchmark_common.py` | `single_domain_item_space=True` now restricts user space to target-domain users (compact embeddings for single-domain models). |

## Dataset characteristics

| Property | Value |
|---|---|
| Domain pair | movie_game |
| Cohort filter | users ≥ 10 total interactions, no overlap filter |
| n_users | 95,490 |
| n_movie_items | 52,772 |
| n_game_items | 13,469 |
| n_movie_interactions | 2,041,794 |
| n_game_interactions | 224,438 |
| Overlap users | 30,606 |
| Split | Leave-last-out on games |

## Benchmark results

| Model | Family | Recall@10 | NDCG@10 | HR@10 | sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0540** | **0.0310** | 0.4600 | 0.2237 |
| NCF | Neural (single-domain) | 0.0435 | 0.0220 | **0.6755** | **0.3723** |
| EMCDR | Mapping (CDR) | 0.0350 | 0.0209 | 0.4895 | 0.2212 |
| PTUPCDR | Personalized mapping (CDR) | 0.0340 | 0.0201 | 0.5420 | 0.2592 |
| CMF | Joint MF (CDR) | 0.0180 | 0.0095 | 0.2155 | 0.0972 |
| MF-BPR | MF (single-domain) | 0.0100 | 0.0056 | 0.1405 | 0.0577 |

### % gap vs LightGCN (full-rank Recall@10)

| Model | Gap |
|---|---|
| NCF | -19.4% |
| EMCDR | -35.2% |
| PTUPCDR | -37.0% |
| CMF | -66.7% |
| MF-BPR | -81.5% |

## LightGCN hyperparameters (final)

| Parameter | Value | Note |
|---|---|---|
| embedding_dim | 96 | |
| num_layers | 3 | Was 2 (bug), fixed to match old repo |
| epochs | 50 (early stop) | Best val at epoch 48, stopped at 63 |
| lr | 0.001 | |
| reg_lambda | 0.001 | Was 0.01 (bug), fixed to match old repo |
| dropout | 0.1 | |
| batch_size | 4096 | |
| neg_sampling | popularity (α=0.75) | |
| val_every | 3 | |
| early_stopping_patience | 5 | |

## LightGCN debugging notes

The original bench config had two bugs that severely degraded performance:
1. **`reg_lambda=0.01` instead of `0.001`** — 10× too much L2 regularization crushed embeddings
2. **`num_layers=2` instead of `3`** — one fewer GCN layer reduced neighborhood aggregation, which is the core advantage of LightGCN over plain MF

Adding validation-based early stopping (NDCG@10 on val set, patience=5, eval every 3 epochs) ensured the model saved the best checkpoint rather than running a fixed number of epochs.

## Key takeaways

1. **LightGCN leads on full-rank metrics** (Recall@10=0.0540, NDCG@10=0.0310), confirming the claim that a well-tuned single-domain graph model outperforms CDR on the standard mixed-population benchmark.

2. **NCF dominates sampled metrics** (HR@10=0.6755, sampled NDCG@10=0.3723). This discrepancy with full-rank is a known limitation of the 1+99 sampled protocol — NCF discriminates well against random negatives but doesn't rank as well against the full catalog.

3. **CDR models underperform** on this mixed population. EMCDR and PTUPCDR are competitive but trail LightGCN by 35-37% on Recall@10. CMF performs poorly. This makes sense: the mixed population includes many game-heavy users who don't benefit from movie→game transfer.

4. **MF-BPR is the weakest**, confirming Lesson 1's finding that plain MF without graph structure or neural architecture is insufficient.

5. **The benchmark population matters**. With 95K users and only 30K overlap users, many users are single-domain. CDR models should improve on overlap-user cohorts (Lesson 3) and especially on cold-start users (Lesson 6).
