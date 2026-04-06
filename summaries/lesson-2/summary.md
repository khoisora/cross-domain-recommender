# Lesson 2 — Single-domain graph beats CDR on mixed population

**Claim**: On standard LLO with a mixed user population, a well-tuned single-domain graph model (LightGCN) outperforms all collaborative CDR models.

**Result**: Confirmed. LightGCN leads by 1.7x over MF-BPR and 1.8x over the best CDR model (EMCDR) on full-rank Recall@10.

---

## What changed vs Lesson 1

| Aspect | Lesson 1 | Lesson 2 |
|--------|----------|----------|
| **Models** | MF Explicit, MF BPR (2 models) | +LightGCN, +NCF, +CMF, +EMCDR, +PTUPCDR (5 new models) |
| **Data** | Same | Same (no user k-core, ~1M randomly sampled users, implicit rating >= 4) |
| **Evaluation** | Same | Same |

**Kept constant**: Dataset (1M users, no k-core, 5.2% overlap, implicit rating >= 4), item k-core, LLO split, POSITIVE_THRESHOLD=4, all metrics @10.

---

## What changed vs Lesson 1

| Variable | Lesson 1 | Lesson 2 | Why |
|---|---|---|---|
| **User filter** | k-core >= 10 | No k-core, sampled 100K | Test natural low-overlap population |
| **Overlap %** | 32% (not controlled) | 5.8% (natural ratio) | Demonstrate CDR failure at low overlap |
| **Models** | MF-Explicit, MF-BPR | + LightGCN, NCF, CMF, EMCDR, PTUPCDR | Full model portfolio for comparison |

**Kept constant**: Item k-core (movies>=20, games>=10), positive threshold (>=4), LLO split on games, metrics @10.

## File changes

| File | Change |
|---|---|
| `ml/models/lightgcn.py` | Ported from old repo. Val-based early stopping. |
| `ml/models/ncf.py` | Ported NCF (NeuMF) model from old repo. |
| `ml/models/cmf.py` | Ported CMF model from old repo. |
| `ml/models/emcdr.py` | Ported EMCDR model from old repo. |
| `ml/models/ptupcdr.py` | Ported PTUPCDR model from old repo. |
| `ml/models/_cdr_base.py` | Ported shared CDR training loop (EMCDR/PTUPCDR). |
| `ml/scripts/benchmarks/bench_lightgcn.py` | New bench script. layers=3, reg=0.001, val early stopping. |
| `ml/scripts/benchmarks/bench_ncf.py` | New bench script for NCF. |
| `ml/scripts/benchmarks/bench_cmf.py` | New bench script for CMF. |
| `ml/scripts/benchmarks/bench_emcdr.py` | New bench script for EMCDR. |
| `ml/scripts/benchmarks/bench_ptupcdr.py` | New bench script for PTUPCDR. |

## Dataset characteristics

| Property | Value |
|---|---|
| Domain pair | movie_game |
| Cohort filter | no user k-core filter, sampled to ~1000K users |
| Implicit conversion | rating >= 4 (POSITIVE_THRESHOLD) |
| n_users | 1,000,000 |
| n_movie_items | 45,933 |
| n_game_items | 11,706 |
| n_movie_interactions | 1,804,737 |
| n_game_interactions | 337,803 |
| Overlap users | 52,281 (5.2%) |
| Split | Leave-last-out on games |

## Benchmark results

| Model | Family | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0290** | **0.0157** | 0.3050 | 0.1425 |
| MF-BPR | MF (single-domain) | 0.0170 | 0.0093 | 0.1710 | 0.0783 |
| EMCDR | Mapping (CDR) | 0.0160 | 0.0082 | 0.8045 | 0.4805 |
| NCF | Neural (single-domain) | 0.0120 | 0.0069 | 0.8880 | 0.5410 |
| PTUPCDR | Personalized mapping (CDR) | 0.0085 | 0.0054 | 0.8850 | 0.5397 |
| CMF | Joint MF (CDR) | 0.0050 | 0.0022 | 0.6945 | 0.4273 |

### % gap vs LightGCN (full-rank Recall@10)

| Model | Gap |
|---|---|
| MF-BPR | -41.4% |
| EMCDR | -44.8% |
| NCF | -58.6% |
| PTUPCDR | -70.7% |
| CMF | -82.8% |

## Benchmark plots

- Main comparison: `artifacts/plots/lesson_2_20260406_141818.png`

## Key takeaways

1. **LightGCN dominates** on full-rank metrics (Recall@10=0.0290), 1.7x over MF-BPR and 1.8x over the best CDR model. The graph structure captures collaborative signal effectively even on sparse data.

2. **CDR models underperform single-domain**. EMCDR (0.0160) is the best CDR but still below MF-BPR (0.0170). With only 5.2% overlap, cross-domain mappings have too few shared users to learn meaningful transfer.

3. **Sampled metrics diverge from full-rank**. NCF, PTUPCDR, and EMCDR all show very high sampled HR@10 (0.80-0.89) but low full-rank Recall@10. The 1-vs-99 protocol is too easy — models learn user/item biases that beat random negatives but fail to rank the full 11.7K item catalog.

4. **MF-BPR holds up well**. Despite being the simplest model, it beats all CDR models on full-rank. BPR's pairwise objective produces discriminative scores even on sparse data.

5. **Lesson 3 setup**: Filtering to 100% overlap users should improve CDR by giving transfer mappings enough shared users. The key question: can CDR catch LightGCN when overlap is guaranteed?
