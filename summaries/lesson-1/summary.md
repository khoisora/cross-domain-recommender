# Lesson 1 Summary — Explicit Rating Prediction vs Implicit Ranking

**Claim**: Models trained to minimize RMSE on star ratings optimize a different objective than top-10 ranking. BPR-trained implicit models outperform explicit MF on Recall@10 even at the same model capacity.

---

## What changed vs Phase 0

| Aspect | Phase 0 | Lesson 1 |
|--------|---------|----------|
| **Models** | Infrastructure only | +MF Explicit (Surprise SVD), +MF BPR (pairwise BPR loss) |
| **Data** | Raw JSONL → parquet pipeline | No user k-core, sampled to ~100K users |
| **Evaluation** | Framework built | First benchmark run with full-rank + sampled@99 |

**Kept constant**: Item k-core (movies >= 20, games >= 10), leave-last-out split on games, POSITIVE_THRESHOLD = 4, all metrics @10.

---

## File Changes

| File | Change | Description |
|------|--------|-------------|
| `ml/models/matrix_factorization.py` | Added | MF Explicit using Surprise SVD (score = mu + b_u + b_i + q_i^T * p_u) |
| `ml/models/matrix_factorization_bpr.py` | Added | MF with BPR pairwise loss, vectorized mini-batch SGD |
| `ml/models/id_utils.py` | Modified | Added `map_user_item_columns()` for DataFrame → index mapping |
| `ml/scripts/benchmarks/bench_mf_explicit.py` | Added | Benchmark script for MF Explicit |
| `ml/scripts/benchmarks/bench_mf_bpr.py` | Added | Benchmark script for MF BPR |
| `ml/data/process_data.py` | Modified | Added `--min-user-interactions` and `--sample-users` CLI args, conditional k-core, proportional user sampling |
| `ml/scripts/benchmarks/benchmark_common.py` | Modified | Dynamic `_cohort_filter_string()` reads from `dataset_metadata.json`; removed subgroup logic (deferred to L4) |

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Domain pair | movie_game |
| Cohort filter | no user k-core filter, sampled to ~100K users |
| Total users | 99,999 |
| Movie interactions | 180,070 |
| Game interactions | 37,683 |
| Game items | 8,471 |
| Overlap users | 5,752 (5.8%) |
| Split | Leave-last-out on games |
| Game train / val / test | 11,340 / 2,447 / 23,896 |
| Eval users (sampled) | 2,000 of 17,904 with relevant test items |

---

## Benchmark Results

### Overall Metrics

| Model | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 | Train Time |
|-------|-----------|---------|---------------|-----------------|------------|
| MF_Explicit (SVD, emb=128) | **0.0080** | **0.0033** | **0.2715** | **0.1179** | 0.4s |
| MF_BPR (emb=64, 60 epochs) | 0.0015 | 0.0010 | 0.1050 | 0.0404 | 1.3s |

**MF Explicit advantage**: 5.3x on Recall@10, 3.3x on NDCG@10, 2.6x on sampled HR@10.

---

## Benchmark Plots

- Main comparison: `artifacts/plots/lesson_1_20260406_121508.png`

---

## Key Takeaways

1. **Lesson claim reversed**: On this extremely sparse dataset (100K users, no k-core), MF Explicit beats MF BPR on all metrics. This is the opposite of the original claim. The reason: BPR only has 8,944 positive pairs to learn from (game train ratings >= 4), which is insufficient for pairwise learning across 100K users and 8K items.

2. **Why BPR fails here**: BPR loss barely moves (0.693 → 0.692 over 60 epochs), meaning the model cannot learn meaningful user-item rankings from so few positive pairs. MF Explicit uses all 11K explicit ratings (including lower ratings), giving it more signal to fit user/item biases.

3. **Extreme sparsity**: With no user k-core filter, most of the 100K users have very few game interactions. Only 5.8% of users appear in both domains. The game train set has just 11,340 interactions across 100K users — an average of 0.11 game train interactions per user.

4. **Absolute numbers are very low**: Even MF Explicit only achieves 0.8% Recall@10 in full-rank. This motivates adding more powerful models (graph, neural, CDR) in Lesson 2 on the same data to see if architecture can compensate for sparsity.

5. **Sampled metrics paint a different picture**: MF Explicit achieves 27% HR@10 in the sampled protocol (1 pos + 99 neg), showing the model does learn something — it just drowns in the full-rank setting with 8,471 candidate items.

---

## Match with Original Plan

The lesson's original claim — that BPR outperforms explicit MF — is **not confirmed** on this dataset. The reversal is driven by extreme sparsity: without k-core filtering, BPR's pairwise loss has too few positive pairs to learn from. MF Explicit's advantage of using all explicit ratings (not just positives) gives it an edge in this regime. This sets up an interesting question for Lesson 2: can more powerful architectures (LightGCN, NCF) and cross-domain transfer (CMF, EMCDR, PTUPCDR) overcome this sparsity?
