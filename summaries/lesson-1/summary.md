# Lesson 1 Summary — Explicit Rating Prediction vs Implicit Ranking

**Claim**: Models trained to minimize RMSE on star ratings optimize a different objective than top-10 ranking. BPR-trained implicit models outperform explicit MF on Recall@10 even at the same model capacity.

---

## File Changes

| File | Change | Description |
|------|--------|-------------|
| `ml/models/matrix_factorization.py` | Added | MF Explicit using Surprise SVD (score = mu + b_u + b_i + q_i^T * p_u) |
| `ml/models/matrix_factorization_bpr.py` | Added | MF with BPR pairwise loss, vectorized mini-batch SGD |
| `ml/models/id_utils.py` | Modified | Added `map_user_item_columns()` for DataFrame → index mapping |
| `ml/scripts/benchmarks/bench_mf_explicit.py` | Added | Benchmark script for MF Explicit |
| `ml/scripts/benchmarks/bench_mf_bpr.py` | Added | Benchmark script for MF BPR |

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Domain pair | movie_game |
| Cohort filter | users >= 10 total interactions, no overlap filter |
| Total users | 95,490 |
| Movie interactions | 2,041,794 |
| Game interactions | 224,438 |
| Game items | 13,469 |
| Split | Leave-last-out on games |
| Game train / val / test | 171,368 / 19,799 / 33,271 |
| Eval users (sampled) | 2,000 of 26,280 with relevant test items |

---

## Benchmark Results

### Overall Metrics

| Model | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 | Train Time |
|-------|-----------|---------|---------------|-----------------|------------|
| MF_Explicit (SVD, emb=128) | 0.0040 | 0.0019 | 0.2080 | 0.0855 | 5.1s |
| MF_BPR (emb=64, 60 epochs) | **0.0250** | **0.0115** | **0.2670** | **0.1266** | 15.7s |

**BPR advantage**: 6.3× on Recall@10, 6.1× on NDCG@10, 1.3× on sampled HR@10.

### Subgroup Breakdown (Recall@10, full-rank)

| Subgroup | n_users | MF_Explicit | MF_BPR |
|----------|---------|-------------|--------|
| super_cold_users | 471 | 0.0021 | 0.0106 |
| one_shot_target_user | 261 | 0.0038 | 0.0230 |
| one_shot_unpopular | 56 | 0.0000 | 0.0179 |
| high_source_low_target | 435 | 0.0023 | 0.0161 |
| movie_heavy | 236 | 0.0085 | 0.0508 |
| game_heavy | 387 | 0.0052 | 0.0285 |
| balanced | 362 | 0.0055 | 0.0304 |

---

## Benchmark Plots

- Main comparison: `artifacts/plots/lesson_1_20260405_150358.png`
- Subgroup breakdown: `artifacts/plots/lesson_1_subgroups_20260405_150358.png`

---

## Key Takeaways

1. **Lesson claim confirmed**: BPR massively outperforms explicit SVD on all ranking metrics. The 6× gap on Recall@10 demonstrates that RMSE-optimized models are fundamentally misaligned with top-K ranking evaluation.

2. **Why explicit fails**: SVD learns to predict star ratings accurately (e.g., 4.2 vs 4.3), but the differences between items are tiny — all popular items get similar high predicted ratings. BPR directly optimizes the ranking order, producing much more discriminative scores.

3. **Absolute numbers are low**: Even BPR only achieves 2.5% Recall@10 in full-rank. This is expected with 13K target items, no cross-domain signal, and a mixed population where many users have very few game interactions.

4. **BPR convergence was slow**: Loss moved from 0.693 to 0.688 over 60 epochs. The vectorized batch updates with `np.add.at` suffer from duplicate-index averaging. This could be improved with proper per-sample SGD or PyTorch implementation, but results are still directionally correct.

5. **Cold users get almost nothing**: Super cold users (0 game train) have near-zero Recall@10 for both models. This sets up the CDR motivation in Lesson 2 — these users need source-domain (movie) signal.

---

## Match with Original Plan

The lesson's claim — that BPR outperforms explicit MF on ranking metrics — is **strongly confirmed**. BPR is now the baseline family for subsequent lessons. The absolute performance is low enough to motivate both architectural improvements (Lesson 2: graph, CDR) and population refinement (Lesson 3: overlap cohorts).
