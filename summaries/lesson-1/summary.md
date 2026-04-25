# Lesson 1 Summary — Explicit Rating Prediction vs Implicit Ranking

**Claim**: BPR-trained implicit models outperform explicit MF on Recall@10 even at the same model capacity.

**Result**: Confirmed. BPR wins 5.2x on Recall@10.

---

## What changed vs Phase 0

| Aspect | Phase 0 | Lesson 1 |
|--------|---------|----------|
| **Models** | Infrastructure only | +MF Explicit (Surprise SVD), +MF BPR (pairwise BPR loss) |
| **Data** | Raw JSONL → parquet pipeline | Implicit conversion (rating>=4), no user k-core, randomly sampled to ~1M users |
| **Evaluation** | Framework built | First benchmark run with full-rank + sampled@99 |

**Kept constant**: Item k-core (movies >= 20, games >= 10), leave-last-out split on games, POSITIVE_THRESHOLD = 4, all metrics @10.

---

## Dataset Characteristics

### Original Amazon 2023 dataset (before sampling)

| Property | Value |
|----------|-------|
| Total users | 3,690,788 |
| Movie ratings | 6,770,775 |
| Game ratings | 1,394,811 |
| Total ratings | 8,165,586 |
| Movie items | 53,383 (k-core >= 20) |
| Game items | 14,502 (k-core >= 10) |
| Overlap users | 212,299 (5.8%) |

### Lesson 1 dataset (sampled to ~1M users)

| Property | Value |
|----------|-------|
| Domain pair | movie_game |
| Cohort filter | no user k-core filter, sampled to ~1000K users |
| Total users | 1,000,000 |
| Movie interactions | 1,804,737 (all rating >= 4) |
| Game interactions | 337,803 (all rating >= 4) |
| Game items | 11,706 |
| Overlap users | 52,281 (5.2%) |
| Split | Leave-last-out on games |

---

## Benchmark Results

| Model | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|-------|-----------|---------|---------------|-----------------|
| **MF_BPR** | **0.0130** | **0.0067** | 0.1685 | 0.0758 |
| MF_Explicit | 0.0025 | 0.0010 | 0.1860 | 0.0720 |

**BPR advantage**: 5.2x on Recall@10, 6.7x on NDCG@10.

---

## Key Takeaways

1. **Lesson claim confirmed**: BPR massively outperforms explicit SVD on full-rank ranking metrics.

2. **Implicit conversion makes explicit even worse**: With all ratings now 4-5, SVD has almost no variance to learn from. BPR's pairwise objective is the right fit for implicit data.

3. **Absolute numbers are low**: BPR achieves 1.3% Recall@10 with no cross-domain signal. Motivates adding more powerful models in Lesson 2.

## Benchmark Plots

- Main comparison: `artifacts/plots/lesson_1_20260406_150249.png`
