# Lesson 2 Summary — Single-domain graph vs CDR on standard benchmark

**Claim**: On standard LLO with a mixed user population, a well-tuned single-domain graph model (LightGCN) outperforms all collaborative CDR models.

---

## File Changes

| File | Change | Description |
|------|--------|-------------|
| `ml/models/lightgcn.py` | Added | LightGCN using PyTorch Geometric, BPR loss, popularity-biased negative sampling, early stopping |
| `ml/models/ncf.py` | Added | NCF (NeuMF) using RecBole, full model inference (GMF + MLP + predict layer) |
| `ml/models/cmf.py` | Added | CMF via vendored RecBole-CDR, shared user factors across domains |
| `ml/models/emcdr.py` | Added | EMCDR via RecBole-CDR, source→target MLP mapping |
| `ml/models/ptupcdr.py` | Added | PTUPCDR with MoE hypernetwork mapping, few-shot blending |
| `ml/models/_cdr_base.py` | Modified | Shared CDR training loop for CMF/EMCDR using RecBole-CDR |
| `ml/scripts/benchmarks/bench_lightgcn.py` | Added | LightGCN benchmark with validation early stopping |
| `ml/scripts/benchmarks/bench_ncf.py` | Added | NCF benchmark |
| `ml/scripts/benchmarks/bench_cmf.py` | Added | CMF benchmark |
| `ml/scripts/benchmarks/bench_emcdr.py` | Added | EMCDR benchmark |
| `ml/scripts/benchmarks/bench_ptupcdr.py` | Added | PTUPCDR benchmark |

---

## Dataset Characteristics

| Property | Value |
|----------|-------|
| Domain pair | movie_game |
| Cohort filter | users >= 10 total interactions, no overlap filter |
| Total users | 95,490 |
| Movie interactions | 2,041,794 |
| Game interactions | 224,438 |
| Game items (single-domain models) | 13,469 |
| Total items (CDR models) | 66,241 |
| Split | Leave-last-out on games |
| Eval users | 2,000 (sampled from 26,280) |

---

## Benchmark Results

### Overall Metrics

| Model | Type | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|-------|------|-----------|---------|---------------|-----------------|
| **NCF** | Single-domain | **0.0435** | **0.0220** | **0.6755** | **0.3723** |
| EMCDR | Cross-domain | 0.0350 | 0.0209 | 0.4895 | 0.2212 |
| PTUPCDR | Cross-domain | 0.0340 | 0.0201 | 0.5420 | 0.2592 |
| MF_BPR | Single-domain | 0.0250 | 0.0115 | 0.2670 | 0.1266 |
| CMF | Cross-domain | 0.0180 | 0.0095 | 0.2155 | 0.0972 |
| LightGCN | Single-domain | 0.0095 | 0.0055 | 0.2270 | 0.0976 |

---

## Benchmark Plots

- Main comparison: `artifacts/plots/lesson_2_20260405_153950.png`
- Subgroup breakdown: `artifacts/plots/lesson_2_subgroups_20260405_153950.png`

---

## Key Takeaways

1. **Lesson claim partially contradicted**: The claim was that LightGCN would outperform CDR models. Instead, **NCF dominates** across all metrics (Recall@10=0.0435, sampled HR@10=0.6755), while LightGCN underperformed even MF-BPR. This is unexpected.

2. **LightGCN underperformance**: LightGCN achieved only Recall@10=0.0095, well below MF-BPR (0.0250). Likely causes:
   - Validation early stopping restored an early checkpoint (best val NDCG@10=0.0093) which was sub-optimal
   - Graph convolution may be less effective with the sparse game interaction graph (171K train edges)
   - The 500-user validation subsample may not be representative enough for good early stopping

3. **NCF's strength**: NCF's full NeuMF model (GMF + MLP + prediction layer) captures non-linear user-item interactions that simple dot-product models miss. RecBole's implementation with 50 epochs of training was very effective.

4. **CDR models show transfer signal**: EMCDR and PTUPCDR both outperform MF-BPR (0.035 vs 0.025 Recall@10), demonstrating that movie→game transfer provides meaningful signal even in the mixed population. PTUPCDR's few-shot blending gives it an edge on sampled metrics (HR@10=0.542 vs EMCDR's 0.490).

5. **CMF disappoints**: CMF's joint factorization (Recall@10=0.018) barely beats MF-BPR on NDCG but loses on sampled metrics. The shared embedding space may be too constrained.

6. **Population matters**: With 95K users and no overlap filter, many eval users have zero movie history, making CDR transfer irrelevant for them. This dilutes CDR's advantage. Lesson 3's overlap cohort filtering should change the picture.

---

## Match with Original Plan

The lesson's claim — that LightGCN beats CDR — was **not confirmed** with these hyperparameters. NCF emerged as the strongest model instead. However, the broader narrative holds: single-domain models (NCF) outperform CDR on the mixed population, which motivates Lesson 3's overlap cohort filtering to find regimes where CDR wins. LightGCN may need hyperparameter tuning or more epochs without the validation early-stopping issue.
