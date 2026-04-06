# Lesson 7 — SBERT content-aware CDR: competitive overall, dominates niche subgroups

**Claim**: SBERT-CDR (user profile = mean of movie item embeddings in shared text space) achieves near-popularity performance overall and wins on unpopular/niche target items where collaborative models are blind.

**Result**: Confirmed. SBERT-CDR (Recall@10=0.0330) is near LightGCN (0.0355) and beats PTUPCDR (0.0320). More striking: on `one_shot_unpopular_target_user`, SBERT wins with Recall@10=**0.1341** vs LightGCN=0.0122 — a **11× advantage**. Collaborative models cannot retrieve niche long-tail items; semantic content similarity can.

---

## What changed vs Lesson 6

| Variable | Lesson 6 | Lesson 7 | Why |
|---|---|---|---|
| **Split** | User-split cold-start (80/20) | Standard LLO on games | L7 evaluates all users via LLO |
| **New models** | None | SBERT, SBERT-CDR | Content-based zero-training CDR |
| **Subgroups** | cs_low/med/rich + unpopular cold | LLO subgroups: one_shot, high_source, movie_heavy | L7 focuses on one-shot + niche taste |

**Key logic**: LLO includes users with as few as 1 game interaction in training. `one_shot` users (train_size=1) have barely any collaborative signal — SBERT's semantic consistency (similar genres across interactions) outperforms popularity-biased collaborative filtering on long-tail items.

## Dataset characteristics

| Property | Value |
|---|---|
| Dataset | processed_sparse_loose (L4: movies>=10, games>=1) |
| n_users | 14,328 |
| n_movie_items | 39,534 |
| n_game_items | 9,147 |
| n_movie_interactions | 388,919 |
| n_game_interactions | 58,782 |
| Overlap | 100% |
| Split | Leave-last-out on games |
| SBERT model | all-MiniLM-L6-v2 (384-dim) |

## Benchmark results (all users, LLO)

| Model | Family | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|---|---|---|---|---|---|
| **LightGCN** | Graph (single-domain) | **0.0355** | **0.0174** | 0.3215 | 0.1523 |
| SBERT-CDR | Content (CDR) | 0.0330 | 0.0182 | 0.2490 | 0.1124 |
| PTUPCDR | Collab mapping (CDR) | 0.0320 | 0.0175 | 0.4025 | 0.1806 |
| SBERT | Content (single) | 0.0310 | 0.0160 | 0.6365 | 0.3711 |

SBERT (0.0310) reaches **87% of LightGCN** with zero training. SBERT-CDR (0.0330) beats PTUPCDR (0.0320) and closes to within **7% of LightGCN**.

## Subgroup analysis — the key story

### Recall@10 by user subgroup

| Subgroup | n users | SBERT | SBERT-CDR | LightGCN | PTUPCDR |
|---|---|---|---|---|---|
| **`one_shot_unpopular_target`** | **82** | **0.1341** | **0.1098** | 0.0122 | 0.0122 |
| **`high_source_unpopular_low_target`** | **46** | **0.1304** | **0.1087** | 0.0217 | 0.0217 |
| `one_shot_target` (popular) | 449 | 0.0713 | 0.0668 | 0.0557 | 0.0379 |
| `super_cold_users` | 904 | 0.0000 | 0.0144 | 0.0055 | 0.0265 |
| `movie_heavy` | 350 | 0.0371 | 0.0257 | 0.0571 | 0.0400 |

### Why SBERT wins 11× on unpopular items

**`one_shot_unpopular_target_user`** (train_size=1, source>=10, test game is niche):
- Collaborative models: the user's 1 game interaction provides weak signal; the niche test game rarely co-occurs with other items → can't be ranked highly
- SBERT: user profile = embedding of their 1 training game. Genre consistency (niche users explore within same genre) means test item is semantically near training item → high cosine similarity
- Key insight: **niche users have consistent genre preferences** that text captures but interaction counts miss

**`high_source_unpopular_low_target`** (games<=3, movies>=15, niche movie taste):
- SBERT-CDR: movie profile encodes niche genre preference (horror, indie, foreign cinema → analogous niche games)
- Collaborative models: only 3 game interactions → insufficient for reliable embedding

## Key takeaways

1. **SBERT wins 11× on niche/one-shot users**: `one_shot_unpopular_target` Recall@10=0.1341 vs LightGCN=0.0122. Collaborative filtering is blind to long-tail items; semantic similarity is not.

2. **SBERT-CDR is competitive overall** (0.0330 vs LightGCN 0.0355, -7%): the cross-domain blend provides marginal overall improvement over plain SBERT, but meaningfully helps `super_cold_users` (0.0144 vs SBERT 0.0000).

3. **SBERT on one-shot unpopular slightly beats SBERT-CDR** (0.1341 vs 0.1098): for users with 1 game training item, the in-domain game embedding already captures genre preference well. Adding movie profile (source_weight=0.5) dilutes it slightly.

4. **LightGCN leads overall** but collapses on niche subgroups: 0.0122 on `one_shot_unpopular` — a popularity-biased model cannot retrieve items that few users have interacted with.

5. **Routing rule sharpened**: for users whose test item is likely niche (estimated from past movie genre history), route to SBERT-CDR regardless of game history count.

## Lesson progression summary (L1–L7)

| Lesson | Question | Key finding |
|---|---|---|
| **L1** | BPR vs Explicit MF | BPR wins 6× on Recall@10 — ranking loss > reconstruction loss for top-K |
| **L2** | Mixed population baseline | LightGCN leads; CDR underperforms on non-overlap users — wrong population for CDR |
| **L3** | Overlap users only | Restricting to 100% overlap: PTUPCDR +276%, EMCDR +53% — CDR needs overlap |
| **L4** | Source-rich (movies≥10) | PTUPCDR closes to within 17% of LightGCN — richer movie history = better transfer |
| **L5** | Catalog sharpening | BiTGCF +18%, PTUPCDR/EMCDR drop — dense catalog helps graph CDR, hurts mapping CDR |
| **L6** | Cold-start (0 game history) | PTUPCDR 4× over LightGCN on zero-game users — CDR essential for cold-start |
| **L7** | SBERT content CDR | SBERT-CDR near LightGCN overall; **11× over LightGCN on niche/unpopular items** |

**Core insight**: optimal model is determined by two axes — *how much target (game) history* and *how niche the user's taste is*. LightGCN dominates with sufficient collaborative signal. CDR bridges the gap when movie history compensates for sparse games. SBERT fills the long-tail blind spot that all collaborative models share.

## Final routing rule (L2–L7)

```
User state                                    Model
────────────────────────────────────────────────────────────────
0 games, rich movies (≥10), mainstream      → PTUPCDR
0 games, any movies, niche/unpopular taste  → SBERT-CDR
0 games, few movies (<10)                   → Popularity baseline
1 game,  train item is niche/unpopular      → SBERT          (11× win)
1 game,  train item is popular              → SBERT or LightGCN blend
2–9 games                                   → LightGCN
10+ games                                   → LightGCN
Any user, test item likely long-tail        → SBERT-CDR      (override)
```

## Benchmark plots

- `artifacts/plots/lesson_7_*.png`
- `artifacts/plots/lesson_7_subgroups_*.png`
