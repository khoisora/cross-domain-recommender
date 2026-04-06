# Lesson 7 — SBERT content-aware CDR: null result overall, wins on unpopular items

**Claim**: SBERT-CDR (user profile = mean of movie item embeddings in shared text space) beats collaborative CDR (PTUPCDR) overall and especially on unpopular-item users.

**Result**: Mixed. SBERT-CDR fails overall on cold-start users (Recall@10=0.0025 vs PTUPCDR=0.0380 — 15× worse). Text similarity alone does not capture behavioral preference. However, the subgroup analysis reveals an important exception: on `one_shot_unpopular_target_user` (cold users whose only game test item is an unpopular/niche game), SBERT-CDR (0.0059) beats PTUPCDR (0.0000). Collaborative CDR is blind to items it hasn't seen enough, but content CDR can find niche games via semantic similarity.

---

## What changed vs Lesson 6

| Variable | Lesson 6 | Lesson 7 | Why |
|---|---|---|---|
| **Dataset** | processed_overlap (L3) | processed_overlap (L3) | Same cold-start data |
| **Split** | User-split cold-start | User-split cold-start | Same evaluation protocol |
| **New models** | None | SBERT, SBERT-CDR | Content-based zero-training CDR |
| **Subgroups** | cs_low/med/rich_movies | + unpopular/popular target subgroups | L7 focuses on item popularity dimension |

**Key logic**: All 4 models are evaluated on the same 2,000 cold users (zero game training history). SBERT uses in-domain game profiles → collapses for cold users. SBERT-CDR uses movie profiles to recommend games via semantic similarity in the shared SBERT space. PTUPCDR uses a learned movie→game embedding mapping.

## Dataset characteristics

| Property | Value |
|---|---|
| Dataset | processed_overlap (L3: overlap users, movies >= 5, games >= 1) |
| Eligible users (games>=2, movies>=5) | 12,654 |
| Warm users | 10,124 |
| Cold eval users | 2,000 |
| Game train interactions (warm) | 64,057 |
| Movie interactions (all) | 430,736 |
| Split | User-split cold-start (80/20) |
| SBERT model | all-MiniLM-L6-v2 (384-dim) |

## Benchmark results (cold users only)

| Model | Family | Recall@10 | NDCG@10 | Sampled HR@10 | Sampled NDCG@10 |
|---|---|---|---|---|---|
| **PTUPCDR** | Collab mapping (CDR) | **0.0380** | **0.0184** | 0.4965 | 0.2167 |
| LightGCN | Graph (single-domain) | 0.0075 | 0.0045 | 0.1845 | 0.0769 |
| SBERT-CDR | Content (CDR) | 0.0025 | 0.0010 | 0.1120 | 0.0443 |
| SBERT | Content (single-domain) | 0.0020 | 0.0006 | — | — |

*Note: SBERT sampled HR=1.0 is a tie-breaking artifact (uniform scores for cold users) — not a real signal. Full-rank metrics are reliable.*

## Subgroup analysis

### Recall@10 by user subgroup

| Subgroup | n users | PTUPCDR | SBERT-CDR | Δ (CDR winner) |
|---|---|---|---|---|
| `cs_rich_movies` (>=50 movies) | 158 | 0.0253 | **0.0127** | PTUPCDR +2× |
| `cs_med_movies` (15–49 movies) | 649 | **0.0493** | 0.0015 | PTUPCDR +33× |
| `cs_low_movies` (<15 movies) | 1,193 | 0.0335 | 0.0017 | PTUPCDR +20× |
| `one_shot_popular_target_user` | 1,661 | **0.0458** | 0.0018 | PTUPCDR +25× |
| **`one_shot_unpopular_target_user`** | **339** | **0.0000** | **0.0059** | **SBERT-CDR wins** |
| `high_source_unpopular_low_target` | 9 | 0.0000 | 0.0000 | Tie (too small) |

### Key subgroup finding

**SBERT-CDR beats PTUPCDR on `one_shot_unpopular_target_user` (0.0059 vs 0.0000)**:
- Cold users whose test game is a niche/unpopular title
- PTUPCDR: completely blind — unpopular games have sparse collaborative signals, can't be retrieved
- SBERT-CDR: uses semantic text similarity → "indie horror movie fan → indie horror game" works even for long-tail items
- This is the correct regime for content CDR: niche taste + niche target

**PTUPCDR dominates everywhere else** because behavioral collaborative signals are far stronger than text similarity for popular items where enough training data exists.

## Key takeaways

1. **Null result overall — text similarity ≠ behavioral preference**: SBERT-CDR (0.0025) is 15× worse than PTUPCDR (0.0380) on cold users. Having "similar text" between a movie and a game does not mean a user who liked the movie will like the game.

2. **SBERT collapses for cold users** (0.0020): without game history to build a profile from, SBERT assigns uniform scores. This confirms that in-domain content models are useless for zero-game-history users.

3. **SBERT-CDR wins on long-tail unpopular items** (0.0059 vs 0.0000): the one regime where content CDR outperforms collaborative CDR. Collaborative models are blind to niche items; semantic similarity can still find them. This suggests a hybrid: PTUPCDR for most users + SBERT-CDR fallback for niche-taste cold users.

4. **Rich movie history helps SBERT-CDR** (cs_rich_movies: 0.0127 vs 0.0017 for low-movie users): more source embeddings = more accurate semantic profile. SBERT-CDR is most useful when the user has many diverse movie interactions.

5. **PTUPCDR's sweet spot is cs_med_movies** (0.0493): 15–49 movies gives enough source signal for the hypernetwork to learn a personalized mapping without overfitting to noise.

## The routing rule (updated with L7)

| User state | Recommended model | Key evidence |
|---|---|---|
| 0 games, rich movies (>=50), popular taste | PTUPCDR | L6, L7 |
| 0 games, any movies, unpopular/niche taste | SBERT-CDR | L7 subgroup |
| 0 games, few movies (<10) | Popularity baseline | L6 |
| 1–2 games, rich movies | PTUPCDR | L4, L6 |
| 3–9 games | LightGCN | L2, L4 |
| 10+ games | LightGCN | L2 |

## Benchmark plots

- `artifacts/plots/lesson_7_*.png`
- `artifacts/plots/lesson_7_subgroups_*.png`
