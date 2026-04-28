# User Journey & Recommender Routing

How the demo backend chooses between Single-Domain (SDR) and Cross-Domain (CDR) recommenders, and how the experience evolves as a new user rates items.

## TL;DR

| Game ratings | Segment | Routed to | Why |
|---|---|---|---|
| 0 | `cold_start` | **CDR rows** (CMF, EMCDR, PTUPCDR) | No collaborative game signal — transfer movie taste into game space. |
| ≥ 1 | `warm` | **SDR rows** (LightGCN, MF-BPR, NeuMF) | Game collaborative signal exists — use direct game-space models. |

Cooc and SBERT-CDR rows are shown in both segments — they always activate from movie ratings.

The classifier lives at `backend/demo/recommender.py:_classify_segment`. Counts come from `store.user_ratings`, which is the in-memory mirror of the SQLite ratings table (`data/demo.db`).

## Routing in detail

### Cold-start path (0 game ratings)

Rendered rows, in order:
1. **CMF + Co-occurrence** — "Recommended for You"
2. **EMCDR + Co-occurrence** — "Based on Your Movie Taste"
3. **PTUPCDR + Co-occurrence** — "You Might Also Like"
4. **Co-occurrence standalone** — "Players Who Watched Your Movies Also Played" (only if the user has ≥1 movie rating with rating ≥ 4)
5. **SBERT-CDR Hidden Gems** — niche games whose descriptions match the user's recent liked movies (only if ≥1 liked movie)

A cold-start user with **zero ratings** sees no rows. Once they rate one or more movies, CDR rows appear.

### Warm path (≥ 1 game rating)

Rendered rows, in order:
1. **LightGCN + Co-occurrence** — "Top Picks for You"
2. **MF-BPR + Co-occurrence** — "You Might Also Like"
3. **NeuMF + Co-occurrence** — "Neural Collaborative Picks"
4. **Co-occurrence standalone** — same as cold-start, gated on movie ratings
5. **SBERT-CDR Hidden Gems** — same as cold-start, gated on liked movies

The first game rating flips the segment from `cold_start` to `warm` and the row stack changes accordingly.

## New users (registered via `POST /api/users`)

A registered user gets `external_id = "custom_<timestamp>"`. This ID is **not** in any precomputed user-to-index mapping, so trained user embeddings (LightGCN/MF-BPR/NeuMF/CMF/EMCDR/PTUPCDR) cannot be looked up directly.

The recommender handles this via **fold-in**: when the trained user vector is missing, it builds one on the fly as the rating-weighted mean of the user's rated-item embeddings. Implementation: `_resolve_sd_user_vec`, `_resolve_cd_user_vec`, `_foldin_user_vec` in `recommender.py`.

- **SD fold-in** uses **game ratings only** (the SD item embedding space is games-only).
- **CD fold-in** uses **movies + games** (the CD item embedding space is unified).

### What a new user sees, step by step

| Action | Segment | Rows that appear |
|---|---|---|
| Just registered, no ratings | `cold_start` | None (no signal to transfer) |
| Rates 1+ movies | `cold_start` | CMF, EMCDR, PTUPCDR (CDR via CD fold-in from movie embeddings) + cooc + hidden gems |
| Rates 1+ games (with or without prior movies) | `warm` | LightGCN, MF-BPR, NeuMF (SDR via SD fold-in from game embeddings) + cooc + hidden gems |
| Returns later (after server restart) | unchanged | Same — ratings persist in SQLite and are restored to memory at startup with their `domain` re-attached from the item catalog. |

### Why this works without retraining

Both SDR and CDR scoring is `item_embedding @ user_vector`. Item embeddings are fixed, so substituting a fold-in user vector produces meaningful (though not optimal) scores immediately. The hourly retrain scheduler (`backend/demo/retrain.py`) eventually folds these users into the trained graph; until then, fold-in keeps the experience populated.

## Always-on cross-domain signals

Two rows are cross-domain regardless of segment because they consume movie ratings directly, not user embeddings:

- **Co-occurrence standalone** — for each rated movie, look up `cooc[movie_id] -> {game_id: score}` and rank games. Updates instantly when a movie is rated.
- **SBERT-CDR Hidden Gems** — semantic match between the user's last 5 liked movies' SBERT vectors and game descriptions, restricted to bottom-50%-popularity games (so head items don't dominate). See `_movie_sbert_profile` and `_hidden_gems_row`.

Both also receive a cooc boost (`COOC_LAM = 0.05`).

## Persistence

- Ratings → `data/demo.db` (SQLite) via `DemoDB.add_rating`. The same call updates the in-memory `DemoStore.user_ratings` so re-ranking is instant.
- On startup, `lifespan()` calls `db.get_all_runtime_ratings()` and replays them into memory, looking up `domain` from the item catalog so segment classification, cooc, and SBERT all keep working after a restart.

## Front-end surface

- The segment + explainer is rendered as a banner on `/recs/<user_id>` (recs.js → `segment-banner`). The `warm` banner uses an orange highlight; cold-start uses the default purple.
- The home page tabs (`Cold Start`, `Few Target`, `Balanced`, `New`) are built from `/api/user-groups`. The `New` tab lists the 10 most recently registered non-sample users with `created_at >= 2026-01-01`.
