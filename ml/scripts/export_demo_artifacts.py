"""Export all model embeddings + catalog data for the demo web app.

Trains each model on the overlap dataset, extracts embeddings, and saves
everything to artifacts/demo/ as numpy arrays + JSON files.

Usage: PYTHONPATH=. python ml/scripts/export_demo_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.scripts.benchmarks.benchmark_common import (
    load_cross_domain_split, POSITIVE_THRESHOLD, DATA_DIR,
)
from ml.models.lightgcn import LightGCN
from ml.models.emcdr import EMCDRWrapper
from ml.models.ptupcdr import PTUPCDRWrapper
from ml.models.sbert_model import SBERTModel
from ml.scripts.benchmarks.cooc_rerank import build_movie_game_cooc
from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT = Path(_root) / "artifacts" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(_root) / "ml" / "data" / "amazon_2023" / "movies_games.db"


def save_npy(name: str, arr: np.ndarray) -> None:
    path = OUT / f"{name}.npy"
    np.save(path, arr)
    logger.info("Saved %s: shape=%s, %.1fMB", name, arr.shape, arr.nbytes / 1e6)


def export_mappings(data_sd, data_cd) -> None:
    """Export user/item index mappings."""
    json.dump(data_sd.user_to_idx, open(OUT / "user_to_idx.json", "w"))
    json.dump(data_sd.item_to_idx, open(OUT / "item_to_idx.json", "w"))
    json.dump({str(v): k for k, v in data_sd.item_to_idx.items()},
              open(OUT / "idx_to_item.json", "w"))

    # Cross-domain mappings (unified item space for CDR models)
    json.dump(data_cd.user_to_idx, open(OUT / "cross_domain_user_to_idx.json", "w"))
    json.dump(data_cd.item_to_idx, open(OUT / "cross_domain_item_to_idx.json", "w"))
    json.dump({str(v): k for k, v in data_cd.item_to_idx.items()},
              open(OUT / "cross_domain_idx_to_item.json", "w"))

    logger.info("Exported mappings: %d users, %d items (SD), %d items (CD)",
                len(data_sd.user_to_idx), len(data_sd.item_to_idx), len(data_cd.item_to_idx))


def export_catalog(data_sd, data_cd) -> None:
    """Export item catalog and user list from SQLite DB."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Items
    items_rows = conn.execute("SELECT * FROM items ORDER BY idx").fetchall()
    items_list = []
    for row in items_rows:
        d = dict(row)
        # Map to our index space
        ext_id = d.get("external_id", "")
        sd_idx = data_sd.item_to_idx.get(ext_id)
        cd_idx = data_cd.item_to_idx.get(ext_id)
        d["sd_idx"] = sd_idx
        d["cd_idx"] = cd_idx
        items_list.append(d)
    json.dump(items_list, open(OUT / "items.json", "w"), default=str)
    logger.info("Exported %d items to items.json", len(items_list))

    # Users (sample 200 for demo picker — must be in model's user space)
    sd_user_ids = set(data_sd.user_to_idx.keys())
    users_rows = conn.execute(
        "SELECT * FROM users WHERE game_ratings > 0 ORDER BY total_ratings DESC"
    ).fetchall()
    users_list = []
    for row in users_rows:
        d = dict(row)
        if d["external_id"] not in sd_user_ids:
            continue
        d["is_sample"] = True
        users_list.append(d)
        if len(users_list) >= 200:
            break
    json.dump(users_list, open(OUT / "users.json", "w"), default=str)
    logger.info("Exported %d sample users to users.json (filtered to model space)", len(users_list))

    # Train ratings
    ratings = []
    for _, r in data_cd.cross_train.iterrows():
        ratings.append({
            "user_id": str(r["user_id"]),
            "item_id": str(r["item_id"]),
            "rating": float(r["rating"]),
            "domain": str(r.get("domain", "")),
        })
    json.dump(ratings, open(OUT / "train_ratings.json", "w"))
    logger.info("Exported %d train ratings", len(ratings))

    conn.close()


def export_lightgcn(data_sd) -> None:
    """Train LightGCN and export embeddings."""
    t0 = time.time()
    target_mask = np.zeros(data_sd.num_items, dtype=bool)
    for idx in data_sd.target_item_indices:
        target_mask[idx] = True

    model = LightGCN(data_sd.num_users, data_sd.num_items,
                     embedding_dim=96, num_layers=3, device="cpu", dropout=0.1)
    model.fit(data_sd.target_train, data_sd.user_to_idx, data_sd.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=0.001, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD,
              neg_sampling="popularity", neg_popularity_alpha=0.75,
              neg_item_indices=np.array(sorted(data_sd.target_item_indices), dtype=np.int64))
    logger.info("LightGCN trained in %.1fs", time.time() - t0)

    save_npy("lightgcn_user", model.get_user_embeddings())
    save_npy("lightgcn_item", model.get_item_embeddings())


def export_emcdr(data_cd) -> None:
    """Train EMCDR and export embeddings."""
    t0 = time.time()
    model = EMCDRWrapper(data_cd.num_users, data_cd.num_items, embedding_dim=64, device="cpu")
    model.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
              epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    logger.info("EMCDR trained in %.1fs", time.time() - t0)

    save_npy("emcdr_user", model.get_user_embeddings())
    save_npy("emcdr_item", model.get_item_embeddings())


def export_ptupcdr(data_cd) -> None:
    """Train PTUPCDR and export embeddings."""
    t0 = time.time()
    model = PTUPCDRWrapper(data_cd.num_users, data_cd.num_items, embedding_dim=64, device="cpu")
    model.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
              epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD,
              meta_epochs=30, meta_lr=0.001, n_experts=8)
    logger.info("PTUPCDR trained in %.1fs", time.time() - t0)

    save_npy("ptupcdr_user", model.get_user_embeddings())
    save_npy("ptupcdr_item", model.get_item_embeddings())


def export_sbert(data_cd) -> None:
    """Encode items with SBERT and export content embeddings."""
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    games_df = pd.read_parquet(DATA_DIR / "games.parquet")
    items_df = pd.concat([movies_df, games_df], ignore_index=True)

    t0 = time.time()
    model = SBERTModel()
    model.encode_items(items_df, data_cd.item_to_idx)
    logger.info("SBERT encoding done in %.1fs", time.time() - t0)

    save_npy("content_emb", model.item_embeddings)

    # Precompute top-20 similar items per item (for item detail page)
    emb = model.item_embeddings
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms

    # Batch cosine similarity (process in chunks to avoid OOM)
    n_items = emb_norm.shape[0]
    k = 20
    indices = np.zeros((n_items, k), dtype=np.int32)
    scores = np.zeros((n_items, k), dtype=np.float32)

    chunk = 1000
    for start in range(0, n_items, chunk):
        end = min(start + chunk, n_items)
        sim = emb_norm[start:end] @ emb_norm.T  # (chunk, n_items)
        sim[np.arange(end - start), np.arange(start, end)] = -1  # mask self
        top_k = np.argpartition(sim, -k, axis=1)[:, -k:]
        for i in range(end - start):
            sorted_idx = top_k[i][np.argsort(sim[i, top_k[i]])[::-1]]
            indices[start + i] = sorted_idx
            scores[start + i] = sim[i, sorted_idx]

    save_npy("content_sim_indices", indices)
    save_npy("content_sim_scores", scores)


def export_lightgcn_movies(data_cd) -> None:
    """Train LightGCN on movie interactions and export embeddings."""
    from ml.models.id_utils import normalize_id
    movie_train = data_cd.movie_train
    movie_item_ids = set(movie_train["item_id"].map(normalize_id).unique())

    # Create movie-only item mapping
    movie_items = sorted(movie_item_ids & set(data_cd.item_to_idx.keys()))
    movie_item_to_idx = {iid: i for i, iid in enumerate(movie_items)}
    num_movie_items = len(movie_item_to_idx)

    t0 = time.time()
    model = LightGCN(data_cd.num_users, num_movie_items,
                     embedding_dim=96, num_layers=3, device="cpu", dropout=0.1)
    model.fit(movie_train, data_cd.user_to_idx, movie_item_to_idx,
              epochs=30, lr=0.001, reg_lambda=0.001, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD,
              neg_sampling="popularity", neg_popularity_alpha=0.75,
              neg_item_indices=np.arange(num_movie_items, dtype=np.int64))
    logger.info("LightGCN-movies trained in %.1fs (%d items)", time.time() - t0, num_movie_items)

    save_npy("lightgcn_movie_user", model.get_user_embeddings())
    save_npy("lightgcn_movie_item", model.get_item_embeddings())
    json.dump(movie_item_to_idx, open(OUT / "movie_item_to_idx.json", "w"))
    json.dump({str(v): k for k, v in movie_item_to_idx.items()},
              open(OUT / "movie_idx_to_item.json", "w"))


def export_reverse_cooc(data_sd, data_cd) -> None:
    """Build game→movie co-occurrence (reverse direction) for movie recommendations."""
    from ml.models.id_utils import normalize_id
    from collections import defaultdict

    movie_by_u: dict[str, set[str]] = defaultdict(set)
    game_by_u: dict[str, set[str]] = defaultdict(set)

    r_m = pd.to_numeric(data_cd.movie_train["rating"], errors="coerce")
    r_g = pd.to_numeric(data_sd.game_train["rating"], errors="coerce")
    mt = data_cd.movie_train[r_m >= POSITIVE_THRESHOLD]
    gt = data_sd.game_train[r_g >= POSITIVE_THRESHOLD]

    for _, row in mt.iterrows():
        movie_by_u[normalize_id(row["user_id"])].add(normalize_id(row["item_id"]))
    for _, row in gt.iterrows():
        game_by_u[normalize_id(row["user_id"])].add(normalize_id(row["item_id"]))

    # Reverse: game→movie associations
    reverse_cooc: dict[str, dict[str, float]] = {}
    n_overlap = 0
    for u, gs in game_by_u.items():
        ms = movie_by_u.get(u)
        if not ms:
            continue
        n_overlap += 1
        for g in gs:
            row = reverse_cooc.setdefault(g, {})
            for m in ms:
                row[m] = row.get(m, 0) + 1

    # Convert counts to log scores
    for g in reverse_cooc:
        for m in reverse_cooc[g]:
            reverse_cooc[g][m] = float(np.log1p(reverse_cooc[g][m]))

    json.dump(reverse_cooc, open(OUT / "reverse_cooc.json", "w"))
    n_edges = sum(len(v) for v in reverse_cooc.values())
    logger.info("Exported reverse cooc (game→movie): %d games, %d pairs, %d overlap users",
                len(reverse_cooc), n_edges, n_overlap)


def export_game_item_ids(data_sd, data_cd) -> None:
    """Export the set of game item external IDs (from training data, not DB)."""
    game_ids = sorted(set(data_sd.item_to_idx.keys()))  # SD space = game items only
    json.dump(game_ids, open(OUT / "game_item_ids.json", "w"))
    logger.info("Exported %d game item IDs", len(game_ids))


def export_cooc(data_sd) -> None:
    """Build and export co-occurrence matrix."""
    cooc = build_movie_game_cooc(data_sd.movie_train, data_sd.game_train,
                                  rating_threshold=POSITIVE_THRESHOLD)
    # Save as JSON for the backend to load
    json.dump(cooc, open(OUT / "cooc.json", "w"))
    logger.info("Exported cooc: %d movies, %d total pairs",
                len(cooc), sum(len(v) for v in cooc.values()))


def main() -> None:
    logger.info("=== Exporting demo artifacts to %s ===", OUT)

    # Load both single-domain and cross-domain splits
    data_sd = load_cross_domain_split(
        target_domain="game", single_domain_item_space=True,
    )
    data_cd = load_cross_domain_split(
        target_domain="game", single_domain_item_space=False,
    )

    logger.info("SD: %d users, %d items | CD: %d users, %d items",
                data_sd.num_users, data_sd.num_items, data_cd.num_users, data_cd.num_items)

    # Export mappings and catalog
    export_mappings(data_sd, data_cd)
    export_catalog(data_sd, data_cd)

    # Export model embeddings
    export_lightgcn(data_sd)
    export_emcdr(data_cd)
    export_ptupcdr(data_cd)
    export_sbert(data_cd)
    export_lightgcn_movies(data_cd)
    export_reverse_cooc(data_sd, data_cd)
    export_game_item_ids(data_sd, data_cd)
    export_cooc(data_sd)

    logger.info("=== All artifacts exported to %s ===", OUT)
    logger.info("Files: %s", [f.name for f in sorted(OUT.glob("*"))])


if __name__ == "__main__":
    main()
