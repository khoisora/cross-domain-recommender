"""Movie–game co-occurrence reranking.

Builds a sparse movie→game transfer matrix from training co-occurrence
counts, then adds a bonus to candidate game scores at inference time.

For each overlap user: if they liked movie M and game G tend to be liked
by the same users, seeing a user who liked M gives a score boost to G.
Most useful for users with sparse game history but rich movie history.

Score modes:
  raw  — log(1 + count), simple and robust
  pmi  — pointwise mutual information (debiased association)
  npmi — normalized PMI in [-1, 1]

Usage:
    cooc = build_movie_game_cooc(data.movie_train, data.game_train,
                                  rating_threshold=POSITIVE_THRESHOLD)
    predict_fn = wrap_predict_with_cooc(model.predict, data, cooc, lam=0.05)
    metrics = evaluate_cross_domain(ALGO, predict_fn, data)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ml.models.id_utils import normalize_id

logger = logging.getLogger(__name__)


def _positive_interaction_sets(
    movie_train: pd.DataFrame,
    game_train: pd.DataFrame,
    rating_threshold: float,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return per-user positive movie/game item sets."""
    r_m = pd.to_numeric(movie_train["rating"], errors="coerce")
    r_g = pd.to_numeric(game_train["rating"], errors="coerce")
    mt = movie_train[r_m >= rating_threshold]
    gt = game_train[r_g >= rating_threshold]

    movie_by_u: dict[str, set[str]] = defaultdict(set)
    game_by_u: dict[str, set[str]] = defaultdict(set)
    for _, row in mt.iterrows():
        movie_by_u[normalize_id(row["user_id"])].add(normalize_id(row["item_id"]))
    for _, row in gt.iterrows():
        game_by_u[normalize_id(row["user_id"])].add(normalize_id(row["item_id"]))
    return movie_by_u, game_by_u


def build_movie_game_cooc(
    movie_train: pd.DataFrame,
    game_train: pd.DataFrame,
    rating_threshold: float = 4.0,
    score_mode: str = "raw",
    min_count: int = 1,
    shrinkage: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Build movie→game association weights from overlap-user co-occurrence.

    Returns:
        cooc[movie_id][game_id] = transfer weight
    """
    if not len(movie_train) or not len(game_train):
        return {}
    score_mode = score_mode.strip().lower()

    movie_by_u, game_by_u = _positive_interaction_sets(
        movie_train, game_train, rating_threshold
    )

    movie_user_count: dict[str, int] = defaultdict(int)
    game_user_count: dict[str, int] = defaultdict(int)
    pair_count: dict[str, dict[str, int]] = {}
    n_overlap = 0

    # Count co-occurrences: for each overlap user (has both movie and game
    # ratings), every (movie, game) pair the user liked gets a count increment.
    # This is O(users * movies_per_user * games_per_user) but each factor is small.
    for u, ms in movie_by_u.items():
        gs = game_by_u.get(u)
        if not gs:
            continue
        n_overlap += 1
        for m in ms:
            movie_user_count[m] += 1
        for g in gs:
            game_user_count[g] += 1
        for m in ms:
            row = pair_count.setdefault(m, {})
            for g in gs:
                row[g] = row.get(g, 0) + 1

    if n_overlap == 0:
        logger.warning("Co-occurrence: no overlap users with positive movie+game interactions")
        return {}

    # Convert raw counts to association scores. Three modes:
    #   raw:  log(1+count) — simple, robust, favors popular pairs
    #   pmi:  log(P(m,g) / P(m)P(g)) — debiased, can be negative for rare pairs
    #   npmi: pmi / -log(P(m,g)) — normalized to [-1, 1], comparable across scales
    cooc: dict[str, dict[str, float]] = {}
    for m, gs in pair_count.items():
        row: dict[str, float] = {}
        p_m = movie_user_count[m] / n_overlap
        for g, c in gs.items():
            if c < min_count:
                continue
            p_g = game_user_count[g] / n_overlap
            p_mg = c / n_overlap

            if score_mode == "raw":
                score = float(np.log1p(c))
            else:
                if p_m <= 0 or p_g <= 0 or p_mg <= 0:
                    continue
                pmi = float(np.log(p_mg / (p_m * p_g)))
                if score_mode == "pmi":
                    score = pmi
                else:  # npmi
                    score = pmi / float(-np.log(p_mg))

            if score <= 0:
                continue
            # Shrinkage dampens scores for low-count pairs to reduce noise
            if shrinkage > 0:
                score *= c / (c + shrinkage)
            row[g] = score
        if row:
            cooc[m] = row

    n_edges = sum(len(v) for v in cooc.values())
    logger.info(
        "Co-occurrence [%s] overlap_users=%d: %d movies, %d (movie,game) pairs",
        score_mode, n_overlap, len(cooc), n_edges,
    )
    return cooc


def wrap_predict_with_cooc(
    base_predict: Callable[[int], np.ndarray],
    data: Any,
    cooc: dict[str, dict[str, float]],
    lam: float = 0.05,
    rating_threshold: float = 4.0,
    user_agg: str = "sum",
    max_target_train: int | None = None,
) -> Callable[[int], np.ndarray]:
    """Wrap a predict function to add co-occurrence bonus to game scores.

    Args:
        base_predict: (user_idx) -> score array over all items
        data: CrossDomainSplit
        cooc: output of build_movie_game_cooc
        lam: weight of co-occurrence bonus
        rating_threshold: minimum rating for positive movie interactions
        user_agg: "sum" or "mean" over user's movie history
        max_target_train: only apply cooc for users with <= N game train items
            (None = always apply; 0 = cold-start only; 3 = sparse users only)
    """
    if not cooc or lam == 0.0:
        return base_predict

    # Build sparse transfer matrix W where W[movie_row, game_col] = lam * score.
    # At inference: bonus = (user's movie indicator vector) @ W → score boost per game.
    # Using CSR sparse format keeps memory and dot-product cost proportional to nnz.
    movie_keys = sorted(cooc.keys())
    movie_to_r = {m: i for i, m in enumerate(movie_keys)}
    rows, cols, vals = [], [], []
    for m, gs in cooc.items():
        mi = movie_to_r[m]
        for ge, c in gs.items():
            gj = data.item_to_idx.get(ge)
            if gj is None or gj not in data.game_item_indices:
                continue
            rows.append(mi)
            cols.append(int(gj))
            vals.append(lam * float(c))

    if not vals:
        logger.warning("Co-occurrence W matrix empty after item mapping")
        return base_predict

    W = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(movie_keys), data.num_items),
        dtype=np.float64,
    )
    logger.info("Co-occurrence W: shape=%s, nnz=%d", W.shape, W.nnz)

    idx_to_user = {v: k for k, v in data.user_to_idx.items()}

    def predict_with_bonus(uid: int) -> np.ndarray:
        scores = np.array(base_predict(uid), dtype=np.float64, copy=True)

        # Gate: skip if user has too many game training items
        if max_target_train is not None:
            n_seen = len(data.target_train_seen.get(uid, set()))
            if n_seen > max_target_train:
                return scores

        # Get user's positive movie train items
        uid_str = idx_to_user.get(uid)
        if uid_str is None:
            return scores

        r = pd.to_numeric(data.movie_train["rating"], errors="coerce")
        user_movies = data.movie_train[
            (r >= rating_threshold) &
            (data.movie_train["user_id"].map(normalize_id) == uid_str)
        ]["item_id"].map(normalize_id).unique()

        ridx = np.array([movie_to_r[m] for m in user_movies if m in movie_to_r], dtype=np.int32)
        if ridx.size == 0:
            return scores

        u_vec = sparse.csr_matrix(
            (np.ones(len(ridx)), (np.zeros(len(ridx), dtype=np.int32), ridx)),
            shape=(1, W.shape[0]),
        )
        bonus = u_vec.dot(W).toarray().ravel()
        if user_agg == "mean":
            bonus /= float(ridx.size)

        return scores + bonus

    return predict_with_bonus
