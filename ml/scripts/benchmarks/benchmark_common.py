"""Common utilities for the cross-domain benchmark suite.

Imported by all bench_*.py scripts. Provides:
  - `load_cross_domain_split` / `load_user_split_cold_start_split`: parquet →
    leave-last-out (or warm/cold user split) → `CrossDomainSplit`.
  - `evaluate_cross_domain`: single full-rank eval entry point.
  - `save_result`: JSON output with `dataset_info` for reproducibility.
  - `verify_no_leakage`, `make_full_rank_val_fn`, and CLI/logging helpers.

Protocol: movie → game cross-domain, per-user leave-last-out, full-rank Recall@10 + NDCG@10 + HitRate@10.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.evaluator import evaluate_full_rank
from ml.evaluation.metrics import aggregate_metrics, compute_all_metrics
from ml.models.id_utils import normalize_id

logger = logging.getLogger(__name__)

SEED = 42
POSITIVE_THRESHOLD = 4  # rating >= 4 counts as positive
K = 10  # all metrics @10 only

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed_sparse_loose"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"


# ── Logging / CLI ─────────────────────────────────────────────────────────────

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_common_args(parser) -> None:
    parser.add_argument("--target", default="game", choices=["game", "movie"])
    parser.add_argument("--lesson", type=int, default=1,
                        help="Lesson number for result tagging")


# ── CrossDomainSplit ──────────────────────────────────────────────────────────

@dataclass
class CrossDomainSplit:
    """Container for train/test splits, ID maps, and eval structures."""

    # Per-user leave-last-out splits
    game_train: pd.DataFrame
    game_test: pd.DataFrame
    movie_train: pd.DataFrame
    movie_test: pd.DataFrame

    # target_train + all source interactions — used by joint-training CDR models.
    cross_train: pd.DataFrame

    # Raw string IDs → contiguous ints.
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]

    # Per-domain item indices — used to mask non-target items in full-rank eval.
    game_item_indices: set[int]
    movie_item_indices: set[int]

    # {user_idx: set(item_idx)}. "relevant" = test rows with rating >= POSITIVE_THRESHOLD.
    # "seen" = all train items (masked in eval to avoid trivial hits).
    game_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_train_seen: dict[int, set[int]] = field(default_factory=dict)
    movie_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    movie_train_seen: dict[int, set[int]] = field(default_factory=dict)

    eval_user_indices: list[int] = field(default_factory=list)
    user_subgroups: dict[str, list[int]] = field(default_factory=dict)

    target_domain: str = "game"
    device: str = "cpu"
    dataset_info: dict[str, Any] = field(default_factory=dict)

    @property
    def target_train(self) -> pd.DataFrame:
        return self.game_train if self.target_domain == "game" else self.movie_train

    @property
    def target_test(self) -> pd.DataFrame:
        return self.game_test if self.target_domain == "game" else self.movie_test

    @property
    def target_item_indices(self) -> set[int]:
        return self.game_item_indices if self.target_domain == "game" else self.movie_item_indices

    @property
    def target_test_relevant(self) -> dict[int, set[int]]:
        return self.game_test_relevant if self.target_domain == "game" else self.movie_test_relevant

    @property
    def target_train_seen(self) -> dict[int, set[int]]:
        return self.game_train_seen if self.target_domain == "game" else self.movie_train_seen

    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self.item_to_idx)


# ── Split & map helpers ───────────────────────────────────────────────────────

def _leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-last-out, sorted by timestamp.

    n == 1 → 0 train, 1 test   (super cold-start — relies on source domain)
    n == 2 → 1 train, 1 test
    n >= 3 → (n-2) train, 1 val, 1 test
    """
    cols = list(df.columns) if df is not None and len(df) else \
        ["user_id", "item_id", "rating", "domain", "timestamp"]
    empty = pd.DataFrame(columns=cols)
    if df is None or len(df) == 0:
        return empty.copy(), empty.copy(), empty.copy()

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    idx_in_user = df.groupby("user_id").cumcount()
    user_n = df.groupby("user_id")["user_id"].transform("size")
    pos_from_end = user_n - 1 - idx_in_user  # 0 = last (test), 1 = val, >=2 = train

    # Train threshold: for n>=3 the earliest train row has pos_from_end=2;
    # for n==2 it's 1; n==1 has no train rows (threshold set out of range).
    train_threshold = np.where(user_n >= 3, 2, np.where(user_n == 2, 1, user_n + 1))

    test = df[pos_from_end == 0]
    val = df[(pos_from_end == 1) & (user_n >= 3)]
    train = df[pos_from_end >= train_threshold]

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _build_eval_structures(
    test_df: pd.DataFrame, train_df: pd.DataFrame,
    user_to_idx: dict[str, int], item_to_idx: dict[str, int],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Return (test_relevant, train_seen). Test uses rating>=POSITIVE_THRESHOLD; train uses all."""
    def _group(df: pd.DataFrame, positive_only: bool) -> dict[int, set[int]]:
        if len(df) == 0:
            return {}
        uids = df["user_id"].map(normalize_id).map(user_to_idx.get)
        iids = df["item_id"].map(normalize_id).map(item_to_idx.get)
        mask = uids.notna() & iids.notna()
        if positive_only:
            mask &= df["rating"] >= POSITIVE_THRESHOLD
        out: dict[int, set[int]] = {}
        for u, i in zip(uids[mask].astype(int), iids[mask].astype(int)):
            out.setdefault(int(u), set()).add(int(i))
        return out

    return _group(test_df, True), _group(train_df, False)


def _build_id_maps(
    ratings: pd.DataFrame, target_df: pd.DataFrame,
    game_item_ids: set[str], movie_item_ids: set[str],
    target_domain: str, single_domain_item_space: bool,
) -> tuple[dict[str, int], dict[str, int], dict[int, str], set[int], set[int]]:
    """Return (user_to_idx, item_to_idx, idx_to_item, game_item_idx, movie_item_idx).

    Single-domain models keep embeddings compact by only indexing target-domain items.
    CDR models must use the unified space (single_domain_item_space=False).
    """
    if single_domain_item_space:
        users = sorted({normalize_id(u) for u in target_df["user_id"].unique()})
        items = sorted(game_item_ids if target_domain == "game" else movie_item_ids)
        item_to_idx = {it: i for i, it in enumerate(items)}
        all_idx = set(range(len(items)))
        game_idx = all_idx if target_domain == "game" else set()
        movie_idx = all_idx if target_domain == "movie" else set()
    else:
        users = sorted({normalize_id(u) for u in ratings["user_id"].unique()})
        items = sorted({normalize_id(it) for it in ratings["item_id"].unique()})
        item_to_idx = {it: i for i, it in enumerate(items)}
        game_idx = {item_to_idx[iid] for iid in game_item_ids if iid in item_to_idx}
        movie_idx = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_item, game_idx, movie_idx


def _popularity_map(df: pd.DataFrame, item_to_idx: dict[str, int]) -> tuple[dict[int, int], float]:
    """Per-item interaction counts (keyed by item index) + median for niche detection."""
    counts = df.groupby("item_id").size().to_dict()
    pop = {
        item_to_idx[normalize_id(iid)]: int(c)
        for iid, c in counts.items() if normalize_id(iid) in item_to_idx
    }
    median = float(np.median(list(pop.values()))) if pop else 1.0
    return pop, median


def _normalized_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    return {normalize_id(k): int(v) for k, v in df.groupby(col).size().items()}


def _items_by_user(df: pd.DataFrame, item_to_idx: dict[str, int]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for uid_raw, grp in df.groupby("user_id"):
        out[normalize_id(uid_raw)] = {
            item_to_idx[normalize_id(iid)]
            for iid in grp["item_id"] if normalize_id(iid) in item_to_idx
        }
    return out


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _cohort_filter_string() -> str:
    meta = DATA_DIR / "dataset_metadata.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("cohort_filter", "unknown")
    return "unknown"


def _subsample(users: list[int], max_n: int | None, seed: int) -> list[int]:
    if max_n and len(users) > max_n:
        return sorted(np.random.RandomState(seed).choice(users, max_n, replace=False))
    return users


# ── LLO loader ────────────────────────────────────────────────────────────────

def load_cross_domain_split(
    target_domain: str = "game",
    max_eval_users: int = 2000,
    single_domain_item_space: bool = False,
) -> CrossDomainSplit:
    """Load ratings.parquet → per-user LLO splits + eval structures.

    single_domain_item_space=True: item_to_idx restricted to target-domain items
        (use for MF-BPR, LightGCN, NCF). CDR models must leave this False.
    """
    data_path = DATA_DIR / "ratings.parquet"
    logger.info("Loading %s (target=%s)", data_path, target_domain)
    ratings = pd.read_parquet(data_path)

    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df = ratings[ratings["domain"] == "game"].copy()
    logger.info("Loaded %d ratings (movie=%d, game=%d)", len(ratings), len(movie_df), len(game_df))

    game_train, game_val, game_test = _leave_last_out(game_df)
    movie_train, _, movie_test = _leave_last_out(movie_df)
    logger.info("Game split: train=%d, val=%d, test=%d", len(game_train), len(game_val), len(game_test))
    logger.info("Movie split: train=%d, test=%d", len(movie_train), len(movie_test))

    target_train = game_train if target_domain == "game" else movie_train
    source_all = movie_df if target_domain == "game" else game_df
    cross_train = pd.concat([target_train, source_all], ignore_index=True)

    target_df = game_df if target_domain == "game" else movie_df
    game_item_ids = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids = {normalize_id(x) for x in movie_df["item_id"].unique()}
    user_to_idx, item_to_idx, idx_to_item, game_item_indices, movie_item_indices = _build_id_maps(
        ratings, target_df, game_item_ids, movie_item_ids, target_domain, single_domain_item_space,
    )
    logger.info("ID maps: %d users, %d items (game=%d, movie=%d)",
                len(user_to_idx), len(item_to_idx),
                len(game_item_indices), len(movie_item_indices))

    game_test_relevant, game_train_seen = _build_eval_structures(game_test, game_train, user_to_idx, item_to_idx)
    game_val_relevant, _ = _build_eval_structures(game_val, game_train, user_to_idx, item_to_idx)
    movie_test_relevant, movie_train_seen = _build_eval_structures(movie_test, movie_train, user_to_idx, item_to_idx)

    target_test_relevant = game_test_relevant if target_domain == "game" else movie_test_relevant
    eval_user_indices = _subsample(sorted(target_test_relevant.keys()), max_eval_users, SEED)
    logger.info("Eval users: %d (of %d with relevant test items)",
                len(eval_user_indices), len(target_test_relevant))

    subgroups = _llo_subgroups(
        eval_user_indices, user_to_idx, item_to_idx,
        game_df, movie_df, target_train, target_domain,
    )
    for name, uids in subgroups.items():
        if uids:
            logger.info("  subgroup %-35s : %d users", name, len(uids))

    dataset_info = {
        "cohort_filter": _cohort_filter_string(),
        "n_users": len(user_to_idx),
        "n_movie_interactions": len(movie_df),
        "n_game_interactions": len(game_df),
        "split": f"leave-last-out on {target_domain}s",
    }

    return CrossDomainSplit(
        game_train=game_train, game_test=game_test,
        movie_train=movie_train, movie_test=movie_test,
        cross_train=cross_train,
        user_to_idx=user_to_idx, item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        game_item_indices=game_item_indices, movie_item_indices=movie_item_indices,
        game_test_relevant=game_test_relevant, game_val_relevant=game_val_relevant,
        game_train_seen=game_train_seen,
        movie_test_relevant=movie_test_relevant, movie_train_seen=movie_train_seen,
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain=target_domain, device=_device(),
        dataset_info=dataset_info,
    )


def _llo_subgroups(
    eval_user_indices: list[int],
    user_to_idx: dict[str, int], item_to_idx: dict[str, int],
    game_df: pd.DataFrame, movie_df: pd.DataFrame,
    target_train: pd.DataFrame, target_domain: str,
) -> dict[str, list[int]]:
    """Categorize eval users by cold-start severity and cross-domain activity.

    Users can be in multiple cold-start subgroups, but movie_heavy/balanced/
    game_heavy/sparse_target are mutually exclusive.
    """
    item_pop, median_pop = _popularity_map(target_train, item_to_idx)
    user_train_size = _normalized_counts(target_train, "user_id")
    user_train_items = _items_by_user(target_train, item_to_idx)
    user_game_count = _normalized_counts(game_df, "user_id")
    user_movie_count = _normalized_counts(movie_df, "user_id")

    def _niche(uid_norm: str) -> bool:
        items = user_train_items.get(uid_norm, set())
        if not items:
            return False
        return sum(1 for i in items if item_pop.get(i, 0) <= median_pop) > len(items) / 2

    subgroups: dict[str, list[int]] = {
        "super_cold_users": [],                 # 0 target train, rich source
        "one_shot_target_user": [],             # 1 target train, rich source
        "one_shot_unpopular_target_user": [],   # 1 target train + niche
        "high_source_low_target": [],           # <=3 target, >=15 source (CDR sweet spot)
        "high_source_unpopular_low_target": [],
        "movie_heavy": [],
        "balanced": [],
        "game_heavy": [],
        "sparse_target": [],
    }
    idx_to_user = {v: k for k, v in user_to_idx.items()}

    for uid_idx in eval_user_indices:
        uid_str = idx_to_user[uid_idx]
        gc = user_game_count.get(uid_str, 0)
        mc = user_movie_count.get(uid_str, 0)
        target_total = gc if target_domain == "game" else mc
        source_count = mc if target_domain == "game" else gc
        train_size = user_train_size.get(uid_str, 0)
        niche = _niche(uid_str)

        if train_size == 0 and source_count >= 10:
            subgroups["super_cold_users"].append(uid_idx)
        if train_size == 1 and source_count >= 10:
            subgroups["one_shot_unpopular_target_user" if niche else "one_shot_target_user"].append(uid_idx)
        if target_total <= 3 and source_count >= 15:
            subgroups["high_source_unpopular_low_target" if niche else "high_source_low_target"].append(uid_idx)

        if target_total <= 3 and source_count < 10:
            subgroups["sparse_target"].append(uid_idx)
        elif target_total > 3:
            if mc > 2 * gc:
                subgroups["movie_heavy"].append(uid_idx)
            elif gc > 2 * mc:
                subgroups["game_heavy"].append(uid_idx)
            else:
                subgroups["balanced"].append(uid_idx)
    return subgroups


# ── Eval entry point + artifact saving ────────────────────────────────────────

def evaluate_cross_domain(
    model_name: str,
    predict_fn,
    data: CrossDomainSplit,
) -> dict[str, Any]:
    """Full-rank evaluation entry point for all bench scripts."""
    return evaluate_full_rank(model_name, predict_fn, data)


def make_full_rank_val_fn(data: CrossDomainSplit) -> Callable[[Any], float]:
    """Return a `val_fn(model)` that computes full-rank NDCG@10 on val users.

    Used by LightGCN-family models for early stopping. Masks non-target items
    and train-seen items the same way the main evaluator does.
    """
    target_mask = np.zeros(data.num_items, dtype=bool)
    for idx in data.target_item_indices:
        target_mask[idx] = True
    val_users = [uid for uid in data.eval_user_indices if data.game_val_relevant.get(uid)][:500]

    def val_fn(model) -> float:
        per_user = []
        for uid in val_users:
            relevant = data.game_val_relevant.get(uid, set())
            if not relevant:
                continue
            scores = model.predict(uid).copy()
            scores[~target_mask] = -np.inf
            for iid in data.target_train_seen.get(uid, set()):
                if 0 <= iid < data.num_items:
                    scores[iid] = -np.inf
            top = np.argsort(scores)[::-1][:K].tolist()
            per_user.append(compute_all_metrics(top, relevant, k_values=[K]))
        return aggregate_metrics(per_user).get("ndcg@10", 0.0) if per_user else 0.0

    return val_fn


def verify_no_leakage(data: CrossDomainSplit) -> None:
    """Raise if target-train and target-test share (user, item) pairs."""
    train_pairs = set(zip(data.target_train["user_id"], data.target_train["item_id"]))
    test_pairs = set(zip(data.target_test["user_id"], data.target_test["item_id"]))
    assert not (train_pairs & test_pairs), "LEAK: test items found in train!"
    logger.info("Leakage check passed")


def save_result(
    algo: str,
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    lesson: int,
    train_time: float = 0.0,
    description: str = "",
) -> None:
    """Write artifacts/results/<algo>_lesson<N>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{algo.lower()}_lesson{lesson}.json"
    entry = {
        "model": algo,
        "lesson": lesson,
        "description": description,
        "time_completed": datetime.now().isoformat(),
        "train_time_s": train_time,
        "dataset_info": dataset_info,
        **metrics,
    }
    path.write_text(json.dumps(entry, indent=2, default=str))
    logger.info("Saved %s results to %s", algo, path)


# ── Cold-start user-split loader (Lesson 6) ───────────────────────────────────

def load_user_split_cold_start_split(
    cold_start_ratio: float = 0.2,
) -> CrossDomainSplit:
    """80/20 warm/cold user split for cold-start evaluation.

    Cold users: ALL game interactions removed from training. Test = their
    chronologically last game interaction. Warm users: everything kept.
    """
    MIN_GAMES, MIN_MOVIES, MAX_EVAL_USERS = 2, 5, 2000
    ratings = pd.read_parquet(DATA_DIR / "ratings.parquet")
    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df = ratings[ratings["domain"] == "game"].copy()
    logger.info("Cold-start split: %d ratings (movie=%d, game=%d)",
                len(ratings), len(movie_df), len(game_df))

    game_counts = game_df.groupby("user_id").size()
    movie_counts = movie_df.groupby("user_id").size()
    eligible = sorted(set(game_counts[game_counts >= MIN_GAMES].index) &
                      set(movie_counts[movie_counts >= MIN_MOVIES].index))
    logger.info("Eligible users: %d (games>=%d, movies>=%d)",
                len(eligible), MIN_GAMES, MIN_MOVIES)

    rng = np.random.RandomState(SEED)
    n_cold = max(1, int(len(eligible) * cold_start_ratio))
    cold_ids = set(rng.choice(eligible, size=n_cold, replace=False).tolist())
    warm_ids = set(eligible) - cold_ids
    logger.info("Warm: %d, Cold: %d", len(warm_ids), len(cold_ids))

    game_train = game_df[game_df["user_id"].isin(warm_ids)].copy()
    cold_games = game_df[game_df["user_id"].isin(cold_ids)].copy()
    if cold_games["timestamp"].notna().any():
        cold_games = cold_games.sort_values("timestamp")
    game_test = cold_games.groupby("user_id").tail(1).copy()
    cross_train = pd.concat([movie_df, game_train], ignore_index=True)

    game_item_ids = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids = {normalize_id(x) for x in movie_df["item_id"].unique()}
    user_to_idx, item_to_idx, idx_to_item, game_item_indices, movie_item_indices = _build_id_maps(
        ratings, ratings, game_item_ids, movie_item_ids,
        target_domain="game", single_domain_item_space=False,
    )

    game_test_relevant, game_train_seen = _build_eval_structures(
        game_test, game_train, user_to_idx, item_to_idx,
    )

    # Drop cold users whose test game was never seen during warm training.
    warm_game_items = {normalize_id(iid) for iid in game_train["item_id"].unique()}
    valid = [
        uid for uid, items in game_test_relevant.items()
        if any(idx_to_item.get(i, "") in warm_game_items for i in items)
    ]
    logger.info("Cold users with valid test items: %d / %d", len(valid), len(cold_ids))
    eval_user_indices = _subsample(sorted(valid), MAX_EVAL_USERS, SEED)

    subgroups = _cold_start_subgroups(
        eval_user_indices, user_to_idx, item_to_idx,
        game_train, movie_df, cold_ids, movie_counts, game_test_relevant,
    )
    for name, uids in subgroups.items():
        logger.info("  subgroup %-35s : %d users", name, len(uids))

    dataset_info = {
        "cohort_filter": "cold-start user-split (80/20)",
        "n_users": len(user_to_idx),
        "n_warm_users": len(warm_ids),
        "n_cold_users": len(cold_ids),
        "n_eval_cold_users": len(eval_user_indices),
        "n_movie_interactions": len(movie_df),
        "n_game_interactions": len(game_train),
        "n_game_interactions_warm": len(game_train),
        "split": "user-split cold-start on games",
    }

    return CrossDomainSplit(
        game_train=game_train, game_test=game_test,
        movie_train=movie_df, movie_test=pd.DataFrame(columns=movie_df.columns),
        cross_train=cross_train,
        user_to_idx=user_to_idx, item_to_idx=item_to_idx, idx_to_item=idx_to_item,
        game_item_indices=game_item_indices, movie_item_indices=movie_item_indices,
        game_test_relevant=game_test_relevant, game_train_seen=game_train_seen,
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain="game", device=_device(),
        dataset_info=dataset_info,
    )


def _cold_start_subgroups(
    eval_user_indices: list[int],
    user_to_idx: dict[str, int], item_to_idx: dict[str, int],
    game_train: pd.DataFrame, movie_df: pd.DataFrame,
    cold_ids: set[str], movie_counts: pd.Series,
    game_test_relevant: dict[int, set[int]],
) -> dict[str, list[int]]:
    """Cold-start subgroups by movie richness and test-item popularity."""
    game_item_pop, median_game_pop = _popularity_map(game_train, item_to_idx)
    movie_item_pop, median_movie_pop = _popularity_map(movie_df, item_to_idx)
    cold_user_movie_items = _items_by_user(
        movie_df[movie_df["user_id"].isin(cold_ids)], item_to_idx,
    )

    def _movie_niche(uid_norm: str) -> bool:
        items = cold_user_movie_items.get(uid_norm, set())
        if not items:
            return False
        return sum(1 for i in items if movie_item_pop.get(i, 0) <= median_movie_pop) > len(items) / 2

    subgroups: dict[str, list[int]] = {
        "cs_low_movies": [],                    # cold, < 15 movies
        "cs_med_movies": [],                    # cold, 15-49 movies
        "cs_rich_movies": [],                   # cold, >= 50 movies
        "one_shot_unpopular_target_user": [],   # test game ≤ median pop
        "one_shot_popular_target_user": [],     # test game > median pop
        "high_source_unpopular_low_target": [], # >=15 movies, majority niche
        "high_source_popular_low_target": [],   # >=15 movies, mostly popular
    }

    eval_set = set(eval_user_indices)
    for raw_uid, mc in movie_counts.items():
        if raw_uid not in cold_ids:
            continue
        uid_norm = normalize_id(raw_uid)
        uidx = user_to_idx.get(uid_norm)
        if uidx is None or uidx not in eval_set:
            continue

        if mc < 15:
            subgroups["cs_low_movies"].append(uidx)
        elif mc < 50:
            subgroups["cs_med_movies"].append(uidx)
        else:
            subgroups["cs_rich_movies"].append(uidx)

        test_iid = next(iter(game_test_relevant.get(uidx, set())), None)
        if test_iid is not None:
            key = ("one_shot_unpopular_target_user"
                   if game_item_pop.get(test_iid, 0) <= median_game_pop
                   else "one_shot_popular_target_user")
            subgroups[key].append(uidx)

        if mc >= 15:
            key = "high_source_unpopular_low_target" if _movie_niche(uid_norm) \
                else "high_source_popular_low_target"
            subgroups[key].append(uidx)

    return subgroups
