"""Common utilities for the cross-domain benchmark suite.

Protocol: movie → game cross-domain recommendation.
Split: per-user leave-last-out on target-domain (game) interactions.
Metrics: Recall@10, NDCG@10 (full-rank) + sampled HR@10, NDCG@10 (1 pos + 99 neg).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root (benchmark_common.py lives at ml/scripts/benchmarks/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics
from ml.evaluation.evaluator import evaluate_full_rank, evaluate_sampled
from ml.models.id_utils import normalize_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
POSITIVE_THRESHOLD = 4
K = 10  # all metrics @10 only

# ---------------------------------------------------------------------------
# Domain pair paths — Phase 0: only movie_game
# ---------------------------------------------------------------------------
_DOMAIN_PAIR_PATHS: dict[str, tuple[Path, Path]] = {
    "movie_game": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed",
        PROJECT_ROOT / "artifacts",
    ),
}

# Mutable state — set by configure_benchmark()
DATA_DIR = _DOMAIN_PAIR_PATHS["movie_game"][0]
ARTIFACTS_DIR = _DOMAIN_PAIR_PATHS["movie_game"][1]
RESULTS_DIR = ARTIFACTS_DIR / "results"
_benchmark_domain_pair: str = "movie_game"


def configure_benchmark(domain_pair: str = "movie_game") -> None:
    """Point DATA_DIR and artifact dirs at the chosen domain pair."""
    global DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR, _benchmark_domain_pair
    if domain_pair not in _DOMAIN_PAIR_PATHS:
        raise ValueError(f"Unknown domain_pair {domain_pair!r}. Available: {list(_DOMAIN_PAIR_PATHS)}")
    _benchmark_domain_pair = domain_pair
    DATA_DIR, ARTIFACTS_DIR = _DOMAIN_PAIR_PATHS[domain_pair]
    RESULTS_DIR = ARTIFACTS_DIR / "results"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser) -> None:
    """Add --domain-pair, --target, --lesson to an argparse parser."""
    parser.add_argument(
        "--domain-pair",
        choices=list(_DOMAIN_PAIR_PATHS),
        default="movie_game",
    )
    parser.add_argument("--target", default="game", choices=["game", "movie"])
    parser.add_argument("--lesson", type=int, required=True, help="Lesson number for result tagging")


# ═══════════════════════════════════════════════════════════════════════════
# CrossDomainSplit
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrossDomainSplit:
    """All data needed by every benchmark script."""

    # Per-user leave-last-out splits
    game_train: pd.DataFrame
    game_val: pd.DataFrame
    game_test: pd.DataFrame
    movie_train: pd.DataFrame
    movie_val: pd.DataFrame
    movie_test: pd.DataFrame

    # Full domain data (for CDR models that need all source interactions)
    movie_all: pd.DataFrame
    game_all: pd.DataFrame

    # Combined training set: game_train + all movie interactions
    cross_train: pd.DataFrame

    # ID mappings
    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]

    # Item index sets per domain
    game_item_indices: set[int]
    movie_item_indices: set[int]

    # Eval structures: {user_idx: set of relevant item indices}
    game_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_train_seen: dict[int, set[int]] = field(default_factory=dict)
    movie_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    movie_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    movie_train_seen: dict[int, set[int]] = field(default_factory=dict)

    eval_user_indices: list[int] = field(default_factory=list)

    # Subgroup assignments: {subgroup_name: [user_indices]}
    user_subgroups: dict[str, list[int]] = field(default_factory=dict)

    target_domain: str = "game"
    domain_pair: str = "movie_game"
    device: str = "cpu"

    # Dataset metadata for save_result() / plot annotation
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
    def source_all(self) -> pd.DataFrame:
        return self.movie_all if self.target_domain == "game" else self.game_all

    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self.item_to_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Leave-last-out split
# ═══════════════════════════════════════════════════════════════════════════

def _empty_df(template: pd.DataFrame) -> pd.DataFrame:
    """Empty DataFrame with same columns as template."""
    if template is None or len(template) == 0:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "domain", "timestamp"])
    return template.iloc[0:0].copy()


def _leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-last-out split sorted by timestamp.

    n == 1 → 0 train, 1 test  (super cold-start: must rely on source domain)
    n == 2 → 1 train, 1 test  (cold-start)
    n >= 3 → (n-2) train, 1 val, 1 test  (standard)
    """
    if df is None or len(df) == 0:
        empty = _empty_df(df)
        return empty.copy(), empty.copy(), empty.copy()

    train_rows, val_rows, test_rows = [], [], []
    for _, grp in df.groupby("user_id"):
        grp = grp.sort_values("timestamp") if grp["timestamp"].notna().any() else grp
        n = len(grp)
        if n == 1:
            test_rows.append(grp)
        elif n == 2:
            train_rows.append(grp.iloc[:1])
            test_rows.append(grp.iloc[1:])
        else:
            train_rows.append(grp.iloc[:n - 2])
            val_rows.append(grp.iloc[n - 2:n - 1])
            test_rows.append(grp.iloc[n - 1:])

    cols = df.columns
    train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=cols)
    val = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=cols)
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=cols)
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def _build_eval_structures(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build (test_relevant, train_seen) dicts keyed by user index."""
    test_relevant: dict[int, set[int]] = {}
    for _, row in test_df.iterrows():
        uid = user_to_idx.get(normalize_id(row["user_id"]))
        iid = item_to_idx.get(normalize_id(row["item_id"]))
        if uid is not None and iid is not None and row["rating"] >= POSITIVE_THRESHOLD:
            test_relevant.setdefault(uid, set()).add(iid)

    train_seen: dict[int, set[int]] = {}
    for _, row in train_df.iterrows():
        uid = user_to_idx.get(normalize_id(row["user_id"]))
        iid = item_to_idx.get(normalize_id(row["item_id"]))
        if uid is not None and iid is not None:
            train_seen.setdefault(uid, set()).add(iid)

    return test_relevant, train_seen


def load_cross_domain_split(
    domain_pair: str = "movie_game",
    target_domain: str = "game",
    max_eval_users: int = 2000,
    single_domain_item_space: bool = False,
) -> CrossDomainSplit:
    """Load ratings.parquet → leave-last-out splits.

    Args:
        domain_pair: Key in _DOMAIN_PAIR_PATHS.
        target_domain: Domain to evaluate on ("game" or "movie").
        max_eval_users: Cap on eval users (random subsample if exceeded).
        single_domain_item_space: If True, item_to_idx only includes target-domain
            items. Use for single-domain models (MF-BPR, LightGCN, NCF).
            Do NOT use for CDR models that need unified item space.
    """
    configure_benchmark(domain_pair)

    data_path = DATA_DIR / "ratings.parquet"
    logger.info("Loading %s (domain_pair=%s, target=%s)", data_path, domain_pair, target_domain)
    ratings = pd.read_parquet(data_path)

    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df = ratings[ratings["domain"] == "game"].copy()

    logger.info("Loaded %d ratings (movie=%d, game=%d)", len(ratings), len(movie_df), len(game_df))

    # Leave-last-out on both domains
    game_train, game_val, game_test = _leave_last_out(game_df)
    movie_train, movie_val, movie_test = _leave_last_out(movie_df)

    logger.info("Game split: train=%d, val=%d, test=%d", len(game_train), len(game_val), len(game_test))
    logger.info("Movie split: train=%d, val=%d, test=%d", len(movie_train), len(movie_val), len(movie_test))

    # Cross-domain training: target train + all source interactions
    if target_domain == "game":
        cross_train = pd.concat([game_train, movie_df], ignore_index=True)
    else:
        cross_train = pd.concat([movie_train, game_df], ignore_index=True)

    # Build unified user/item ID maps
    all_users = sorted({normalize_id(u) for u in ratings["user_id"].unique()})
    user_to_idx = {u: i for i, u in enumerate(all_users)}

    game_item_ids = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids = {normalize_id(x) for x in movie_df["item_id"].unique()}

    if single_domain_item_space:
        # Only target-domain items in the embedding space
        target_ids = game_item_ids if target_domain == "game" else movie_item_ids
        all_items = sorted(target_ids)
        item_to_idx = {it: i for i, it in enumerate(all_items)}
        idx_to_item = {i: it for it, i in item_to_idx.items()}
        game_item_indices = set(range(len(all_items))) if target_domain == "game" else set()
        movie_item_indices = set(range(len(all_items))) if target_domain == "movie" else set()
    else:
        all_items = sorted({normalize_id(it) for it in ratings["item_id"].unique()})
        item_to_idx = {it: i for i, it in enumerate(all_items)}
        idx_to_item = {i: it for it, i in item_to_idx.items()}
        game_item_indices = {item_to_idx[iid] for iid in game_item_ids if iid in item_to_idx}
        movie_item_indices = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}

    logger.info(
        "ID maps: %d users, %d items (game=%d, movie=%d)",
        len(user_to_idx), len(item_to_idx),
        len(game_item_indices), len(movie_item_indices),
    )

    # Build eval structures
    game_test_relevant, game_train_seen = _build_eval_structures(game_test, game_train, user_to_idx, item_to_idx)
    game_val_relevant, _ = _build_eval_structures(game_val, game_train, user_to_idx, item_to_idx)
    movie_test_relevant, movie_train_seen = _build_eval_structures(movie_test, movie_train, user_to_idx, item_to_idx)
    movie_val_relevant, _ = _build_eval_structures(movie_val, movie_train, user_to_idx, item_to_idx)

    target_test_relevant = game_test_relevant if target_domain == "game" else movie_test_relevant
    eval_user_indices = sorted(target_test_relevant.keys())

    if max_eval_users and len(eval_user_indices) > max_eval_users:
        rng = np.random.RandomState(SEED)
        eval_user_indices = sorted(rng.choice(eval_user_indices, max_eval_users, replace=False))

    logger.info("Eval users: %d (of %d with relevant test items)", len(eval_user_indices), len(target_test_relevant))

    # ── Subgroups ───────────────────────────────────────────────────────────
    user_movie_count = movie_df.groupby(movie_df["user_id"].map(normalize_id)).size().to_dict()
    user_game_count = game_df.groupby(game_df["user_id"].map(normalize_id)).size().to_dict()

    target_train_df = game_train if target_domain == "game" else movie_train
    user_train_size = {
        normalize_id(uid): int(cnt)
        for uid, cnt in target_train_df.groupby("user_id").size().to_dict().items()
    }

    # Item popularity for unpopular detection
    item_pop = {
        item_to_idx[normalize_id(iid)]: int(cnt)
        for iid, cnt in target_train_df.groupby("item_id").size().to_dict().items()
        if normalize_id(iid) in item_to_idx
    }
    median_pop = float(np.median(list(item_pop.values()))) if item_pop else 1.0

    # Per-user target train items
    user_train_items: dict[str, set[int]] = {}
    for uid_raw, grp in target_train_df.groupby("user_id"):
        uid_norm = normalize_id(uid_raw)
        user_train_items[uid_norm] = {
            item_to_idx[normalize_id(iid)]
            for iid in grp["item_id"]
            if normalize_id(iid) in item_to_idx
        }

    def _unpopular(uid_norm: str) -> bool:
        items = user_train_items.get(uid_norm, set())
        if not items:
            return False
        return sum(1 for i in items if item_pop.get(i, 0) <= median_pop) > len(items) / 2

    subgroups: dict[str, list[int]] = {
        "super_cold_users": [],
        "one_shot_target_user": [],
        "one_shot_unpopular_target_user": [],
        "high_source_low_target": [],
        "high_source_unpopular_low_target": [],
        "movie_heavy": [],
        "game_heavy": [],
        "balanced": [],
    }

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    for uid_idx in eval_user_indices:
        uid_str = idx_to_user[uid_idx]
        mc = user_movie_count.get(uid_str, 0)
        gc = user_game_count.get(uid_str, 0)
        target_total = gc if target_domain == "game" else mc
        source_count = mc if target_domain == "game" else gc
        train_size = user_train_size.get(uid_str, 0)

        # Cold-start groups (non-exclusive)
        if train_size == 0 and source_count >= 10:
            subgroups["super_cold_users"].append(uid_idx)
        if train_size == 1 and source_count >= 10:
            if _unpopular(uid_str):
                subgroups["one_shot_unpopular_target_user"].append(uid_idx)
            else:
                subgroups["one_shot_target_user"].append(uid_idx)
        if target_total <= 3 and source_count >= 15:
            if _unpopular(uid_str):
                subgroups["high_source_unpopular_low_target"].append(uid_idx)
            else:
                subgroups["high_source_low_target"].append(uid_idx)

        # Warm-start groups
        if target_total > 3:
            if mc > 2 * gc:
                subgroups["movie_heavy"].append(uid_idx)
            elif gc > 2 * mc:
                subgroups["game_heavy"].append(uid_idx)
            else:
                subgroups["balanced"].append(uid_idx)

    for name, uids in subgroups.items():
        if uids:
            logger.info("  subgroup %-35s: %d users", name, len(uids))

    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Dataset info for save_result() and plot annotation
    dataset_info = {
        "domain_pair": domain_pair,
        "cohort_filter": _cohort_filter_string(domain_pair),
        "n_users": len(user_to_idx),
        "n_movie_interactions": len(movie_df),
        "n_game_interactions": len(game_df),
        "split": f"leave-last-out on {target_domain}s",
    }

    return CrossDomainSplit(
        game_train=game_train,
        game_val=game_val,
        game_test=game_test,
        movie_train=movie_train,
        movie_val=movie_val,
        movie_test=movie_test,
        movie_all=movie_df,
        game_all=game_df,
        cross_train=cross_train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        game_item_indices=game_item_indices,
        movie_item_indices=movie_item_indices,
        game_test_relevant=game_test_relevant,
        game_val_relevant=game_val_relevant,
        game_train_seen=game_train_seen,
        movie_test_relevant=movie_test_relevant,
        movie_val_relevant=movie_val_relevant,
        movie_train_seen=movie_train_seen,
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain=target_domain,
        domain_pair=domain_pair,
        device=device,
        dataset_info=dataset_info,
    )


def _cohort_filter_string(domain_pair: str) -> str:
    """Human-readable cohort filter description for dataset_info."""
    # Phase 0: only movie_game with k-core >= 10
    return "k-core >= 10 (both movies and games)"


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation entry point — delegates to evaluator.py
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_cross_domain(
    model_name: str,
    predict_fn,
    data: CrossDomainSplit,
    eval_user_override: list[int] | None = None,
) -> dict[str, Any]:
    """Single entry point: full-rank + sampled evaluation. All bench scripts call this."""
    full_rank = evaluate_full_rank(model_name, predict_fn, data, eval_user_override)
    sampled = evaluate_sampled(model_name, predict_fn, data)
    return {**full_rank, **sampled}


# ═══════════════════════════════════════════════════════════════════════════
# Leakage check
# ═══════════════════════════════════════════════════════════════════════════

def verify_no_leakage(data: CrossDomainSplit) -> None:
    """Raise AssertionError if train/test/val sets leak."""
    train_pairs = set(zip(data.target_train["user_id"], data.target_train["item_id"]))
    test_pairs = set(zip(data.target_test["user_id"], data.target_test["item_id"]))
    assert not (train_pairs & test_pairs), "LEAK: test items found in train!"
    logger.info("Leakage check passed")


# ═══════════════════════════════════════════════════════════════════════════
# Artifact saving
# ═══════════════════════════════════════════════════════════════════════════

def save_result(
    algo: str,
    metrics: dict[str, Any],
    dataset_info: dict[str, Any],
    lesson: int,
    train_time: float = 0.0,
    description: str = "",
) -> None:
    """Save benchmark results to artifacts/<domain-pair>/results/<algo>_lesson<N>.json.

    The JSON includes the dataset_info block for plot annotation.
    """
    from datetime import datetime as _dt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{algo.lower()}_lesson{lesson}.json"
    path = RESULTS_DIR / filename

    entry = {
        "model": algo,
        "lesson": lesson,
        "description": description,
        "time_completed": _dt.now().isoformat(),
        "train_time_s": train_time,
        "dataset_info": dataset_info,
        **metrics,
    }

    path.write_text(json.dumps(entry, indent=2, default=str))
    logger.info("Saved %s results to %s", algo, path)
