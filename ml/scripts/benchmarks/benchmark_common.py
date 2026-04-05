"""Common utilities for the cross-domain benchmark suite.

Protocol (pair-dependent)
-------------------------
* ``movie_game`` (data: ``ml/data/amazon_2023/processed/``, artifacts: ``artifacts/``):
  default target = **game**; source = **movie**; optional ``--target movie``.
* ``movie_book`` (data: ``.../processed_movie_book/``, artifacts: ``artifacts_movie_book/``):
  default target = **movie**; source = **book**; optional ``--target book``.

Split: per-user leave-last-out on **target-domain** interactions only.

Training:
    target-only baselines → ``target_train``
    cross-domain models   → ``target_train`` ∪ ``source_all`` (= ``cross_train``)

Evaluation:
    all models on the same held-out **target** test users/items; non-target
    items masked to -inf.

Metrics: Recall@K, NDCG@K, HitRate@K  (K ∈ {5, 10, 20, 50})
Subgroups: movie/game/book-heavy, balanced, cold-start cohorts.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Resolve project root  (benchmark_common.py lives at ml/scripts/benchmarks/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Per cross-domain pair: processed parquet dir + separate artifact root for side-by-side comparison.
_DOMAIN_PAIR_PATHS: dict[str, tuple[Path, Path]] = {
    "movie_game": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed",
        PROJECT_ROOT / "artifacts",
    ),
    "movie_book": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed_movie_book",
        PROJECT_ROOT / "artifacts_movie_book",
    ),
    "movie_game_transfer_loose": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed_transfer_loose",
        PROJECT_ROOT / "artifacts_transfer_loose",
    ),
    "movie_game_transfer_strict": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed_transfer_strict",
        PROJECT_ROOT / "artifacts_transfer_strict",
    ),
    "movie_game_transfer_loose_filtered": (
        PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed_transfer_loose_filtered",
        PROJECT_ROOT / "artifacts_transfer_loose_filtered",
    ),
}

# Use for ``--domain-pair`` in all ``bench_*.py`` CLIs (keeps paths in sync).
BENCHMARK_DOMAIN_PAIR_CHOICES: tuple[str, ...] = tuple(_DOMAIN_PAIR_PATHS.keys())

# Mutable — set by configure_benchmark() / load_cross_domain_split()
DATA_DIR = _DOMAIN_PAIR_PATHS["movie_game"][0]
ARTIFACTS_DIR = _DOMAIN_PAIR_PATHS["movie_game"][1]
DEMO_DIR = ARTIFACTS_DIR / "demo"
RESULTS_DIR = ARTIFACTS_DIR / "results"
_benchmark_domain_pair: str = "movie_game"


def load_item_records_for_sbert(data_dir: Path) -> list[dict]:
    """Load ``movies`` + ``games`` or ``movies`` + ``books`` parquet rows as dicts for SBERT."""
    movies_path = data_dir / "movies.parquet"
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing {movies_path}")
    movies = pd.read_parquet(movies_path)
    games_path = data_dir / "games.parquet"
    books_path = data_dir / "books.parquet"
    if games_path.exists():
        secondary = pd.read_parquet(games_path)
    elif books_path.exists():
        secondary = pd.read_parquet(books_path)
    else:
        raise FileNotFoundError(f"Need {games_path} or {books_path} next to movies.parquet")
    combined = pd.concat([movies, secondary], ignore_index=True)
    return combined.to_dict("records")


def validate_cross_domain_cli_args(
    domain_pair: str,
    target_domain: str | None,
) -> None:
    """Exit with message if ``target_domain`` is incompatible with ``domain_pair``."""
    if target_domain is None:
        return
    valid: dict[str, tuple[str, ...]] = {
        "movie_game": ("game", "movie"),
        "movie_book": ("movie", "book"),
        "movie_game_transfer_loose": ("game", "movie"),
        "movie_game_transfer_strict": ("game", "movie"),
        "movie_game_transfer_loose_filtered": ("game", "movie"),
    }
    allowed = valid.get(domain_pair, ("game", "movie"))
    if target_domain not in allowed:
        raise SystemExit(
            f"Invalid target domain {target_domain!r} for domain_pair={domain_pair!r}; "
            f"expected one of {allowed}"
        )


def configure_benchmark(domain_pair: Literal["movie_game", "movie_book"] = "movie_game") -> None:
    """Point DATA_DIR and artifact dirs at the chosen cross-domain pair."""
    global DATA_DIR, ARTIFACTS_DIR, DEMO_DIR, RESULTS_DIR, _benchmark_domain_pair
    if domain_pair not in _DOMAIN_PAIR_PATHS:
        raise ValueError(f"Unknown domain_pair {domain_pair!r}")
    _benchmark_domain_pair = domain_pair
    DATA_DIR, ARTIFACTS_DIR = _DOMAIN_PAIR_PATHS[domain_pair]
    DEMO_DIR = ARTIFACTS_DIR / "demo"
    RESULTS_DIR = ARTIFACTS_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.data_splitter import leave_last_out_split
from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics
from ml.models.id_utils import normalize_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
POSITIVE_THRESHOLD = 3.5
K_VALUES = [5, 10, 20, 50, 99]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def monitor_training_progress(start_time: float, model_name: str, total_epochs: int = None) -> None:
    """Log training progress to show model is not stuck."""
    elapsed = time.time() - start_time
    if total_epochs:
        # Estimate progress if we know total epochs
        avg_epoch_time = elapsed / 10  # Rough estimate after 10 epochs
        estimated_total = avg_epoch_time * total_epochs
        progress_pct = min(100, (elapsed / estimated_total) * 100)
        logger.info(f"{model_name} training progress: ~{progress_pct:.1f}% complete ({elapsed:.1f}s elapsed)")
    else:
        # Just log elapsed time for models without clear epoch count
        logger.info(f"{model_name} training in progress... ({elapsed:.1f}s elapsed)")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-domain split
# ═══════════════════════════════════════════════════════════════════════════

def _empty_ratings_like(template: pd.DataFrame) -> pd.DataFrame:
    if template is None or len(template) == 0:
        return pd.DataFrame(
            columns=["user_id", "item_id", "rating", "domain", "timestamp", "review_text"]
        )
    return template.iloc[0:0].copy()


@dataclass
class CrossDomainSplit:
    """All data needed by every benchmark script.

    ``domain_pair`` is ``movie_game`` or ``movie_book``. Unused domain frames
    (e.g. games when pair is movie_book) are empty DataFrames with matching columns.
    """

    # ── Per-user leave-last-out splits ───────────────────────────────────
    game_train: pd.DataFrame
    game_val: pd.DataFrame
    game_test: pd.DataFrame

    movie_train: pd.DataFrame
    movie_val: pd.DataFrame
    movie_test: pd.DataFrame

    book_train: pd.DataFrame
    book_val: pd.DataFrame
    book_test: pd.DataFrame

    # ── Full domain data ─────────────────────────────────────────────────
    movie_all: pd.DataFrame
    game_all: pd.DataFrame
    book_all: pd.DataFrame

    cross_train: pd.DataFrame

    user_to_idx: dict[str, int]
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]

    game_item_indices: set[int]
    movie_item_indices: set[int]
    book_item_indices: set[int]

    game_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    game_train_seen: dict[int, set[int]] = field(default_factory=dict)

    movie_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    movie_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    movie_train_seen: dict[int, set[int]] = field(default_factory=dict)

    book_test_relevant: dict[int, set[int]] = field(default_factory=dict)
    book_val_relevant: dict[int, set[int]] = field(default_factory=dict)
    book_train_seen: dict[int, set[int]] = field(default_factory=dict)

    eval_user_indices: list[int] = field(default_factory=list)

    # movie_heavy / game_heavy / book_heavy / balanced — see load_cross_domain_split
    user_subgroups: dict[str, list[int]] = field(default_factory=dict)

    target_domain: str = "game"
    domain_pair: str = "movie_game"

    device: str = "cpu"

    @property
    def target_train(self) -> pd.DataFrame:
        if self.target_domain == "game":
            return self.game_train
        if self.target_domain == "book":
            return self.book_train
        return self.movie_train

    @property
    def target_val(self) -> pd.DataFrame:
        if self.target_domain == "game":
            return self.game_val
        if self.target_domain == "book":
            return self.book_val
        return self.movie_val

    @property
    def target_test(self) -> pd.DataFrame:
        if self.target_domain == "game":
            return self.game_test
        if self.target_domain == "book":
            return self.book_test
        return self.movie_test

    @property
    def target_item_indices(self) -> set[int]:
        if self.target_domain == "game":
            return self.game_item_indices
        if self.target_domain == "book":
            return self.book_item_indices
        return self.movie_item_indices

    @property
    def target_test_relevant(self) -> dict[int, set[int]]:
        if self.target_domain == "game":
            return self.game_test_relevant
        if self.target_domain == "book":
            return self.book_test_relevant
        return self.movie_test_relevant

    @property
    def target_val_relevant(self) -> dict[int, set[int]]:
        if self.target_domain == "game":
            return self.game_val_relevant
        if self.target_domain == "book":
            return self.book_val_relevant
        return self.movie_val_relevant

    @property
    def target_train_seen(self) -> dict[int, set[int]]:
        if self.target_domain == "game":
            return self.game_train_seen
        if self.target_domain == "book":
            return self.book_train_seen
        return self.movie_train_seen

    @property
    def _base_pair(self) -> str:
        """Resolve transfer-tier variants to their base domain pair."""
        if self.domain_pair.startswith("movie_game"):
            return "movie_game"
        return "movie_book"

    @property
    def source_all(self) -> pd.DataFrame:
        """All source-domain interactions for cross-domain training."""
        if self._base_pair == "movie_game":
            return self.movie_all if self.target_domain == "game" else self.game_all
        # movie_book
        return self.book_all if self.target_domain == "movie" else self.movie_all

    @property
    def cdr_source_domain(self) -> str:
        """``ratings['domain']`` label for recbole *source* interactions (see ``fit_cdr``)."""
        if self._base_pair == "movie_game":
            return "movie" if self.target_domain == "game" else "game"
        return "book" if self.target_domain == "movie" else "movie"

    @property
    def cdr_target_domain(self) -> str:
        """``ratings['domain']`` label for recbole *target* interactions."""
        return self.target_domain

    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self.item_to_idx)


def _leave_last_out(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-last-out split sorted by timestamp.

    Threshold rules (inclusive of cold-start users):
      n == 1 → 0 train, 1 test  (super cold-start: model must rely entirely on source domain)
      n == 2 → 1 train, 1 test  (cold-start: model must rely on source domain)
      n >= 3 → (n-2) train, 1 val, 1 test  (standard leave-last-out)

    Including n=1 users enables the super_cold_start subgroup: users with zero
    target-domain training items who can only be served via cross-domain transfer.
    """
    if df is None or len(df) == 0:
        empty = _empty_ratings_like(df)
        return empty.copy(), empty.copy(), empty.copy()

    train_rows, val_rows, test_rows = [], [], []
    ts_col = "timestamp"
    for _, grp in df.groupby("user_id"):
        grp = grp.sort_values(ts_col) if grp[ts_col].notna().any() else grp
        n = len(grp)
        if n == 1:
            test_rows.append(grp)            # super cold-start → 0 train, 1 test
        elif n == 2:
            train_rows.append(grp.iloc[:1])  # cold-start: 1 train, 1 test
            test_rows.append(grp.iloc[1:])
        else:
            train_rows.append(grp.iloc[: n - 2])
            val_rows.append(grp.iloc[n - 2 : n - 1])
            test_rows.append(grp.iloc[n - 1 :])
    cols = df.columns
    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=cols)
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=cols)
    return train, val, test


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
    max_eval_users: int = 2000,
    max_interactions: int | None = None,
    target_domain: str | None = None,
    overlap_min_interactions: int = 0,
    domain_pair: str = "movie_game",
    single_domain_item_space: bool = False,
) -> CrossDomainSplit:
    """Load ``ratings.parquet`` for the chosen pair → leave-last-out splits.

    Args:
        target_domain: Domain to evaluate on. If ``None``:
            ``movie_game`` → ``game``; ``movie_book`` → ``movie`` (books = source,
            movies = target — smaller catalog for full-rank eval, rich book source).
        domain_pair: ``movie_game`` | ``movie_book`` | ``movie_game_transfer_loose``
            | ``movie_game_transfer_strict``.
        single_domain_item_space: If ``True``, ``item_to_idx`` / ``num_items`` include
            **only** target-domain items (e.g. games when ``target_domain=game``).
            User ids are unchanged (same overlap users). Use for pure target-domain
            baselines (LightGCN, MF, NCF, SBERT). **Do not** set for CDR models
            (EMCDR, Bi-TGCF, …) — they need a unified item id space for source+target.
    """
    configure_benchmark(domain_pair)

    # Resolve the base pair for domain logic (transfer variants are movie_game subsets)
    base_pair = "movie_game" if domain_pair.startswith("movie_game") else "movie_book"

    if target_domain is None:
        target_domain = "game" if base_pair == "movie_game" else "movie"

    valid_targets: dict[str, tuple[str, ...]] = {
        "movie_game": ("game", "movie"),
        "movie_book": ("movie", "book"),
    }
    allowed = valid_targets[base_pair]
    if target_domain not in allowed:
        raise ValueError(
            f"target_domain must be one of {allowed} for domain_pair={domain_pair!r}, "
            f"got {target_domain!r}"
        )

    data_path = DATA_DIR / "ratings.parquet"
    logger.info("Loading %s (domain_pair=%s target=%s)", data_path, domain_pair, target_domain)
    ratings = pd.read_parquet(data_path)

    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df = ratings[ratings["domain"] == "game"].copy() if base_pair == "movie_game" else _empty_ratings_like(movie_df)
    book_df = ratings[ratings["domain"] == "book"].copy() if base_pair == "movie_book" else _empty_ratings_like(movie_df)

    secondary = "game" if base_pair == "movie_game" else "book"
    secondary_df = game_df if secondary == "game" else book_df

    logger.info(
        "Loaded %d ratings (movie=%d, %s=%d)",
        len(ratings),
        len(movie_df),
        secondary,
        len(secondary_df),
    )

    if max_interactions is not None:
        rng = np.random.RandomState(SEED)

        def _sample_by_users(df: pd.DataFrame, domain: str) -> pd.DataFrame:
            if len(df) <= max_interactions:
                return df
            users = df["user_id"].unique()
            target_users = max(1, max_interactions // 10)
            if target_users >= len(users):
                return df
            sampled = rng.choice(users, size=target_users, replace=False)
            out = df[df["user_id"].isin(sampled)].copy()
            logger.info("Sampled %s to %d interactions (%d users)", domain, len(out), out["user_id"].nunique())
            return out

        movie_df = _sample_by_users(movie_df, "movie")
        if secondary == "game":
            game_df = _sample_by_users(game_df, "game")
        else:
            book_df = _sample_by_users(book_df, "book")
            secondary_df = book_df

    game_train, game_val, game_test = _leave_last_out(game_df)
    movie_train, movie_val, movie_test = _leave_last_out(movie_df)
    book_train, book_val, book_test = _leave_last_out(book_df)

    logger.info(
        "Game split: train=%d, val=%d, test=%d",
        len(game_train),
        len(game_val),
        len(game_test),
    )
    logger.info(
        "Movie split: train=%d, val=%d, test=%d",
        len(movie_train),
        len(movie_val),
        len(movie_test),
    )
    logger.info(
        "Book split: train=%d, val=%d, test=%d",
        len(book_train),
        len(book_val),
        len(book_test),
    )

    if target_domain == "game":
        cross_train = pd.concat([game_train, movie_df], ignore_index=True)
    elif target_domain == "movie":
        cross_train = pd.concat([movie_train, secondary_df], ignore_index=True)
    else:
        cross_train = pd.concat([book_train, movie_df], ignore_index=True)
    logger.info("Cross-domain training set: %d interactions", len(cross_train))

    ratings_used = pd.concat([movie_df, secondary_df], ignore_index=True)
    all_users = sorted({normalize_id(u) for u in ratings_used["user_id"].unique()})
    user_to_idx = {u: i for i, u in enumerate(all_users)}

    game_item_ids = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids = {normalize_id(x) for x in movie_df["item_id"].unique()}
    book_item_ids = {normalize_id(x) for x in book_df["item_id"].unique()}

    if single_domain_item_space:
        if target_domain == "game":
            target_items_source = game_df
        elif target_domain == "movie":
            target_items_source = movie_df
        else:
            target_items_source = book_df
        all_items = sorted(
            {normalize_id(it) for it in target_items_source["item_id"].unique()}
        )
        item_to_idx = {it: i for i, it in enumerate(all_items)}
        idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}
        n_t = len(all_items)
        game_item_indices = set(range(n_t)) if target_domain == "game" else set()
        movie_item_indices = set(range(n_t)) if target_domain == "movie" else set()
        book_item_indices = set(range(n_t)) if target_domain == "book" else set()
        logger.info(
            "Single-domain item space: %d users, %d %s items only (embeddings exclude other domains)",
            len(user_to_idx),
            n_t,
            target_domain,
        )
    else:
        all_items = sorted({normalize_id(it) for it in ratings_used["item_id"].unique()})
        item_to_idx = {it: i for i, it in enumerate(all_items)}
        idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}
        game_item_indices = {item_to_idx[iid] for iid in game_item_ids if iid in item_to_idx}
        movie_item_indices = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}
        book_item_indices = {item_to_idx[iid] for iid in book_item_ids if iid in item_to_idx}
        logger.info(
            "Unified maps: %d users, %d items (game=%d, movie=%d, book=%d)",
            len(user_to_idx),
            len(item_to_idx),
            len(game_item_indices),
            len(movie_item_indices),
            len(book_item_indices),
        )

    game_test_relevant, game_train_seen = _build_eval_structures(
        game_test, game_train, user_to_idx, item_to_idx
    )
    game_val_relevant, _ = _build_eval_structures(game_val, game_train, user_to_idx, item_to_idx)
    movie_test_relevant, movie_train_seen = _build_eval_structures(
        movie_test, movie_train, user_to_idx, item_to_idx
    )
    movie_val_relevant, _ = _build_eval_structures(movie_val, movie_train, user_to_idx, item_to_idx)
    book_test_relevant, book_train_seen = _build_eval_structures(
        book_test, book_train, user_to_idx, item_to_idx
    )
    book_val_relevant, _ = _build_eval_structures(book_val, book_train, user_to_idx, item_to_idx)

    target_test_relevant = {
        "game": game_test_relevant,
        "movie": movie_test_relevant,
        "book": book_test_relevant,
    }[target_domain]
    eval_user_indices = sorted(target_test_relevant.keys())

    _mv = movie_df.copy()
    _mv["_u"] = _mv["user_id"].map(normalize_id)
    user_movie_count: dict[str, int] = _mv.groupby("_u").size().to_dict()
    _gv = game_df.copy()
    _gv["_u"] = _gv["user_id"].map(normalize_id)
    user_game_count: dict[str, int] = _gv.groupby("_u").size().to_dict()
    _bv = book_df.copy()
    _bv["_u"] = _bv["user_id"].map(normalize_id)
    user_book_count: dict[str, int] = _bv.groupby("_u").size().to_dict()

    # Overlap filter: both domains in the pair must have ≥N interactions
    if overlap_min_interactions > 0:
        idx_to_user_pre = {v: k for k, v in user_to_idx.items()}
        before = len(eval_user_indices)
        eval_user_indices = []
        for uid_idx in sorted(target_test_relevant.keys()):
            uid_str = idx_to_user_pre.get(uid_idx)
            if not uid_str:
                continue
            mc = user_movie_count.get(uid_str, 0)
            if base_pair == "movie_game":
                gc = user_game_count.get(uid_str, 0)
                if mc >= overlap_min_interactions and gc >= overlap_min_interactions:
                    eval_user_indices.append(uid_idx)
            else:
                bc = user_book_count.get(uid_str, 0)
                if mc >= overlap_min_interactions and bc >= overlap_min_interactions:
                    eval_user_indices.append(uid_idx)
        logger.info(
            "Overlap filter (both domains ≥%d): %d → %d eval users",
            overlap_min_interactions,
            before,
            len(eval_user_indices),
        )

    if max_eval_users is not None and len(eval_user_indices) > max_eval_users:
        rng = np.random.RandomState(SEED)
        eval_user_indices = sorted(rng.choice(eval_user_indices, max_eval_users, replace=False))
    logger.info(
        "Eval users (%s target): %d (of %d with relevant test items)",
        target_domain,
        len(eval_user_indices),
        len(target_test_relevant),
    )

    # ── Subgroup support data ────────────────────────────────────────────
    # Per-user target train size (number of items in target train set)
    target_train_df = {"game": game_train, "movie": movie_train, "book": book_train}[target_domain]
    _train_size_raw = target_train_df.groupby("user_id").size().to_dict()
    user_train_size: dict[str, int] = {normalize_id(uid): int(cnt) for uid, cnt in _train_size_raw.items()}

    # Item popularity: interaction count in target train (for unpopular detection)
    _item_pop_raw = target_train_df.groupby("item_id").size().to_dict()
    item_train_pop: dict[int, int] = {
        item_to_idx[normalize_id(iid)]: int(cnt)
        for iid, cnt in _item_pop_raw.items()
        if normalize_id(iid) in item_to_idx
    }
    median_pop = float(np.median(list(item_train_pop.values()))) if item_train_pop else 1.0

    # Per-user: set of their target train item indices (for unpopular check)
    user_train_items: dict[str, set[int]] = {}
    for _uid_raw, _grp in target_train_df.groupby("user_id"):
        _uid_norm = normalize_id(_uid_raw)
        user_train_items[_uid_norm] = {
            item_to_idx[normalize_id(iid)]
            for iid in _grp["item_id"]
            if normalize_id(iid) in item_to_idx
        }

    def _unpopular_concentrated(uid_norm: str) -> bool:
        """True if majority of the user's target train items are ≤ median popularity."""
        items = user_train_items.get(uid_norm, set())
        if not items:
            return False
        unpop = sum(1 for i in items if item_train_pop.get(i, 0) <= median_pop)
        return unpop > len(items) / 2

    subgroups: dict[str, list[int]] = {
        # Cold-start groups (users with very few target interactions but rich source)
        "super_cold_users": [],               # train_size=0, source>=10  (zero target train items)
        "one_shot_target_user": [],           # train_size=1, source>=10, popular target item
        "one_shot_unpopular_target_user": [], # train_size=1, source>=10, unpopular target item
        "high_source_low_target": [],         # target_total<=3, source>=15, popular
        "high_source_unpopular_low_target": [], # target_total<=3, source>=15, unpopular
        # Warm-start groups
        "movie_heavy": [],
        "balanced": [],
        "game_heavy": [],
        "book_heavy": [],
        "sparse_target": [],                  # few target interactions, insufficient source
    }

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    for uid_idx in eval_user_indices:
        uid_str = idx_to_user[uid_idx]
        mc = user_movie_count.get(uid_str, 0)
        gc = user_game_count.get(uid_str, 0)
        bc = user_book_count.get(uid_str, 0)

        if target_domain == "game":
            target_total, source_count = gc, mc
        elif target_domain == "movie":
            source_count = gc if base_pair == "movie_game" else bc
            target_total = mc
        else:
            target_total, source_count = bc, mc

        train_size = user_train_size.get(uid_str, 0)

        # ── Cold-start groups (non-exclusive: a user can appear in multiple) ──
        if train_size == 0 and source_count >= 10:
            subgroups["super_cold_users"].append(uid_idx)

        if train_size == 1 and source_count >= 10:
            if _unpopular_concentrated(uid_str):
                subgroups["one_shot_unpopular_target_user"].append(uid_idx)
            else:
                subgroups["one_shot_target_user"].append(uid_idx)

        if target_total <= 3 and source_count >= 15:
            if _unpopular_concentrated(uid_str):
                subgroups["high_source_unpopular_low_target"].append(uid_idx)
            else:
                subgroups["high_source_low_target"].append(uid_idx)

        # ── Warm-start groups (target_total > 3, or insufficient source) ──────
        if target_total <= 3 and source_count < 10:
            subgroups["sparse_target"].append(uid_idx)
        elif target_total > 3:
            if base_pair == "movie_game":
                if mc > 2 * gc:
                    subgroups["movie_heavy"].append(uid_idx)
                elif gc > 2 * mc:
                    subgroups["game_heavy"].append(uid_idx)
                else:
                    subgroups["balanced"].append(uid_idx)
            else:
                if mc > 2 * bc:
                    subgroups["movie_heavy"].append(uid_idx)
                elif bc > 2 * mc:
                    subgroups["book_heavy"].append(uid_idx)
                else:
                    subgroups["balanced"].append(uid_idx)

    for name, uids in subgroups.items():
        logger.info("  subgroup %-30s : %d users", name, len(uids))

    device = (
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    return CrossDomainSplit(
        game_train=game_train,
        game_val=game_val,
        game_test=game_test,
        movie_train=movie_train,
        movie_val=movie_val,
        movie_test=movie_test,
        book_train=book_train,
        book_val=book_val,
        book_test=book_test,
        movie_all=movie_df,
        game_all=game_df,
        book_all=book_df,
        cross_train=cross_train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        game_item_indices=game_item_indices,
        movie_item_indices=movie_item_indices,
        book_item_indices=book_item_indices,
        game_test_relevant=game_test_relevant,
        game_val_relevant=game_val_relevant,
        game_train_seen=game_train_seen,
        movie_test_relevant=movie_test_relevant,
        movie_val_relevant=movie_val_relevant,
        movie_train_seen=movie_train_seen,
        book_test_relevant=book_test_relevant,
        book_val_relevant=book_val_relevant,
        book_train_seen=book_train_seen,
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain=target_domain,
        domain_pair=domain_pair,
        device=device,
    )


def load_user_split_cold_start_split(
    cold_start_ratio: float = 0.2,
    min_game_interactions: int = 2,
    min_movie_interactions: int = 5,
    domain_pair: str = "movie_game_transfer_loose",
    seed: int = SEED,
) -> CrossDomainSplit:
    """User-split cold-start evaluation for proving CDR advantage.

    Splits users into warm (game in training) and cold-start (no game in
    training) groups. All test items come from cold-start users' leave-last-out
    game interaction, which is guaranteed to exist in the item vocabulary
    (warm users have interacted with the same games).

    Design:
      Warm users   (1 - cold_start_ratio):
        ALL their game interactions → game_train
        ALL their movie interactions → movie_train / cross_train

      Cold-start eval users (cold_start_ratio):
        ZERO game interactions in training (hidden completely)
        ALL their movie interactions → cross_train only (not game_train)
        Test item: their chronologically LAST game interaction (leave-last-out)

    This means:
      - LightGCN: trained on warm users' game graph; cold-start users are
        invisible → falls back to popularity-biased scores
      - CDR models: cold-start users get cross-domain transfer from their
        rich movie history → mapping(source_emb) or SBERT profile

    Args:
        cold_start_ratio: Fraction of users held out as cold-start eval users.
        min_game_interactions: Minimum game interactions a user must have to
            be eligible (ensures test item + ≥1 other game exists).
        min_movie_interactions: Minimum movie interactions for CDR signal.
        domain_pair: Dataset to use (default: movie_game_transfer_loose).
        seed: Random seed for reproducible user split.
    """
    if not domain_pair.startswith("movie_game"):
        raise ValueError("load_user_split_cold_start_split only supports movie_game pairs")

    configure_benchmark(domain_pair)

    data_path = DATA_DIR / "ratings.parquet"
    logger.info(
        "User-split cold-start: loading %s (cold_ratio=%.2f, min_game=%d, min_movie=%d)",
        data_path, cold_start_ratio, min_game_interactions, min_movie_interactions,
    )
    ratings = pd.read_parquet(data_path)

    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df  = ratings[ratings["domain"] == "game"].copy()

    # ── Eligible users: enough game AND movie interactions ────────────────
    game_counts  = game_df.groupby("user_id").size()
    movie_counts = movie_df.groupby("user_id").size()

    eligible = set(
        game_counts[game_counts >= min_game_interactions].index
    ) & set(
        movie_counts[movie_counts >= min_movie_interactions].index
    )
    logger.info("Eligible users (game>=%d, movie>=%d): %d", min_game_interactions, min_movie_interactions, len(eligible))

    # ── Random user split ─────────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    eligible_sorted = sorted(eligible)
    n_cold = max(1, int(len(eligible_sorted) * cold_start_ratio))
    cold_ids = set(rng.choice(eligible_sorted, size=n_cold, replace=False).tolist())
    warm_ids = set(eligible_sorted) - cold_ids
    logger.info("Split: %d warm users, %d cold-start eval users", len(warm_ids), len(cold_ids))

    # ── game_train: ALL game interactions from warm users ─────────────────
    game_train = game_df[game_df["user_id"].isin(warm_ids)].copy()

    # ── Test set: leave-last-out from cold-start users' game history ──────
    cold_game = game_df[game_df["user_id"].isin(cold_ids)].copy()
    cold_game_sorted = cold_game.sort_values("timestamp")
    # Last game interaction per cold user = test item
    game_test = cold_game_sorted.groupby("user_id").last().reset_index()
    # Normalize to standard dataframe format (groupby.last loses some cols sometimes)
    game_test = cold_game_sorted.groupby("user_id").tail(1).copy()
    logger.info("Test set: %d interactions (1 per cold-start user)", len(game_test))

    # ── Unified user/item maps ────────────────────────────────────────────
    all_users = sorted({normalize_id(u) for u in pd.concat([movie_df, game_df])["user_id"].unique()})
    all_items = sorted({normalize_id(i) for i in pd.concat([movie_df, game_df])["item_id"].unique()})
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {it: i for i, it in enumerate(all_items)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    game_item_ids    = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids   = {normalize_id(x) for x in movie_df["item_id"].unique()}
    game_item_indices  = {item_to_idx[iid] for iid in game_item_ids  if iid in item_to_idx}
    movie_item_indices = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}

    logger.info(
        "Maps: %d users, %d items (game=%d, movie=%d)",
        len(user_to_idx), len(item_to_idx),
        len(game_item_indices), len(movie_item_indices),
    )

    # ── Eval structures ───────────────────────────────────────────────────
    # cold-start users have no game_train rows → train_seen is empty for them
    game_test_relevant, game_train_seen = _build_eval_structures(
        game_test, game_train, user_to_idx, item_to_idx,
    )
    eval_user_indices = sorted(game_test_relevant.keys())

    # Filter: keep only cold-start users whose test item is known to warm users
    # (items only appearing in cold-start users' histories can never be recommended)
    warm_item_ids = {normalize_id(x) for x in game_train["item_id"].unique()}
    before = len(eval_user_indices)
    eval_user_indices = [
        uid for uid in eval_user_indices
        if all(idx_to_item.get(iid) in warm_item_ids
               for iid in game_test_relevant.get(uid, set()))
    ]
    n_dropped = before - len(eval_user_indices)
    if n_dropped:
        logger.info(
            "Dropped %d cold-start users whose test item is unique to them "
            "(not seen by warm users) — %d remain",
            n_dropped, len(eval_user_indices),
        )
    else:
        logger.info("All test items seen by warm users — no new-item problem")

    # cross_train for CDR: all movies + warm-user game interactions
    cross_train = pd.concat([movie_df, game_train], ignore_index=True)
    logger.info(
        "cross_train: %d (movie=%d + warm_game=%d); cold-start users contribute movie only",
        len(cross_train), len(movie_df), len(game_train),
    )

    # ── Subgroups by movie richness ───────────────────────────────────────
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    subgroups: dict[str, list[int]] = {"cold_start_eval": list(eval_user_indices)}
    mc_norm = {normalize_id(u): int(c) for u, c in movie_counts.items()}
    for uid_idx in eval_user_indices:
        uid_str = idx_to_user.get(uid_idx, "")
        mc = mc_norm.get(uid_str, 0)
        bucket = (
            "cs_low_movies"  if mc < 15
            else "cs_med_movies"  if mc < 50
            else "cs_rich_movies"
        )
        subgroups.setdefault(bucket, []).append(uid_idx)

    device = (
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Eval users (cold-start): %d", len(eval_user_indices))

    empty_game = _empty_ratings_like(game_df)
    return CrossDomainSplit(
        game_train=game_train,
        game_val=empty_game,
        game_test=game_test,
        movie_train=movie_df,
        movie_val=empty_game,
        movie_test=empty_game,
        book_train=empty_game,
        book_val=empty_game,
        book_test=empty_game,
        movie_all=movie_df,
        game_all=game_df,
        book_all=empty_game,
        cross_train=cross_train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        game_item_indices=game_item_indices,
        movie_item_indices=movie_item_indices,
        book_item_indices=set(),
        game_test_relevant=game_test_relevant,
        game_val_relevant={},
        game_train_seen=game_train_seen,
        movie_test_relevant={},
        movie_val_relevant={},
        movie_train_seen={},
        book_test_relevant={},
        book_val_relevant={},
        book_train_seen={},
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain="game",
        domain_pair=domain_pair,
        device=device,
    )


def load_temporal_cold_start_split(
    cutoff_date: str,
    min_movie_interactions: int = 5,
    domain_pair: str = "movie_game_transfer_loose",
    first_post_t_test_only: bool = False,
) -> CrossDomainSplit:
    """Temporal cold-start: new gamers (zero target history before cutoff T).

    Train:
        - All movie interactions (all time)
        - Game interactions with timestamp strictly before ``cutoff_date``

    Test population:
        Users with **no** game interactions before T, at least one game interaction
        on or after T, and at least ``min_movie_interactions`` movie ratings.

    Test labels:
        All game interactions on or after T for those users (rating ≥ threshold),
        unless ``first_post_t_test_only=True`` (then chronologically first post-T
        game only — closer to leave-one-out).

    LightGCN sees only ``game_train`` (before-T games); cold users have no game
    edges → popularity-like scores. CDR models use ``cross_train`` (movies +
    before-T games).

    Args:
        cutoff_date: ISO date ``YYYY-MM-DD``; games before 00:00:00 of this date
            are training; games from this instant onward are test (for cold users).
        min_movie_interactions: Minimum movie ratings required for a test user.
        domain_pair: Processed parquet bundle (movie_game transfer variants).
        first_post_t_test_only: If True, one test game per user (first after T).
    """
    if not domain_pair.startswith("movie_game"):
        raise ValueError("load_temporal_cold_start_split only supports movie_game pairs")

    configure_benchmark(domain_pair)

    data_path = DATA_DIR / "ratings.parquet"
    logger.info(
        "Temporal cold-start: loading %s (cutoff=%s, min_movies=%d, first_post_t_only=%s)",
        data_path,
        cutoff_date,
        min_movie_interactions,
        first_post_t_test_only,
    )
    ratings = pd.read_parquet(data_path)
    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df = ratings[ratings["domain"] == "game"].copy()

    T = pd.Timestamp(cutoff_date)
    ts = pd.to_datetime(game_df["timestamp"], errors="coerce")
    valid_ts = ts.notna()
    if not valid_ts.any():
        raise ValueError("No game rows with valid timestamps — cannot build temporal split")

    g = game_df.loc[valid_ts].copy()
    g["_ts"] = ts[valid_ts]
    game_before = g.loc[g["_ts"] < T].drop(columns=["_ts"])
    game_after = g.loc[g["_ts"] >= T].drop(columns=["_ts"])

    users_with_before = set(game_before["user_id"].unique())
    users_with_after = set(game_after["user_id"].unique())
    cold_user_ids = users_with_after - users_with_before

    movie_counts = movie_df.groupby("user_id").size()
    eligible_cold = {
        uid
        for uid in cold_user_ids
        if int(movie_counts.get(uid, 0)) >= min_movie_interactions
    }
    logger.info(
        "Cold candidates (0 game before T, ≥1 after, ≥%d movies): %d users",
        min_movie_interactions,
        len(eligible_cold),
    )

    ga = game_after[game_after["user_id"].isin(eligible_cold)].sort_values(
        ["user_id", "timestamp"], kind="mergesort"
    )
    if first_post_t_test_only:
        game_test = ga.groupby("user_id", sort=False).head(1).copy()
    else:
        game_test = ga.copy()

    game_train = game_before
    cross_train = pd.concat([movie_df, game_train], ignore_index=True)
    logger.info(
        "game_train (before T): %d rows | game_test (cold, after T): %d rows",
        len(game_train),
        len(game_test),
    )

    all_users = sorted(
        {normalize_id(u) for u in pd.concat([movie_df, game_df])["user_id"].unique()}
    )
    all_items = sorted(
        {normalize_id(i) for i in pd.concat([movie_df, game_df])["item_id"].unique()}
    )
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {it: i for i, it in enumerate(all_items)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    game_item_ids = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids = {normalize_id(x) for x in movie_df["item_id"].unique()}
    game_item_indices = {item_to_idx[iid] for iid in game_item_ids if iid in item_to_idx}
    movie_item_indices = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}

    game_test_relevant, game_train_seen = _build_eval_structures(
        game_test, game_train, user_to_idx, item_to_idx
    )
    eval_user_indices = sorted(game_test_relevant.keys())

    warm_item_ids = {normalize_id(x) for x in game_train["item_id"].unique()}
    before_filter = len(eval_user_indices)
    eval_user_indices = [
        uid
        for uid in eval_user_indices
        if game_test_relevant.get(uid)
        and all(
            normalize_id(idx_to_item[iid]) in warm_item_ids
            for iid in game_test_relevant[uid]
        )
    ]
    if before_filter != len(eval_user_indices):
        logger.info(
            "Dropped %d users whose post-T test item(s) never appear in game_train "
            "(before-T corpus) — %d remain",
            before_filter - len(eval_user_indices),
            len(eval_user_indices),
        )

    # Movie-count subgroups (same buckets as temporal bench summary)
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    movie_counts_n = movie_df.groupby(movie_df["user_id"].map(normalize_id)).size()

    subgroups: dict[str, list[int]] = {"temporal_cold_start": list(eval_user_indices)}
    for uid_idx in eval_user_indices:
        uid_str = idx_to_user.get(uid_idx, "")
        mc = int(movie_counts_n.get(uid_str, 0))
        if mc < 15:
            subgroups.setdefault("cold_start_low_movies", []).append(uid_idx)
        elif mc < 50:
            subgroups.setdefault("cold_start_med_movies", []).append(uid_idx)
        else:
            subgroups.setdefault("cold_start_rich_movies", []).append(uid_idx)

    empty_game = _empty_ratings_like(game_df)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Temporal cold-start eval users: %d", len(eval_user_indices))

    return CrossDomainSplit(
        game_train=game_train,
        game_val=empty_game,
        game_test=game_test,
        movie_train=movie_df,
        movie_val=empty_game,
        movie_test=empty_game,
        book_train=empty_game,
        book_val=empty_game,
        book_test=empty_game,
        movie_all=movie_df,
        game_all=game_df,
        book_all=empty_game,
        cross_train=cross_train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        game_item_indices=game_item_indices,
        movie_item_indices=movie_item_indices,
        book_item_indices=set(),
        game_test_relevant=game_test_relevant,
        game_val_relevant={},
        game_train_seen=game_train_seen,
        movie_test_relevant={},
        movie_val_relevant={},
        movie_train_seen={},
        book_test_relevant={},
        book_val_relevant={},
        book_train_seen={},
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain="game",
        domain_pair=domain_pair,
        device=device,
    )


def load_few_shot_split(
    min_games: int = 2,
    max_games: int = 10,
    min_movie_interactions: int = 5,
    domain_pair: str = "movie_game_transfer_loose",
    seed: int = SEED,
) -> CrossDomainSplit:
    """Few-shot evaluation: test users have 2–max_games total game interactions.

    Design:
      Few-shot eval users (min_games ≤ total_games ≤ max_games):
        k-1 games in training (leave-last-out), last game = test
        All movie interactions always in cross_train

      Warm users (total_games > max_games):
        ALL game interactions in training (no test item)

    This isolates the regime where CDR models can use movie signals to
    supplement sparse game history, and lets us compare:
      LightGCN  — uses only the k-1 game interactions (sparse)
      EMCDR     — uses only mapping(movie_emb); ignores k-1 games
      CMF       — joint embedding updated by both movies and k-1 games
      PTUPCDR   — personalized mapping, can blend pref + game emb

    Subgroups by game training count (k_train):
      k1:   exactly 1 game in training  (2 total)
      k2:   exactly 2 games in training (3 total)
      k35:  3–5 games in training
      k69:  6–9 games in training
    """
    if not domain_pair.startswith("movie_game"):
        raise ValueError("load_few_shot_split only supports movie_game pairs")

    configure_benchmark(domain_pair)

    data_path = DATA_DIR / "ratings.parquet"
    logger.info(
        "Few-shot split: loading %s (min_games=%d, max_games=%d, min_movie=%d)",
        data_path, min_games, max_games, min_movie_interactions,
    )
    ratings = pd.read_parquet(data_path)

    movie_df = ratings[ratings["domain"] == "movie"].copy()
    game_df  = ratings[ratings["domain"] == "game"].copy()

    game_counts  = game_df.groupby("user_id").size()
    movie_counts = movie_df.groupby("user_id").size()

    # Few-shot users: min_games ≤ total_games ≤ max_games AND enough movies
    few_shot_ids = set(
        game_counts[(game_counts >= min_games) & (game_counts <= max_games)].index
    ) & set(movie_counts[movie_counts >= min_movie_interactions].index)

    # Warm users: > max_games games (all game in training, no test)
    warm_ids = set(game_counts[game_counts > max_games].index)

    logger.info(
        "Users: %d few-shot (2–%d games), %d warm (>%d games)",
        len(few_shot_ids), max_games, len(warm_ids), max_games,
    )

    # ── Leave-last-out for few-shot users ─────────────────────────────────
    few_shot_game = game_df[game_df["user_id"].isin(few_shot_ids)].copy()
    few_shot_sorted = few_shot_game.sort_values(["user_id", "timestamp"])
    game_test = few_shot_sorted.groupby("user_id").tail(1).copy()
    # Training: all but last game for few-shot users
    test_mask = few_shot_sorted.index.isin(game_test.index)
    few_shot_game_train = few_shot_sorted[~test_mask].copy()

    # ── game_train: few-shot users' training games + ALL warm users' games ─
    warm_game = game_df[game_df["user_id"].isin(warm_ids)].copy()
    game_train = pd.concat([few_shot_game_train, warm_game], ignore_index=True)

    # ── Unified maps ──────────────────────────────────────────────────────
    all_users = sorted({normalize_id(u) for u in pd.concat([movie_df, game_df])["user_id"].unique()})
    all_items = sorted({normalize_id(i) for i in pd.concat([movie_df, game_df])["item_id"].unique()})
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {it: i for i, it in enumerate(all_items)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    game_item_ids    = {normalize_id(x) for x in game_df["item_id"].unique()}
    movie_item_ids   = {normalize_id(x) for x in movie_df["item_id"].unique()}
    game_item_indices  = {item_to_idx[iid] for iid in game_item_ids  if iid in item_to_idx}
    movie_item_indices = {item_to_idx[iid] for iid in movie_item_ids if iid in item_to_idx}

    logger.info(
        "Maps: %d users, %d items (game=%d, movie=%d)",
        len(user_to_idx), len(item_to_idx),
        len(game_item_indices), len(movie_item_indices),
    )

    # ── Eval structures ───────────────────────────────────────────────────
    cross_train = pd.concat([movie_df, game_train], ignore_index=True)
    game_test_relevant, game_train_seen = _build_eval_structures(
        game_test, game_train, user_to_idx, item_to_idx,
    )
    eval_user_indices = sorted(game_test_relevant.keys())

    logger.info(
        "cross_train: %d (movie=%d + game_train=%d); eval few-shot users: %d",
        len(cross_train), len(movie_df), len(game_train), len(eval_user_indices),
    )

    # ── Subgroups by k_train (game interactions in training) ─────────────
    idx_to_user = {v: k for k, v in user_to_idx.items()}
    # Per-user game training count
    user_game_train_counts = (
        few_shot_game_train.groupby("user_id").size().to_dict()
    )

    subgroups: dict[str, list[int]] = {"few_shot_eval": list(eval_user_indices)}
    for uid_idx in eval_user_indices:
        uid_str = idx_to_user.get(uid_idx, "")
        k = user_game_train_counts.get(uid_str, 0)
        if k == 1:
            bucket = "k1_game_train"
        elif k == 2:
            bucket = "k2_game_train"
        elif k <= 5:
            bucket = "k3_5_game_train"
        else:
            bucket = "k6_9_game_train"
        subgroups.setdefault(bucket, []).append(uid_idx)

    n_per_bucket = {k: len(v) for k, v in subgroups.items()}
    logger.info("Subgroup sizes: %s", n_per_bucket)

    device = (
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )

    empty_game = _empty_ratings_like(game_df)
    return CrossDomainSplit(
        game_train=game_train,
        game_val=empty_game,
        game_test=game_test,
        movie_train=movie_df,
        movie_val=empty_game,
        movie_test=empty_game,
        book_train=empty_game,
        book_val=empty_game,
        book_test=empty_game,
        movie_all=movie_df,
        game_all=game_df,
        book_all=empty_game,
        cross_train=cross_train,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        game_item_indices=game_item_indices,
        movie_item_indices=movie_item_indices,
        book_item_indices=set(),
        game_test_relevant=game_test_relevant,
        game_val_relevant={},
        game_train_seen=game_train_seen,
        movie_test_relevant={},
        movie_val_relevant={},
        movie_train_seen={},
        book_test_relevant={},
        book_val_relevant={},
        book_train_seen={},
        eval_user_indices=eval_user_indices,
        user_subgroups=subgroups,
        target_domain="game",
        domain_pair=domain_pair,
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Leakage checks
# ═══════════════════════════════════════════════════════════════════════════

def verify_no_leakage(data: CrossDomainSplit) -> None:
    """Raise AssertionError if any leakage is detected (checks target domain split)."""
    logger.info("Running leakage checks …")

    target_train = data.target_train
    target_test = data.target_test
    target_val = data.target_val

    # 1. No test/val item appears in train for the same user
    train_pairs = set(zip(target_train["user_id"], target_train["item_id"]))
    test_pairs = set(zip(target_test["user_id"], target_test["item_id"]))
    val_pairs = set(zip(target_val["user_id"], target_val["item_id"]))
    leak_test = train_pairs & test_pairs
    leak_val = train_pairs & val_pairs
    assert len(leak_test) == 0, f"LEAK: {len(leak_test)} target test items in train!"
    assert len(leak_val) == 0, f"LEAK: {len(leak_val)} target val items in train!"

    # 2. No future target interactions in training — vectorised
    if target_test["timestamp"].notna().any():
        test_ts = target_test.groupby("user_id")["timestamp"].min()
        train_ts = target_train[["user_id", "timestamp"]].dropna(subset=["timestamp"])
        merged = train_ts.join(test_ts.rename("test_ts"), on="user_id", how="inner")
        leaked = merged[merged["timestamp"] > merged["test_ts"]]
        assert len(leaked) == 0, f"LEAK: {len(leaked)} train rows after test timestamp"

    # 3. Eval users all have target test relevant items
    for uid in data.eval_user_indices:
        assert uid in data.target_test_relevant, f"Eval user {uid} has no relevant test items"

    logger.info("✓ All leakage checks passed")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_cross_domain(
    model_name: str,
    predict_fn,
    data: CrossDomainSplit,
    eval_user_override: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate on held-out target-domain test set with full ranking.

    Args:
        model_name: For logging.
        predict_fn: ``(user_idx: int) -> np.ndarray`` of scores for **all**
                    items (unified index space).  Non-target items are masked.
        data: The shared ``CrossDomainSplit``.
        eval_user_override: If set, evaluate only these user indices instead of
                            data.eval_user_indices.

    Returns:
        Dict with overall metrics + per-subgroup metrics.
    """
    num_items = data.num_items
    target_items_arr = np.array(sorted(data.target_item_indices), dtype=np.int64)

    target_mask = np.zeros(num_items, dtype=bool)
    target_mask[target_items_arr] = True

    per_user_metrics: list[dict[str, float]] = []
    per_user_uid: list[int] = []

    eval_users = eval_user_override if eval_user_override is not None else data.eval_user_indices
    for uid in eval_users:
        relevant = data.target_test_relevant.get(uid, set())
        if not relevant:
            continue
        try:
            scores = predict_fn(uid).copy()
        except Exception:
            continue

        scores[~target_mask] = -np.inf
        for iid in data.target_train_seen.get(uid, set()):
            if 0 <= iid < num_items:
                scores[iid] = -np.inf
        top_indices = np.argsort(scores)[::-1][:max(K_VALUES)]
        recommended = top_indices.tolist()

        m = compute_all_metrics(recommended, relevant, k_values=K_VALUES)
        per_user_metrics.append(m)
        per_user_uid.append(uid)

    # ── Overall aggregation ──────────────────────────────────────────────
    overall = aggregate_metrics(per_user_metrics)
    logger.info(
        "%s [%s]  overall (%d users): Recall@10=%.4f  NDCG@10=%.4f",
        model_name, data.target_domain, len(per_user_metrics),
        overall.get("recall@10", 0),
        overall.get("ndcg@10", 0),
    )

    # ── Subgroup aggregation ─────────────────────────────────────────────
    uid_to_pos = {uid: i for i, uid in enumerate(per_user_uid)}
    subgroup_results: dict[str, dict[str, float]] = {}
    for sg_name, sg_uids in data.user_subgroups.items():
        sg_metrics = [
            per_user_metrics[uid_to_pos[uid]]
            for uid in sg_uids if uid in uid_to_pos
        ]
        if sg_metrics:
            agg = aggregate_metrics(sg_metrics)
            subgroup_results[sg_name] = agg
            logger.info(
                "  %-35s (%4d users): Recall@10=%.4f  NDCG@10=%.4f",
                sg_name, len(sg_metrics),
                agg.get("recall@10", 0), agg.get("ndcg@10", 0),
            )

    return {**overall, "subgroups": subgroup_results, "n_eval_users": len(per_user_metrics)}


def evaluate_sampled_negatives(
    model_name: str,
    predict_fn,
    data: CrossDomainSplit,
    n_negatives: int = 99,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate using the sampled-negative protocol (1 positive + n_negatives).

    For each eval user:
      - Candidate set = {test_item} ∪ n_negatives randomly sampled game items
        (excluding train items and the test item itself)
      - Rank the test item among these n_negatives + 1 items
      - Compute HR@K and NDCG@K within this candidate set

    This is stricter than full-rank evaluation and emphasises whether the model
    can distinguish the positive from nearby noise.  Cutoffs 1, 5, 10 are used.
    """
    rng = np.random.RandomState(seed)
    target_items_arr = np.array(sorted(data.target_item_indices), dtype=np.int64)
    n_target = len(target_items_arr)
    neg_k_values = [1, 5, 10]

    per_user_metrics: list[dict[str, float]] = []
    per_user_uid: list[int] = []

    for uid in data.eval_user_indices:
        relevant = data.target_test_relevant.get(uid, set())
        if not relevant:
            continue
        test_item = next(iter(relevant))  # single test item (leave-last-out)

        # Pool: target items minus train-seen minus test item
        exclude = data.target_train_seen.get(uid, set()) | {test_item}
        candidate_pool = target_items_arr[
            ~np.isin(target_items_arr, list(exclude))
        ]
        if len(candidate_pool) < n_negatives:
            continue  # not enough negatives — skip user

        neg_items = rng.choice(candidate_pool, size=n_negatives, replace=False)
        candidate_items = np.concatenate([[test_item], neg_items])  # pos first

        try:
            scores = predict_fn(uid)
        except Exception:
            continue

        cand_scores = scores[candidate_items]
        # Rank descending: position of test item (index 0) in sorted candidate set
        rank = int((cand_scores > cand_scores[0]).sum()) + 1  # 1-indexed

        m: dict[str, float] = {}
        for k in neg_k_values:
            hit = float(rank <= k)
            ndcg = (1.0 / np.log2(rank + 1)) if rank <= k else 0.0
            m[f"hr@{k}"] = hit
            m[f"ndcg@{k}"] = ndcg
        per_user_metrics.append(m)
        per_user_uid.append(uid)

    overall: dict[str, float] = {}
    if per_user_metrics:
        for key in per_user_metrics[0]:
            overall[f"sampled_{key}"] = float(np.mean([m[key] for m in per_user_metrics]))

    logger.info(
        "%s [sampled@%d]  (%d users): HR@1=%.4f  HR@10=%.4f  NDCG@10=%.4f",
        model_name, n_negatives, len(per_user_metrics),
        overall.get("sampled_hr@1", 0),
        overall.get("sampled_hr@10", 0),
        overall.get("sampled_ndcg@10", 0),
    )

    # Subgroup aggregation
    uid_to_pos = {uid: i for i, uid in enumerate(per_user_uid)}
    subgroup_results: dict[str, dict[str, float]] = {}
    for sg_name, sg_uids in data.user_subgroups.items():
        sg_m = [per_user_metrics[uid_to_pos[uid]] for uid in sg_uids if uid in uid_to_pos]
        if sg_m:
            sg_agg: dict[str, float] = {}
            for key in sg_m[0]:
                sg_agg[f"sampled_{key}"] = float(np.mean([m[key] for m in sg_m]))
            subgroup_results[sg_name] = sg_agg
            logger.info(
                "  %-35s (%4d users): HR@10=%.4f  NDCG@10=%.4f",
                sg_name, len(sg_m),
                sg_agg.get("sampled_hr@10", 0), sg_agg.get("sampled_ndcg@10", 0),
            )

    return {**overall, "sampled_subgroups": subgroup_results, "n_sampled_users": len(per_user_metrics)}


def evaluate_validation(
    model_name: str,
    predict_fn,
    data: CrossDomainSplit,
) -> dict[str, float]:
    """Evaluate on held-out target-domain *validation* set (for early stopping).

    Same protocol as ``evaluate_cross_domain`` but uses ``target_val_relevant``
    instead of ``target_test_relevant``, so the test set stays untouched.
    """
    num_items = data.num_items
    target_items_arr = np.array(sorted(data.target_item_indices), dtype=np.int64)
    target_mask = np.zeros(num_items, dtype=bool)
    target_mask[target_items_arr] = True

    per_user_metrics: list[dict[str, float]] = []
    val_relevant = data.target_val_relevant

    for uid in data.eval_user_indices:
        relevant = val_relevant.get(uid, set())
        if not relevant:
            continue
        try:
            scores = predict_fn(uid).copy()
        except Exception:
            continue

        scores[~target_mask] = -np.inf
        for iid in data.target_train_seen.get(uid, set()):
            if 0 <= iid < num_items:
                scores[iid] = -np.inf
        max_k = max(K_VALUES)
        top_indices = np.argsort(scores)[::-1][:max_k]
        recommended = top_indices.tolist()

        m = compute_all_metrics(recommended, relevant, k_values=K_VALUES)
        per_user_metrics.append(m)

    overall = aggregate_metrics(per_user_metrics)
    logger.info(
        "%s [val]  (%d users): NDCG@10=%.4f  Recall@10=%.4f",
        model_name, len(per_user_metrics),
        overall.get("ndcg@10", 0), overall.get("recall@10", 0),
    )
    return overall


# ═══════════════════════════════════════════════════════════════════════════
# Artifact saving
# ═══════════════════════════════════════════════════════════════════════════

def save_embedding_artifacts(algo: str, user_emb, item_emb) -> None:
    """Save user/item embeddings as .npy to artifacts/demo/."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(user_emb, torch.Tensor):
        user_emb = user_emb.cpu().numpy()
    if isinstance(item_emb, torch.Tensor):
        item_emb = item_emb.cpu().numpy()
    np.save(DEMO_DIR / f"{algo.lower()}_user.npy", user_emb)
    np.save(DEMO_DIR / f"{algo.lower()}_item.npy", item_emb)
    logger.info("Saved %s embeddings to %s", algo, DEMO_DIR)


def save_result(
    algo: str,
    metrics: dict[str, Any],
    train_time: float = 0.0,
    description: str = "",
) -> None:
    """Append a benchmark run to artifacts/results/<algo>.json as a history array.

    Each entry contains: description, time_completed, all metrics, train_time_s.
    Old single-dict files are automatically migrated to the array format.
    """
    from datetime import datetime as _dt
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{algo.lower()}.json"

    # Load existing history, migrating old dict format if needed
    history: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, list):
                history = existing
            elif isinstance(existing, dict):
                history = [existing]  # migrate: wrap single run in array
        except Exception:
            pass

    entry = {
        "description": description,
        "time_completed": _dt.now().isoformat(),
        **metrics,
        "train_time_s": train_time,
    }
    history.append(entry)
    path.write_text(json.dumps(history, indent=2, default=str))
    logger.info("Saved %s results to %s (run #%d)", algo, path, len(history))


def save_mappings(data: CrossDomainSplit) -> None:
    """Save ID mappings to artifacts/demo/ with cross_domain_ prefix."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, mapping in [
        ("cross_domain_user_to_idx", data.user_to_idx),
        ("cross_domain_item_to_idx", data.item_to_idx),
        ("cross_domain_idx_to_item", data.idx_to_item),
    ]:
        (DEMO_DIR / f"{name}.json").write_text(
            json.dumps({str(k): v for k, v in mapping.items()}, indent=0)
        )
    logger.info("Saved cross-domain mappings to %s", DEMO_DIR)


def save_mappings_simple(user_to_idx: dict, item_to_idx: dict, idx_to_item: dict) -> None:
    """Save ID mappings to artifacts/demo/ (for single-domain benchmarks)."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for name, mapping in [
        ("user_to_idx", user_to_idx),
        ("item_to_idx", item_to_idx),
        ("idx_to_item", idx_to_item),
    ]:
        (DEMO_DIR / f"{name}.json").write_text(
            json.dumps({str(k): v for k, v in mapping.items()}, indent=0)
        )
    logger.info("Saved mappings to %s", DEMO_DIR)


def load_and_split(
    domain: str,
    max_interactions: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict, str]:
    """Load ratings.parquet for single domain, build ID maps, and split.

    Args:
        domain: "game" or "movie"
        max_interactions: Optional limit for faster benchmarks

    Returns:
        (train_df, val_df, test_df, user_to_idx, item_to_idx, idx_to_item, device)
    """
    data_path = DATA_DIR / "ratings.parquet"
    logger.info("Loading %s domain data from %s", domain, data_path)
    ratings_df = pd.read_parquet(data_path)
    
    # Filter by domain
    ratings_df = ratings_df[ratings_df["domain"] == domain].copy()
    logger.info("Loaded %d %s ratings", len(ratings_df), domain)

    # User-based sampling (keeps per-user history intact)
    if max_interactions is not None and len(ratings_df) > max_interactions:
        all_users = ratings_df["user_id"].unique()
        rng = np.random.RandomState(SEED)
        target_users = max_interactions // 10  # ~10 interactions per user
        if target_users < len(all_users):
            sampled_users = rng.choice(all_users, size=target_users, replace=False)
            ratings_df = ratings_df[ratings_df["user_id"].isin(sampled_users)].copy()
        logger.info("Sampled to %d interactions (%d users)",
                    len(ratings_df), ratings_df["user_id"].nunique())

    # Build ID maps (string keys)
    unique_users = sorted({normalize_id(u) for u in ratings_df["user_id"].unique()})
    unique_items = sorted({normalize_id(i) for i in ratings_df["item_id"].unique()})
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}

    logger.info("ID maps: %d users, %d items", len(user_to_idx), len(item_to_idx))

    # Leave-last-out split (standard evaluation protocol)
    np.random.seed(SEED)
    train_df, val_df, test_df = leave_last_out_split(ratings_df)
    logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    return train_df, val_df, test_df, user_to_idx, item_to_idx, idx_to_item, device


def evaluate_model(
    model_name: str,
    predict_fn,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    max_users: int = 2000,
) -> dict[str, float]:
    """Evaluate a model using leave-last-out test set.

    Args:
        model_name: Name for logging.
        predict_fn: Callable(user_idx) -> np.ndarray of scores for all items.
        test_df: Test DataFrame with user_id, item_id, rating columns.
        train_df: Training DataFrame (to exclude seen items).
        user_to_idx: User ID -> index mapping.
        item_to_idx: Item ID -> index mapping.
        max_users: Limit number of users to evaluate (for speed).
    """
    # Build per-user test set (items with rating >= threshold) — vectorized
    test_relevant: dict[int, set[int]] = {}
    _t = test_df[test_df["rating"] >= POSITIVE_THRESHOLD].copy()
    _t["_uid"] = _t["user_id"].map(lambda v: user_to_idx.get(normalize_id(v)))
    _t["_iid"] = _t["item_id"].map(lambda v: item_to_idx.get(normalize_id(v)))
    _t = _t.dropna(subset=["_uid", "_iid"])
    for uid, grp in _t.groupby("_uid"):
        test_relevant[int(uid)] = set(grp["_iid"].astype(int))

    # Build per-user train set (to exclude) — vectorized
    train_items: dict[int, set[int]] = {}
    _tr = train_df.copy()
    _tr["_uid"] = _tr["user_id"].map(lambda v: user_to_idx.get(normalize_id(v)))
    _tr["_iid"] = _tr["item_id"].map(lambda v: item_to_idx.get(normalize_id(v)))
    _tr = _tr.dropna(subset=["_uid", "_iid"])
    for uid, grp in _tr.groupby("_uid"):
        train_items[int(uid)] = set(grp["_iid"].astype(int))

    # Evaluate
    eval_users = sorted(test_relevant.keys())
    if len(eval_users) > max_users:
        rng = np.random.RandomState(SEED)
        eval_users = sorted(rng.choice(eval_users, max_users, replace=False))

    per_user = []
    for uid in eval_users:
        relevant = test_relevant.get(uid, set())
        if not relevant:
            continue
        try:
            scores = predict_fn(uid)
        except Exception:
            continue
        exclude = train_items.get(uid, set())
        for iid in exclude:
            if 0 <= iid < len(scores):
                scores[iid] = -np.inf

        max_k = max(K_VALUES)
        top_indices = np.argsort(scores)[::-1][:max_k]
        recommended = top_indices.tolist()
        metrics = compute_all_metrics(recommended, relevant, k_values=K_VALUES)
        per_user.append(metrics)

    agg = aggregate_metrics(per_user)
    logger.info(
        "%s evaluation: %d users | NDCG@10=%.4f | Recall@10=%.4f | HR@10=%.4f",
        model_name, len(per_user),
        agg.get("ndcg@10", 0), agg.get("recall@10", 0), agg.get("hit_rate@10", 0),
    )
    return agg
