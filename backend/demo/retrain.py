"""Background model retraining scheduler.

Periodically re-trains LightGCN embeddings using the latest in-memory ratings,
then hot-swaps the store's embeddings. Cooc and SBERT rows already refresh
instantly (no retrain needed); this handles the collaborative model rows.

Architecture:
  - Runs in a background thread (non-blocking)
  - Interval configurable (default: 1 hour)
  - Uses the store's current user_ratings to build training data
  - Only retrains LightGCN (fastest, highest impact model)
  - Logs each retrain event for the report
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_retrain_thread: threading.Thread | None = None
_stop_event = threading.Event()


def start_retrain_scheduler(interval_seconds: int = 3600) -> None:
    """Start the background retrain loop.

    Args:
        interval_seconds: Seconds between retrains (default: 3600 = 1 hour).
    """
    global _retrain_thread
    if _retrain_thread and _retrain_thread.is_alive():
        logger.warning("Retrain scheduler already running")
        return

    _stop_event.clear()
    _retrain_thread = threading.Thread(
        target=_retrain_loop, args=(interval_seconds,), daemon=True,
    )
    _retrain_thread.start()
    logger.info("Retrain scheduler started (interval=%ds)", interval_seconds)


def stop_retrain_scheduler() -> None:
    """Stop the background retrain loop."""
    _stop_event.set()
    logger.info("Retrain scheduler stopped")


def _retrain_loop(interval: int) -> None:
    """Background loop that retrains models periodically."""
    while not _stop_event.is_set():
        # Wait for interval (interruptible)
        if _stop_event.wait(timeout=interval):
            break
        try:
            _run_retrain()
        except Exception:
            logger.exception("Retrain failed")


def _run_retrain() -> None:
    """Execute one retrain cycle: rebuild LightGCN from current ratings."""
    from backend.demo.store import DemoStore

    store = DemoStore.get()
    if not store.loaded:
        return

    logger.info("=== Starting scheduled retrain ===")
    t0 = time.time()

    try:
        import pandas as pd
        from ml.models.lightgcn import LightGCN

        # Build training data from current in-memory ratings
        rows = []
        for ext_id, ratings in store.user_ratings.items():
            for r in ratings:
                if r.get("domain") == "game" and r["rating"] >= 4.0:
                    uid = store.sd_user_to_idx.get(ext_id)
                    iid = store.sd_item_to_idx.get(r["item_id"])
                    if uid is not None and iid is not None:
                        rows.append({
                            "user_id": ext_id, "item_id": r["item_id"],
                            "rating": r["rating"],
                        })

        if len(rows) < 100:
            logger.info("Too few game ratings (%d) to retrain, skipping", len(rows))
            return

        train_df = pd.DataFrame(rows)
        logger.info("Retrain: %d positive game interactions", len(train_df))

        # Train new LightGCN
        model = LightGCN(
            store.sd_num_users, store.sd_num_items,
            embedding_dim=96, num_layers=3, device="cpu", dropout=0.1,
        )
        game_indices = np.array(sorted(
            i for i in range(store.sd_num_items)
            if store.sd_idx_to_item.get(i) in {r["item_id"] for r in rows}
        ), dtype=np.int64)
        if len(game_indices) == 0:
            game_indices = np.arange(store.sd_num_items, dtype=np.int64)

        model.fit(
            train_df, store.sd_user_to_idx, store.sd_item_to_idx,
            epochs=20, lr=0.001, reg_lambda=0.001, batch_size=4096,
            positive_threshold=4.0,
            neg_item_indices=game_indices,
        )

        # Hot-swap embeddings in store
        new_user = model.get_user_embeddings()
        new_item = model.get_item_embeddings()
        store.lgcn_user = new_user
        store.lgcn_item = new_item

        elapsed = time.time() - t0
        logger.info(
            "=== Retrain complete: LightGCN updated in %.1fs (%d interactions) ===",
            elapsed, len(rows),
        )

    except ImportError as e:
        logger.warning("Retrain skipped (missing dependency): %s", e)
    except Exception:
        logger.exception("Retrain error")
