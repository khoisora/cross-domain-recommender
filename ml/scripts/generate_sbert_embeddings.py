"""Lightweight SBERT item encoder — generates sbert_item.npy without full evaluation.

This is the fastest path to enabling SBERT-powered features in the demo:
  • "Semantically Similar" section on item detail pages
  • "Because You Rated" recommendation row

It skips the expensive cross-domain evaluation loop that bench_sbert.py runs,
so it finishes in 1-3 minutes on CPU (vs ~5+ min for the full benchmark).

Usage:
    python ml/scripts/generate_sbert_embeddings.py
    # or via Make:
    make encode-sbert
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_scripts_dir = str(Path(__file__).resolve().parent)
_root = str(Path(__file__).resolve().parent.parent.parent)
for p in (_scripts_dir, _root):
    if p not in sys.path:
        sys.path.insert(0, p)

_ARTIFACTS_DEMO = Path(_root) / "artifacts" / "demo"
_ITEM_TO_IDX_PATH = _ARTIFACTS_DEMO / "cross_domain_item_to_idx.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> None:
    # ── Load cross-domain item mapping (already saved by any previous benchmark) ──
    if not _ITEM_TO_IDX_PATH.exists():
        logger.error(
            "cross_domain_item_to_idx.json not found at %s\n"
            "Run any benchmark first (e.g. make bench-mf) to generate ID mappings.",
            _ITEM_TO_IDX_PATH,
        )
        sys.exit(1)

    logger.info("Loading cross-domain item mapping from %s …", _ITEM_TO_IDX_PATH)
    with open(_ITEM_TO_IDX_PATH) as f:
        item_to_idx: dict[str, int] = json.load(f)
    logger.info("Mapping covers %d items", len(item_to_idx))

    # ── Load item metadata from SQLite ────────────────────────────────────────
    from backend.db.database import db_service
    logger.info("Loading item metadata from SQLite …")
    items = db_service.get_items(limit=100_000)
    logger.info(
        "Loaded %d items (%d movies, %d games)",
        len(items),
        sum(1 for it in items if it.get("domain") == "movie"),
        sum(1 for it in items if it.get("domain") == "game"),
    )

    # ── Encode with SBERT ─────────────────────────────────────────────────────
    from ml.models.sbert_model import SBERTModel

    t0 = time.time()
    model = SBERTModel(model_name=MODEL_NAME)
    model.encode_items(items, item_to_idx, batch_size=64)
    elapsed = time.time() - t0

    # ── Save item embeddings only ─────────────────────────────────────────────
    _ARTIFACTS_DEMO.mkdir(parents=True, exist_ok=True)
    out_path = _ARTIFACTS_DEMO / "sbert_item.npy"
    np.save(out_path, model.get_item_embeddings())

    logger.info(
        "Saved sbert_item.npy  shape=%s  path=%s  (%.1fs)",
        model.get_item_embeddings().shape,
        out_path,
        elapsed,
    )
    print(f"\nDone! sbert_item.npy saved in {elapsed:.1f}s")
    print("Restart the backend to load the new embeddings.")


if __name__ == "__main__":
    main()
