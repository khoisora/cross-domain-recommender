"""Hyperparameter sweeps for the 5 models described in Section 5.9.

Each sweep trains the model across a small grid, runs the standard full-rank
LLO evaluation, and writes one JSON per point under artifacts/sweeps/.

  CMF         : lr × alpha grid on processed_overlap (lesson 3)
  LightGCN    : num_layers K ∈ {1,2,3,4,5} on processed_overlap
  PTUPCDR     : n_experts ∈ {2,4,8,16} on processed_overlap
  SBERT-CDR   : source_weight ∈ {0.1,0.3,0.5,0.7,0.9} on processed_overlap
  EMCDR-cooc  : blend λ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.3} (post-hoc on single EMCDR run)

Run:
  python scripts/run_hp_sweep.py --model cmf
  python scripts/run_hp_sweep.py --model lightgcn
  python scripts/run_hp_sweep.py --model ptupcdr
  python scripts/run_hp_sweep.py --model sbert_cdr
  python scripts/run_hp_sweep.py --model emcdr_cooc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Switch DATA_DIR → processed_overlap (Lesson 3 cohort) before any loader imports.
import ml.scripts.benchmarks.benchmark_common as bc  # noqa: E402
bc.DATA_DIR = ROOT / "ml" / "data" / "amazon_2023" / "processed_overlap"

from ml.scripts.benchmarks.benchmark_common import (  # noqa: E402
    POSITIVE_THRESHOLD, evaluate_cross_domain, load_cross_domain_split,
    make_full_rank_val_fn, setup_logging, verify_no_leakage, DATA_DIR,
)

OUT_DIR = ROOT / "artifacts" / "sweeps"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(model: str, point: dict, metrics: dict, train_time: float) -> None:
    tag = "_".join(f"{k}{v}" for k, v in point.items()).replace(".", "p")
    path = OUT_DIR / f"{model}__{tag}.json"
    entry = {
        "model": model,
        "hyperparams": point,
        "train_time_s": train_time,
        "recall@10": metrics.get("recall@10"),
        "ndcg@10": metrics.get("ndcg@10"),
        "hit_rate@10": metrics.get("hit_rate@10"),
    }
    path.write_text(json.dumps(entry, indent=2, default=str))
    print(f"  → {path.name}: recall@10={entry['recall@10']:.4f} "
          f"ndcg@10={entry['ndcg@10']:.4f} ({train_time:.0f}s)")


# ── CMF: lr × alpha ─────────────────────────────────────────────────────

def sweep_cmf() -> None:
    from ml.models.cmf import CMF
    data = load_cross_domain_split(target_domain="game", single_domain_item_space=False)
    verify_no_leakage(data)

    lrs    = [0.0001, 0.0005, 0.001, 0.005]
    alphas = [0.01, 0.05, 0.2, 0.5]

    for lr in lrs:
        for alpha in alphas:
            print(f"[CMF] lr={lr} alpha={alpha}")
            model = CMF(data.num_users, data.num_items,
                        embedding_dim=96, device=data.device)
            t0 = time.time()
            model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                      epochs=60, lr=lr, reg_lambda=0.0, batch_size=8192,
                      alpha=alpha, positive_threshold=POSITIVE_THRESHOLD)
            dt = time.time() - t0
            metrics = evaluate_cross_domain("CMF", lambda uid: model.predict(uid), data)
            _save("cmf", {"lr": lr, "alpha": alpha}, metrics, dt)


# ── LightGCN: K layers ──────────────────────────────────────────────────

def sweep_lightgcn() -> None:
    from ml.models.lightgcn import LightGCN
    data = load_cross_domain_split(target_domain="game", single_domain_item_space=True)
    verify_no_leakage(data)
    neg_items = np.array(sorted(data.target_item_indices), dtype=np.int64)

    for K in [1, 2, 3, 4, 5]:
        print(f"[LightGCN] K={K}")
        model = LightGCN(data.num_users, data.num_items,
                         embedding_dim=96, num_layers=K,
                         device="cpu", dropout=0.1)
        t0 = time.time()
        model.fit(data.target_train, data.user_to_idx, data.item_to_idx,
                  epochs=50, lr=0.001, reg_lambda=0.001, batch_size=4096,
                  positive_threshold=POSITIVE_THRESHOLD,
                  neg_sampling="popularity", neg_popularity_alpha=0.75,
                  neg_item_indices=neg_items,
                  early_stopping_patience=5,
                  val_metric_fn=make_full_rank_val_fn(data), val_every=3)
        dt = time.time() - t0
        metrics = evaluate_cross_domain("LightGCN", lambda uid: model.predict(uid), data)
        _save("lightgcn", {"K": K}, metrics, dt)


# ── PTUPCDR: n_experts ──────────────────────────────────────────────────

def sweep_ptupcdr() -> None:
    from ml.models.ptupcdr import PTUPCDRWrapper
    data = load_cross_domain_split(target_domain="game", single_domain_item_space=False)
    verify_no_leakage(data)

    for n_experts in [2, 4, 8, 16]:
        print(f"[PTUPCDR] n_experts={n_experts}")
        model = PTUPCDRWrapper(data.num_users, data.num_items,
                               embedding_dim=64, device=data.device)
        t0 = time.time()
        model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                  epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
                  positive_threshold=POSITIVE_THRESHOLD,
                  meta_epochs=30, meta_lr=0.001, n_experts=n_experts)
        dt = time.time() - t0
        metrics = evaluate_cross_domain("PTUPCDR", lambda uid: model.predict(uid), data)
        _save("ptupcdr", {"n_experts": n_experts}, metrics, dt)


# ── SBERT-CDR: source_weight ────────────────────────────────────────────

def sweep_sbert_cdr() -> None:
    from ml.models.sbert_model import SBERTModel
    import pandas as pd

    data = load_cross_domain_split(target_domain="game", single_domain_item_space=False)
    verify_no_leakage(data)
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    games_df  = pd.read_parquet(DATA_DIR / "games.parquet")
    items_df  = pd.concat([movies_df, games_df], ignore_index=True)

    # Encode items once — only source_weight changes between points.
    model = SBERTModel()
    model.encode_items(items_df, data.item_to_idx)

    for sw in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"[SBERT-CDR] source_weight={sw}")
        t0 = time.time()
        model.compute_cross_domain_user_embeddings(
            data.cross_train, data.user_to_idx, data.item_to_idx,
            source_domain="movie", target_domain="game",
            positive_threshold=POSITIVE_THRESHOLD,
            source_weight=sw,
        )
        dt = time.time() - t0
        metrics = evaluate_cross_domain("SBERT-CDR", lambda uid: model.predict(uid), data)
        _save("sbert_cdr", {"source_weight": sw}, metrics, dt)


# ── EMCDR + cooc post-hoc: sweep blend λ ────────────────────────────────

def sweep_emcdr_cooc() -> None:
    """Train EMCDR once, then sweep the cooc-reranking blend weight λ."""
    from ml.models.emcdr import EMCDRWrapper
    from ml.scripts.benchmarks.cooc_rerank import (
        build_movie_game_cooc, wrap_predict_with_cooc,
    )

    data = load_cross_domain_split(target_domain="game", single_domain_item_space=False)
    verify_no_leakage(data)

    print("[EMCDR-cooc] training EMCDR once…")
    model = EMCDRWrapper(data.num_users, data.num_items,
                         embedding_dim=64, device=data.device)
    t0 = time.time()
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    emcdr_time = time.time() - t0

    cooc = build_movie_game_cooc(data.movie_train, data.game_train,
                                 rating_threshold=POSITIVE_THRESHOLD)

    for lam in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]:
        print(f"[EMCDR-cooc] λ={lam}")
        predict_fn = wrap_predict_with_cooc(model.predict, data, cooc, lam=lam) \
                     if lam > 0 else model.predict
        t0 = time.time()
        metrics = evaluate_cross_domain("EMCDR-cooc", predict_fn, data)
        dt = time.time() - t0 + emcdr_time
        _save("emcdr_cooc", {"lambda": lam}, metrics, dt)


MODELS = {
    "cmf":        sweep_cmf,
    "lightgcn":   sweep_lightgcn,
    "ptupcdr":    sweep_ptupcdr,
    "sbert_cdr":  sweep_sbert_cdr,
    "emcdr_cooc": sweep_emcdr_cooc,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    args = parser.parse_args()

    setup_logging()
    t0 = time.time()
    MODELS[args.model]()
    print(f"\n[{args.model}] total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
