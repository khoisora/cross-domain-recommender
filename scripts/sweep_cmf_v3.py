"""Focused CMF cold-start sweep: reg=0 only, wider emb/epochs/alpha range."""
import json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from ml.scripts.benchmarks.benchmark_common import (
    evaluate_cross_domain, load_user_split_cold_start_split,
    setup_logging, POSITIVE_THRESHOLD,
)
from ml.models._cdr_base import fit_cdr

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sweep")
logger.setLevel(logging.INFO)


class CMFSweep:
    def __init__(self, nu, ni, ed=64):
        self.num_users, self.num_items, self.embedding_dim = nu, ni, ed
        self.user_embeddings = self.item_embeddings = None
        self._valid_items = self._valid_users = None

    def fit(self, ratings, u2i, i2i, epochs=50, lr=0.001, bs=4096, pt=4.0, alpha=0.1):
        m, rbu, rbi = fit_cdr(
            "CMF", {"embedding_size": self.embedding_dim, "alpha": alpha,
                    "lambda": 0.001, "gamma": 0.001},
            ratings, u2i, i2i, epochs, lr, 0.0, bs, pt,
        )
        with torch.no_grad():
            self.user_embeddings = m.user_embedding.weight.cpu().numpy()[rbu]
            self.item_embeddings = m.item_embedding.weight.cpu().numpy()[rbi]
        self._valid_items, self._valid_users = rbi != 0, rbu != 0

    def predict(self, uid, ii=None):
        if not self._valid_users[uid]:
            return np.full(self.num_items if ii is None else len(ii), -np.inf)
        ue = self.user_embeddings[uid]
        it = self.item_embeddings if ii is None else self.item_embeddings[ii]
        sc = (it @ ue).astype(np.float64)
        # sigmoid helps per v1 sweep
        sc = 1.0 / (1.0 + np.exp(-sc))
        v = self._valid_items if ii is None else self._valid_items[ii]
        sc[~v] = -np.inf
        return sc


def main():
    data = load_user_split_cold_start_split(domain_pair="movie_game")
    data.device = "cpu"

    configs = []
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        for ed in [64, 96, 128, 192]:
            for ep in [50, 100, 150]:
                configs.append({"alpha": alpha, "emb_dim": ed, "epochs": ep})

    logger.info("Running %d configs (reg=0, sigmoid=True, lr=0.001)", len(configs))
    results = []

    for i, cfg in enumerate(configs):
        m = CMFSweep(data.num_users, data.num_items, ed=cfg["emb_dim"])
        t0 = time.time()
        m.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=cfg["epochs"], alpha=cfg["alpha"])
        tt = time.time() - t0
        met = evaluate_cross_domain("CMF", lambda uid: m.predict(uid), data)
        r = met.get("recall@10", 0)
        n = met.get("ndcg@10", 0)
        sh = met.get("sampled_hr@10", 0)
        sn = met.get("sampled_ndcg@10", 0)
        results.append({"cfg": cfg, "r": r, "n": n, "sh": sh, "sn": sn, "t": tt})
        print(f"[{i+1}/{len(configs)}] r={r:.4f} n={n:.4f} sh={sh:.4f} sn={sn:.4f} ({tt:.0f}s) | {cfg}",
              flush=True)

    results.sort(key=lambda x: x["r"], reverse=True)
    print("\n=== Top 10 by Recall@10 ===")
    for r in results[:10]:
        print(f"  recall={r['r']:.4f} ndcg={r['n']:.4f} s_hr={r['sh']:.4f} s_ndcg={r['sn']:.4f} | {r['cfg']}")

    json.dump(results, open("artifacts/sweeps/cmf_coldstart_v3.json", "w"), indent=2)


if __name__ == "__main__":
    main()
