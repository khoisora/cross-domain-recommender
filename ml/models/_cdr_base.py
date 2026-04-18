"""Shared setup for recbole-cdr cross-domain model wrappers (CMF, EMCDR, PTUPCDR).

Pipeline:
  1. Write source/target .inter files from ratings DataFrame
  2. Build CDRConfig with domain-specific subconfigs
  3. Train with CrossDomainTrainer (validation skipped — bench scripts evaluate externally)
  4. Build our_idx → recbole unified ID maps for user/item embedding extraction

RecBole assigns overlap / source-only / target-only IDs separately; raw IDs that appear
only in the source .inter file are absent from target_*_remap_dict and vice versa.
Lookups must chain both dicts so every user/item maps to the unified embedding index.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections import ChainMap
from pathlib import Path

import numpy as np
import pandas as pd

# Make recbole_cdr importable as a sibling package
sys.path.insert(0, str(Path(__file__).parent))

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.trainer import CrossDomainTrainer

from .id_utils import normalize_id, normalize_maps


def _write_inter(df: pd.DataFrame, path: Path) -> None:
    """Write a recbole atomic .inter file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "user_id:token\titem_id:token\trating:float\n"
        + df[["user_id", "item_id", "rating"]].to_csv(sep="\t", header=False, index=False)
    )


def fit_cdr(
    model_name: str,
    extra_params: dict,
    ratings: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    epochs: int,
    lr: float,
    reg_lambda: float,
    batch_size: int,
    positive_threshold: float,
    source_domain: str = "movie",
    target_domain: str = "game",
):
    """Train a recbole-cdr model; return (model, rb_users, rb_items).

    rb_users[our_idx] / rb_items[our_idx] = recbole unified IDs (0 = PAD if unknown).
    """
    user_to_idx, item_to_idx = normalize_maps(user_to_idx, item_to_idx)

    # Split ratings by domain — RecBole-CDR expects separate source/target files
    src = ratings[ratings["domain"] == source_domain][["user_id", "item_id", "rating"]].copy()
    tgt = ratings[ratings["domain"] == target_domain][["user_id", "item_id", "rating"]].copy()
    # Normalize IDs to canonical string form for consistent matching
    for d in (src, tgt):
        d.dropna(subset=["user_id", "item_id"], inplace=True)
        d["user_id"] = d["user_id"].map(normalize_id)
        d["item_id"] = d["item_id"].map(normalize_id)

    tmpdir = tempfile.mkdtemp(prefix=f"{model_name.lower()}_")
    try:
        p = Path(tmpdir)
        _write_inter(src, p / "source" / "source.inter")
        _write_inter(tgt, p / "target" / "target.inter")

        domain_cfg = {
            "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating", "LABEL_FIELD": "label",
            "NEG_PREFIX": "neg_",
            "load_col": {"inter": ["user_id", "item_id", "rating"]},
            "user_inter_num_interval": "[1,inf)",
            "item_inter_num_interval": "[1,inf)",
            "val_interval": {"rating": f"[{positive_threshold},inf)"},
            "drop_filter_field": True,
        }
        weight_decay = extra_params.pop("weight_decay", reg_lambda)
        config = CDRConfig(model=model_name, config_dict={
            "model": model_name,
            "data_path": tmpdir,
            "train_epochs": extra_params.pop("train_epochs", [f"BOTH:{epochs}"]),
            "train_batch_size": batch_size, "eval_batch_size": batch_size,
            "overlap_batch_size": batch_size,
            "learning_rate": lr, "weight_decay": weight_decay,
            "neg_sampling": [{"popularity": 1}],
            "train_neg_sample_args": {
                "distribution": "popularity", "sample_num": 1,
                "alpha": 0.75, "dynamic": False, "candidate_num": 0,
            },
            "user_link_file_path": None, "item_link_file_path": None,
            "eval_args": {"split": {"RS": [0.99, 0.005, 0.005]},
                          "split_valid": {"RS": [0.99, 0.01]},
                          "order": "RO", "group_by": "user",
                          "mode": {"valid": "full", "test": "full"}},
            "metrics": ["Recall"], "topk": [10], "valid_metric": "Recall@10",
            "numerical_features": [], "worker": 0,
            "enable_amp": False, "enable_scaler": False,
            "show_progress": False, "save_dataset": False,
            "checkpoint_dir": str(p / "saved"),
            "source_domain": {"dataset": "source", "data_path": tmpdir, **domain_cfg},
            "target_domain": {"dataset": "target", "data_path": tmpdir, **domain_cfg},
            **extra_params,
        })

        dataset = create_dataset(config)
        train_data, _, _ = data_preparation(config, dataset)

        from recbole_cdr.model.cross_domain_recommender import CMF, EMCDR
        model_cls = {"CMF": CMF, "EMCDR": EMCDR}[model_name]
        model = model_cls(config, dataset)

        CrossDomainTrainer(config, model).fit(train_data, None, verbose=True, saved=False)

        # Build our_idx → RecBole unified ID maps. ChainMap merges source and
        # target remap dicts so lookups find the ID in whichever domain it appears.
        # Value 0 = RecBole's PAD token, meaning the user/item was not seen.
        u_remap = ChainMap(dataset.source_user_ID_remap_dict, dataset.target_user_ID_remap_dict)
        i_remap = ChainMap(dataset.source_item_ID_remap_dict, dataset.target_item_ID_remap_dict)
        rb_users = np.zeros(len(user_to_idx), dtype=np.int64)
        rb_items = np.zeros(len(item_to_idx), dtype=np.int64)
        for raw_uid, our_idx in user_to_idx.items():
            rb_users[our_idx] = u_remap.get(raw_uid, 0)
        for raw_iid, our_idx in item_to_idx.items():
            rb_items[our_idx] = i_remap.get(raw_iid, 0)

        return model, rb_users, rb_items
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
