# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Cross-domain recommender system experiments (movies → games) using the Amazon 2023 dataset. Structured as a lesson plan (`LESSON_PLAN.md`) that incrementally builds benchmark infrastructure and model comparisons.

**Reference implementation**: `~/work/MoviesGamesRecommender` — always read it before writing any model, loader, or eval logic. Port and adapt; do not rewrite from scratch.

## Setup

```bash
# Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python is pinned via `ml/scripts/.python-version`.

## Running benchmarks

All bench scripts live in `ml/scripts/benchmarks/`. Run them from the project root with `PYTHONPATH` set:

```bash
PYTHONPATH=. python ml/scripts/benchmarks/bench_mf_bpr.py --domain-pair movie_game --lesson 1
```

Run all models in parallel:
```bash
python ml/scripts/run_parallel_benchmarks.py
python ml/scripts/run_parallel_transfer_benchmarks.py
```

Generate comparison plots after running bench scripts:
```bash
python ml/scripts/plot_results.py --domain-pair movie_game --lesson 1
```

Evaluate saved model checkpoints:
```bash
python ml/scripts/eval_saved_models.py
```

## Data pipeline

Raw data lives in `ml/data/amazon_2023/raw/` (JSONL gzip). Process it to parquet:
```bash
PYTHONPATH=. python ml/data/process_data.py
```

Processed parquet files land in `ml/data/amazon_2023/processed/`. Domain pair variants (loose, strict cohorts) land in sibling dirs (`processed_transfer_loose/`, etc.).

## Architecture

```
ml/
  data/
    process_data.py       # raw JSONL → parquet; cohort filtering logic
    data_splitter.py      # leave-last-out split, CrossDomainSplit dataclass
    dataset.py            # dataset loading helpers
    item_dedup.py         # item deduplication
  models/
    base_recommender.py   # BaseRecommender, BasePyTorchRecommender, BaseLibraryRecommender
    _cdr_base.py          # shared CDR training loop (used by EMCDR, PTUPCDR)
    recbole_cdr/          # vendored RecBole-CDR source (DeepAPF, Bi-TGCF, etc.)
  evaluation/
    metrics.py            # recall_at_k, ndcg_at_k, hit_rate_at_k
    evaluator.py          # evaluate_cross_domain(), full-rank + sampled@99 modes
  scripts/
    benchmarks/
      benchmark_common.py # _DOMAIN_PAIR_PATHS, load_cross_domain_split(),
                          # save_result(), evaluate_cross_domain(), POSITIVE_THRESHOLD
      bench_*.py          # one file per model
    plot_results.py       # reads artifacts/*/results/*.json → PNG charts
    run_parallel_benchmarks.py
    run_parallel_transfer_benchmarks.py
artifacts/
  <domain-pair>/
    results/              # JSON per model run
    plots/                # PNG per lesson
```

**Data flow**: `process_data.py` → parquet → `load_cross_domain_split()` → `CrossDomainSplit` → model `.fit()` → `evaluate_cross_domain()` → `save_result()` → `plot_results.py`.

### benchmark_common.py contracts

- `POSITIVE_THRESHOLD = 4` — defined here, imported everywhere; never hardcode.
- `_DOMAIN_PAIR_PATHS` — add new cohort variants here with a `(data_dir, artifacts_dir)` tuple; bench scripts use `--domain-pair` CLI arg.
- `save_result()` — writes JSON to `artifacts/<domain-pair>/results/<model>_lesson<N>.json`. Must include a `"dataset_info"` block with `domain_pair`, `cohort_filter`, `n_users`, `n_movie_interactions`, `n_game_interactions`, `split`.
- `load_cross_domain_split()` — returns `CrossDomainSplit` with the `dataset_info` block pre-populated.

### Model interface

All models inherit from `BaseRecommender` (or `BasePyTorchRecommender`/`BaseLibraryRecommender`). Required method: `fit(ratings, user_to_idx, item_to_idx, **kwargs) -> dict[str, float]`. Predictions via `predict(user_idx, item_indices)`.

## Implementation rules

- **Lesson order**: implement and verify each lesson end-to-end before starting the next.
- **One file per model, one file per bench script**: `ml/models/<model>.py`, `ml/scripts/benchmarks/bench_<model>.py`.
- **All metrics are @10 only** — no @5, @20, @50 in any output.
- **Single eval entry point**: all bench scripts call `evaluate_cross_domain()` from `benchmark_common.py`; no inline eval logic.
- **recbole_cdr is vendored** at `ml/models/recbole_cdr/` — do not `pip install` it.
- **Venv Python**: `run_parallel_benchmarks.py` resolves `.venv/bin/python` automatically; bench scripts run under that interpreter.
