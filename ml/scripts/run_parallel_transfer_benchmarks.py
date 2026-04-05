#!/usr/bin/env python3
"""Run selected benchmarks in parallel for transfer-focused datasets.

Example (OverlapTransfer-Loose, default models):

    PYTHONPATH=. python3 ml/scripts/run_parallel_transfer_benchmarks.py \\
        --domain-pair movie_game_transfer_loose

Monitor (separate terminals or tmux panes):

    tail -f artifacts_transfer_loose/benchmark_logs/lightgcn.log
    tail -f artifacts_transfer_loose/benchmark_logs/*.log

Or watch all:

    tail -f artifacts_transfer_loose/benchmark_logs/*.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "ml" / "scripts" / "benchmarks"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)

# Maps friendly name → (script, extra CLI args after --domain-pair)
TRANSFER_BENCHMARK_MODELS: dict[str, tuple[str, list[str]]] = {
    "lightgcn": ("bench_lightgcn.py", []),
    "bitgcf": ("bench_bi_tgcf.py", []),
    "ncf": ("bench_ncf.py", []),
    "mf_bpr": ("bench_mf_bpr.py", []),
    "mf_explicit": ("bench_mf_explicit.py", []),
    # Target-train profile only (same as classic bench_sbert).
    "sbert": ("bench_sbert.py", []),
    # Source+target blend for transfer (bench_sbert_cdr.py).
    "sbert_cdr": ("bench_sbert_cdr.py", []),
    "deepapf": ("bench_deepapf.py", []),
    "emcdr": ("bench_emcdr.py", []),
    "sscdr": ("bench_sscdr.py", []),
    "cmf": ("bench_cmf.py", []),
    "ptupcdr": ("bench_ptupcdr.py", []),
}


def _artifacts_root(domain_pair: str) -> Path:
    _ROOTS = {
        "movie_game_transfer_loose": PROJECT_ROOT / "artifacts_transfer_loose",
        "movie_game_transfer_strict": PROJECT_ROOT / "artifacts_transfer_strict",
        "movie_game_transfer_loose_filtered": PROJECT_ROOT / "artifacts_transfer_loose_filtered",
    }
    if domain_pair in _ROOTS:
        return _ROOTS[domain_pair]
    raise ValueError(
        f"Use a transfer domain_pair (got {domain_pair!r}); "
        f"choose from: {', '.join(_ROOTS)}"
    )


def run_one(
    name: str,
    script: str,
    extra_args: list[str],
    domain_pair: str,
    log_dir: Path,
    env: dict[str, str],
) -> tuple[str, int, float]:
    """Run one benchmark; stream log to file. Returns (name, returncode, seconds)."""
    script_path = BENCHMARKS_DIR / script
    log_path = log_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(VENV_PYTHON),
        str(script_path),
        "--domain-pair",
        domain_pair,
        *extra_args,
    ]

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(BENCHMARKS_DIR),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - t0
    return name, proc.returncode, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run transfer-dataset benchmarks in parallel with per-model logs."
    )
    parser.add_argument(
        "--domain-pair",
        default="movie_game_transfer_loose",
        help="Processed data + artifact root (default: movie_game_transfer_loose)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(TRANSFER_BENCHMARK_MODELS.keys()),
        help=f"Subset of: {', '.join(TRANSFER_BENCHMARK_MODELS)}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel subprocesses (default: 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    args = parser.parse_args()

    domain_pair = args.domain_pair
    artifacts = _artifacts_root(domain_pair)
    log_dir = artifacts / "benchmark_logs"

    for m in args.models:
        if m not in TRANSFER_BENCHMARK_MODELS:
            raise SystemExit(f"Unknown model {m!r}. Choose from {list(TRANSFER_BENCHMARK_MODELS)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{BENCHMARKS_DIR}"
    env["PYTHONUNBUFFERED"] = "1"

    jobs = [
        (
            name,
            TRANSFER_BENCHMARK_MODELS[name][0],
            list(TRANSFER_BENCHMARK_MODELS[name][1]),
        )
        for name in args.models
    ]

    print("Transfer parallel benchmarks")
    print(f"  domain_pair:  {domain_pair}")
    print(f"  artifacts:    {artifacts}")
    print(f"  logs:         {log_dir}")
    print(f"  models:       {[j[0] for j in jobs]}")
    print(f"  max_workers:  {args.max_workers}")
    print()
    print("Monitor:")
    print(f"  tail -f {log_dir}/*.log")
    print()

    if args.dry_run:
        for name, script, extra in jobs:
            cmd = [
                str(VENV_PYTHON),
                str(BENCHMARKS_DIR / script),
                "--domain-pair",
                domain_pair,
                *extra,
            ]
            print(" ".join(cmd))
        return

    results: list[tuple[str, int, float]] = []
    workers = min(args.max_workers, len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(run_one, name, script, extra, domain_pair, log_dir, env): name
            for name, script, extra in jobs
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                n, code, sec = fut.result()
                results.append((n, code, sec))
                status = "OK" if code == 0 else f"FAIL({code})"
                print(f"[{status}] {n}  {sec:.1f}s  →  {log_dir / (n + '.log')}")
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                results.append((name, -1, 0.0))

    print()
    failed = [r for r in results if r[1] != 0]
    if failed:
        print(f"Failed ({len(failed)}): {[f[0] for f in failed]}")
        sys.exit(1)
    print("All benchmarks finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
