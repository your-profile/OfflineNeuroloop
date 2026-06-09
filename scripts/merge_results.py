#!/usr/bin/env python3
"""Merge per-trial CSVs under src/results/runs/ into combined result files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_sweep import INTEGRATION_RESULTS_SUFFIX, REPO_ROOT


"""
Example:

python merge_results.py --runs-dir src/results/runs --output-dir src/results --combined_name lunar_trial_results_merged.csv
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "src/results/runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "src/results",
    )
    parser.add_argument(
        "--combined-name",
        default="trial_results_merged.csv",
        help="Single file with all integrations",
    )
    parser.add_argument(
        "--by-integration",
        action="store_true",
        default=True,
        help="Also write trial_results_{finetuning,interleaving,pretraining}.csv",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.exists():
        raise SystemExit(f"No runs directory: {runs_dir}")

    files = sorted(runs_dir.glob("trial_*.csv"))
    if not files:
        raise SystemExit(f"No trial CSV files in {runs_dir}")

    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = args.output_dir / args.combined_name
    combined.to_csv(combined_path, index=False)
    print(f"Merged {len(files)} files -> {combined_path}")

    if args.by_integration:
        for integration, suffix in INTEGRATION_RESULTS_SUFFIX.items():
            subset = [f for f in files if f"_{suffix}.csv" in f.name]
            if not subset:
                continue
            df = pd.concat([pd.read_csv(f) for f in subset], ignore_index=True)
            out = args.output_dir / f"trial_results_{suffix}.csv"
            df.to_csv(out, index=False)
            print(f"  {integration}: {len(subset)} trials -> {out}")


if __name__ == "__main__":
    main()
