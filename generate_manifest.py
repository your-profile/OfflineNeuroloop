#!/usr/bin/env python3
"""Build trial manifest CSV(s) for SLURM job arrays."""
from __future__ import annotations

import argparse
from pathlib import Path


"""
Example:

python generate_manifest.py -s configs/sweep_hpc_PER.yaml --filter-domain Flappy --filter-integration pretrain --filter-granularity binary -o manifests/flappy_pretrain_binary_PER.csv

"""

from experiment_sweep import (
    REPO_ROOT,
    generate_manifest,
    generate_manifest_shards,
    load_sweep,
    iter_trial_specs,
    filter_specs,
    finalize_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a sweep YAML into one or more manifest CSV files."
    )
    parser.add_argument(
        "-s",
        "--sweep",
        type=Path,
        default=REPO_ROOT / "configs/sweep_hpc_PER.yaml",
        help="Sweep definition YAML",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "configs/manifests/trial_manifest.csv",
        help="Output manifest for a single filtered sweep",
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="Optional profile name under profiles: in the sweep file",
    )
    parser.add_argument(
        "--shard-by",
        type=str,
        choices=[
            "domain",
            "integration",
            "ablation",
            "domain_integration",
            "domain_integration_ablation",
            "domain_integration_ablation_granularity",
        ],
        help="Write manifests/manifests/<shard>.csv instead of one file",
    )
    parser.add_argument("--filter-domain", nargs="+", metavar="NAME")
    parser.add_argument("--filter-integration", nargs="+", choices=["pretrain", "finetune", "interleaved"])
    parser.add_argument("--filter-ablation-key", nargs="+", help="e.g. mlp.model_noise neural.beta")
    parser.add_argument("--filter-condition", nargs="+")
    parser.add_argument("--filter-task", nargs="+")
    parser.add_argument(
        "--filter-granularity",
        nargs="+",
        choices=["binary", "ternary", "continuous"],
        help="e.g. --filter-granularity binary ternary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial counts only",
    )
    args = parser.parse_args()
    filters = dict(
        domains=args.filter_domain,
        integrations=args.filter_integration,
        ablation_keys=args.filter_ablation_key,
        conditions=args.filter_condition,
        tasks=args.filter_task,
        granularities=args.filter_granularity,
    )

    if args.shard_by:
        out_dir = REPO_ROOT / "manifests"
        counts = generate_manifest_shards(
            args.sweep,
            out_dir,
            shard_by=args.shard_by,
            profile=args.profile,
            **filters,
        )
        total = sum(counts.values())
        print(f"Wrote {len(counts)} shard manifest(s) under {out_dir} ({total} trials total)")
        for name, n in sorted(counts.items()):
            print(f"  {name}.csv: {n} trials")
        return

    if args.dry_run:
        sweep = load_sweep(args.sweep, profile=args.profile)
        specs = filter_specs(iter_trial_specs(sweep), **filters)
        valid = finalize_specs(specs)
        print(f"Would write {len(valid)} trials to {args.output}")
        return

    n = generate_manifest(
        args.sweep,
        args.output,
        profile=args.profile,
        **filters,
    )
    print(f"Wrote {n} trials to {args.output}")


if __name__ == "__main__":
    main()
