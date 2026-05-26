#!/usr/bin/env python3
"""
Quick smoke-test runner for your MacBook.

Runs a tiny sweep (default: 3 trials — one per integration method) with few
episodes. Edit the QUICK_* settings below or pass --profile macbook to use
configs/sweep_local.yaml.

Usage:
  python run_local_test.py
  python run_local_test.py --dry-run          # print trials only
  python run_local_test.py --integrations finetune --conditions Baseline-PER
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_sweep import (
    REPO_ROOT,
    cfg_from_spec,
    iter_trial_specs,
    load_sweep,
    make_run_name,
    should_skip_spec,
    write_manifest,
)

# --- edit these for ad-hoc local tests without changing YAML ---
QUICK_INTEGRATIONS = ["pretrain", "pretrain", "finetune"]
QUICK_CONDITIONS = ["Baseline-PER"]
QUICK_SEEDS = [42]
QUICK_DOMAIN_CONFIG = "configs/domains/robot/1.yaml"
QUICK_TASK = "Passive"
QUICK_N_EPISODES = 45
QUICK_DATA_PATH = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData/"
QUICK_RESULTS_PATH = str(REPO_ROOT)


def build_quick_sweep(args: argparse.Namespace) -> dict:
    if args.use_yaml:
        return load_sweep(REPO_ROOT / "configs/sweep_local.yaml", profile=args.profile)
    return {
        "base_config": "configs/test.yaml",
        "paths": {
            "data_path": args.data_path or QUICK_DATA_PATH,
            "results_path": args.results_path or QUICK_RESULTS_PATH,
        },
        "integrations": args.integrations or QUICK_INTEGRATIONS,
        "seeds": args.seeds or QUICK_SEEDS,
        "conditions": args.conditions or QUICK_CONDITIONS,
        "granularities": ["binary"],
        "domains": [
            {
                "domain_config": args.domain_config or QUICK_DOMAIN_CONFIG,
                "tasks": [args.task or QUICK_TASK],
            }
        ],
        "ablations": [{"key": ["experiment", "eval_success_threshold"], "vals": [0.0]}],
        "n_episodes_override": args.n_episodes if args.n_episodes is not None else QUICK_N_EPISODES,
    }


def collect_specs(sweep: dict) -> list[dict]:
    specs = iter_trial_specs(sweep)
    valid = []
    for spec in specs:
        if should_skip_spec(spec):
            continue
        spec["run_name"] = make_run_name(cfg_from_spec(spec))
        valid.append(spec)
    for i, spec in enumerate(valid, start=1):
        spec["trial_id"] = i
    return valid


def run_spec(spec: dict, verbose: bool) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_trial.py"),
        "--manifest",
        str(REPO_ROOT / "_local_test_manifest.csv"),
        "--trial-id",
        str(spec["trial_id"]),
    ]
    if verbose:
        cmd.append("--verbose")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Small local experiment smoke tests.")
    parser.add_argument("--use-yaml", action="store_true", help="Use configs/sweep_local.yaml")
    parser.add_argument("-p", "--profile", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="List trials without running")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--integrations", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--domain-config", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--n-episodes", type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-path", type=str)
    args = parser.parse_args()

    sweep = build_quick_sweep(args)
    specs = collect_specs(sweep)
    manifest_path = REPO_ROOT / "_local_test_manifest.csv"
    write_manifest(specs, manifest_path)

    print(f"Local test plan: {len(specs)} trial(s)")
    for spec in specs:
        print(
            f"  [{spec['trial_id']}] {spec['integration']} | {spec['condition']} | "
            f"seed={spec['seed']} | {spec.get('run_name', '')}"
        )

    if args.dry_run:
        print("Dry run — no experiments started.")
        return

    for spec in specs:
        print(f"\n--- Trial {spec['trial_id']}/{len(specs)} ---")
        run_spec(spec, verbose=args.verbose)

    print(f"\nDone. Per-trial CSVs: {REPO_ROOT / 'src/results/runs/'}")
    print("Merge with: python merge_results.py")


if __name__ == "__main__":
    main()
