#!/usr/bin/env python3
"""Run a single experiment trial (for SLURM arrays or local debugging)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from experiment_sweep import (
    REPO_ROOT,
    cfg_from_spec,
    make_run_name,
    read_manifest_row,
    trial_results_filename,
    trial_results_path,
)
from trial import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one trial from the manifest or CLI.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "trial_manifest.csv",
        help="CSV manifest from generate_manifest.py",
    )
    parser.add_argument("--trial-id", type=int, help="1-based trial_id in the manifest")
    parser.add_argument("--integration", choices=["finetune", "interleave", "pretrain"])
    parser.add_argument("--condition", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--granularity", default="binary")
    parser.add_argument("--task", default="Pooled")
    parser.add_argument("--domain-key", type=str, help="e.g. Flappy (uses configs/domains/flappy.yaml)")
    parser.add_argument("--domain-config", type=Path, help="Full domain YAML, e.g. configs/domains/flappy.yaml")
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs/test.yaml")
    parser.add_argument("--ablation-key", default="experiment.finetune_threshold")
    parser.add_argument("--ablation-val", default="0.0")
    parser.add_argument("--n-episodes", type=int, help="Override rl.n_episodes for quick tests")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--results-path", type=Path, default=REPO_ROOT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Exit 0 if this trial's results CSV already exists",
    )
    args = parser.parse_args()

    if args.trial_id is not None:
        spec = read_manifest_row(args.manifest, args.trial_id)

        if os.environ.get("NEUROLOOP_DATA_ROOT"):
            spec = {**spec, "data_path": os.environ["NEUROLOOP_DATA_ROOT"]}
        if os.environ.get("NEUROLOOP_RESULTS_ROOT"):
            spec = {**spec, "results_path": os.environ["NEUROLOOP_RESULTS_ROOT"]}

        if args.skip_if_done and trial_results_path(spec).is_file():
            print(f"Skip trial_id={args.trial_id}: exists {trial_results_path(spec)}")
            return
        cfg = cfg_from_spec(spec)
        if cfg is None:
            print(f"Trial {args.trial_id} skipped by ablation rules.", file=sys.stderr)
            sys.exit(0)
        data_path = spec["data_path"] or str(REPO_ROOT)
        results_path = spec["results_path"] or str(REPO_ROOT)
        trial_id = int(spec["trial_id"])
        integration = spec["integration"]
        run_name = spec.get("run_name") or make_run_name(cfg)
        verbose = cfg["experiment"].get("verbose", False)
    else:
        if not args.integration or not args.condition or args.seed is None:
            parser.error("Without --trial-id, pass --integration, --condition, and --seed")
        ablation = {"key": args.ablation_key.split(".")}
        from experiment_sweep import build_cfg

        cfg = build_cfg(
            integration=args.integration,
            condition=args.condition,
            seed=args.seed,
            granularity=args.granularity,
            task=args.task,
            ablation=ablation,
            ablation_val=_coerce(args.ablation_val),
            base_config=args.base_config,
            domain_key=args.domain_key,
            domain_config=args.domain_config,
            n_episodes_override=args.n_episodes,
        )
        if cfg is None:
            print("Trial skipped by ablation rules.", file=sys.stderr)
            sys.exit(0)
        data_path = str(args.data_path or REPO_ROOT)
        results_path = str(args.results_path)
        trial_id = 0
        integration = args.integration
        run_name = make_run_name(cfg)
        verbose = args.verbose or cfg["experiment"].get("verbose", False)

    if args.verbose:
        cfg["experiment"]["verbose"] = True
        verbose = True

    results_file = trial_results_filename(trial_id, integration)
    print(f"Running trial_id={trial_id} integration={integration} -> {results_file}")
    run(
        cfg,
        run_name=run_name,
        DATA_PATH=data_path,
        RESULTS_PATH=results_path,
        RESULTS_FILE_NAME=results_file,
        verbose=verbose,
        inverse=False,
    )


def _coerce(raw: str):
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


if __name__ == "__main__":
    main()
