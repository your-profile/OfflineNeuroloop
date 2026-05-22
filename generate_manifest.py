#!/usr/bin/env python3
"""Build trial_manifest.csv for SLURM job arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiment_sweep import REPO_ROOT, generate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sweep",
        type=Path,
        default=REPO_ROOT / "configs/sweep_hpc.yaml",
        help="Sweep definition YAML",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "trial_manifest.csv",
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="Optional profile name under profiles: in the sweep file (e.g. macbook)",
    )
    args = parser.parse_args()

    n = generate_manifest(args.sweep, args.output, profile=args.profile)
    print(f"Wrote {n} trials to {args.output}")


if __name__ == "__main__":
    main()
