#!/usr/bin/env python3
"""Run within-subject / LOPO eval and build paper figures.

Examples
--------
  # Run paper within-subject suite + figures
  python scripts/plot_within_subject_results.py --run configs/eval/within_subject_paper.yaml

  # Robot Passive / Active (separate participant cohorts)
  python scripts/plot_within_subject_results.py --run configs/eval/within_subject_robot_passive.yaml
  python scripts/plot_within_subject_results.py --run configs/eval/within_subject_robot_active.yaml --rl-vs-condition RP

  # Plot an existing CSV (no re-train)
  python scripts/plot_within_subject_results.py \\
      --csv results/eval/within_subject_binary_within_subject_paper_YYYYMMDD_HHMMSS.csv \\
      --rl-pids 10,11,18,19

  # Within-subject + multi-subject LOPO
  python scripts/plot_within_subject_results.py \\
      --run configs/eval/within_subject_paper.yaml \\
      --run configs/eval/lopo_paper.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_test import load_cfg, out_path  # noqa: E402
from src.eval import get_test  # noqa: E402
from src.eval.plot_within_subject import mark_rl_subset, plot_all  # noqa: E402


def _parse_pids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip().zfill(3) if p.strip().isdigit() else p.strip() for p in raw.split(",") if p.strip()]


def _latest_glob(pattern: str) -> Path | None:
    candidates = list(Path().glob(pattern)) + list(REPO_ROOT.glob(pattern))
    paths = sorted({p.resolve() for p in candidates if p.is_file()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _run_cfg(cfg_path: Path, rl_pids: list[str]) -> Path | None:
    cfg = load_cfg(cfg_path)
    if rl_pids and not cfg.get("rl_participant_list"):
        cfg["rl_participant_list"] = [int(p) if p.isdigit() else p for p in rl_pids]
    print("\n" + "=" * 72)
    print(f"TEST: {cfg['test']}")
    print(f"CONFIG: {cfg_path}")
    print(
        f"granularity: {cfg.get('granularity', 'binary')}  "
        f"conditions: {cfg.get('conditions', cfg.get('condition_groups', cfg.get('domains', '-')))}"
    )
    print(f"processed: {cfg['paths']['processed']}")
    print("=" * 72)
    df = get_test(cfg["test"])(cfg)
    if df is None or len(df) == 0:
        print("(no rows returned)")
        return None
    path = out_path(cfg, cfg_path)
    df.to_csv(path, index=False)
    print(f"\nsaved → {path}")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Within-subject eval + paper figures")
    p.add_argument(
        "--run",
        action="append",
        type=Path,
        default=[],
        help="Eval YAML to run (repeatable). Results CSV is saved under results/eval/",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Existing results CSV (or glob). Used for figures.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "within_subject",
        help="Directory for PDF/PNG figures",
    )
    p.add_argument(
        "--rl-pids",
        type=str,
        default="10,11,18,19",
        help="Comma-separated participant IDs used in the RL loop (highlighted)",
    )
    p.add_argument(
        "--rl-vs-condition",
        type=str,
        default="RW",
        help="Condition for the RL-subset vs others panel (default: RW)",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="within_subject",
        help="Filename stem for figures",
    )
    args = p.parse_args(argv)

    rl_pids = _parse_pids(args.rl_pids)
    saved_csvs: list[Path] = []

    for cfg_path in args.run:
        cfg_path = cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path
        if not cfg_path.exists():
            print(f"missing config: {cfg_path}", file=sys.stderr)
            return 1
        out = _run_cfg(cfg_path, rl_pids)
        if out is not None:
            saved_csvs.append(out)

    csv_path = None
    if args.csv:
        raw = args.csv
        if any(ch in raw for ch in "*?["):
            csv_path = _latest_glob(raw)
        else:
            csv_path = Path(raw)
            if not csv_path.is_absolute():
                csv_path = (REPO_ROOT / csv_path).resolve()
        if csv_path is None or not csv_path.exists():
            print(f"CSV not found: {args.csv}", file=sys.stderr)
            return 1
    elif saved_csvs:
        within = [p for p in saved_csvs if "within_subject" in p.name]
        csv_path = within[0] if within else saved_csvs[0]

    if csv_path is None:
        print("Nothing to plot. Pass --run and/or --csv.", file=sys.stderr)
        return 1

    print(f"\nFigures from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = mark_rl_subset(df, rl_pids)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    marked = args.out_dir / f"{args.stem}_results.csv"
    df.to_csv(marked, index=False)
    print(f"marked results → {marked}")

    paths = plot_all(
        df,
        args.out_dir,
        stem=args.stem,
        rl_pids=rl_pids,
        rl_vs_condition=args.rl_vs_condition,
    )
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
