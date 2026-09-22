#!/usr/bin/env python3
"""Correlate positive-control scores with within-subject decoding per condition.

Examples
--------
  # Use existing CSVs (fast)
  python scripts/plot_poscontrol_vs_within.py \\
      --positive-csv results/eval/positive_control_positive_control_20260922_124312.csv \\
      --within-csv results/eval/within_subject_*.csv

  # Re-run both evals then plot
  python scripts/plot_poscontrol_vs_within.py \\
      --run-positive configs/eval/positive_control.yaml \\
      --run-within configs/eval/within_subject.yaml

Notes
-----
Positive control now writes ``f1`` (macro-F1) alongside ``auc``. Older CSVs
without ``f1`` fall back to AUC on the x-axis. Within-subject uses the
``metric`` column (macroF1 for discrete, AUC for binary).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_test import load_cfg, out_path  # noqa: E402
from src.eval import get_test  # noqa: E402
from src.eval.plot_poscontrol_vs_within import (  # noqa: E402
    load_positive_control,
    load_within_subject,
    merge_pos_within,
    plot_poscontrol_vs_within,
)
from src.eval.plot_within_subject import mark_rl_subset  # noqa: E402


def _latest_glob(pattern: str) -> Path | None:
    candidates = list(Path().glob(pattern)) + list(REPO_ROOT.glob(pattern))
    paths = sorted(
        {p.resolve() for p in candidates if p.is_file()},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None


def _resolve_csv(raw: str | None, default_globs: list[str]) -> Path | None:
    if raw:
        if any(ch in raw for ch in "*?["):
            return _latest_glob(raw)
        path = Path(raw)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path if path.exists() else None
    for g in default_globs:
        hit = _latest_glob(g)
        if hit is not None:
            return hit
    return None


def _run_cfg(cfg_path: Path) -> Path | None:
    cfg = load_cfg(cfg_path)
    print("\n" + "=" * 72)
    print(f"TEST: {cfg['test']}")
    print(f"CONFIG: {cfg_path}")
    print("=" * 72)
    df = get_test(cfg["test"])(cfg)
    if df is None or len(df) == 0:
        print("(no rows returned)")
        return None
    path = out_path(cfg, cfg_path)
    df.to_csv(path, index=False)
    print(f"saved → {path}")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Plot positive-control vs within-subject correlation by condition"
    )
    p.add_argument(
        "--run-positive",
        type=Path,
        default=None,
        help="Run positive_control YAML before plotting",
    )
    p.add_argument(
        "--run-within",
        type=Path,
        default=None,
        help="Run within_subject YAML before plotting",
    )
    p.add_argument(
        "--positive-csv",
        type=str,
        default=None,
        help="Positive-control results CSV (or glob). Default: latest positive_control_*.csv",
    )
    p.add_argument(
        "--within-csv",
        type=str,
        default=None,
        help="Within-subject results CSV (or glob). Default: latest within_subject_*.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "poscontrol_vs_within",
        help="Directory for figures",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="poscontrol_vs_within",
        help="Filename stem",
    )
    p.add_argument(
        "--rl-pids",
        type=str,
        default="10,11,18,19",
        help="Comma-separated RL participant IDs to highlight",
    )
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="Do not label points with participant IDs",
    )
    args = p.parse_args(argv)

    if args.run_positive is not None:
        cfg = args.run_positive if args.run_positive.is_absolute() else REPO_ROOT / args.run_positive
        if not cfg.exists():
            print(f"missing config: {cfg}", file=sys.stderr)
            return 1
        out = _run_cfg(cfg)
        if out is None:
            return 1
        args.positive_csv = str(out)

    if args.run_within is not None:
        cfg = args.run_within if args.run_within.is_absolute() else REPO_ROOT / args.run_within
        if not cfg.exists():
            print(f"missing config: {cfg}", file=sys.stderr)
            return 1
        out = _run_cfg(cfg)
        if out is None:
            return 1
        args.within_csv = str(out)

    pos_path = _resolve_csv(
        args.positive_csv,
        ["results/eval/positive_control_*.csv"],
    )
    within_path = _resolve_csv(
        args.within_csv,
        ["results/eval/within_subject_*.csv", "results/eval/*within_subject*.csv"],
    )
    if pos_path is None or not pos_path.exists():
        print(
            "Positive-control CSV not found. Pass --positive-csv or --run-positive.",
            file=sys.stderr,
        )
        return 1
    if within_path is None or not within_path.exists():
        print(
            "Within-subject CSV not found. Pass --within-csv or --run-within.",
            file=sys.stderr,
        )
        return 1

    print(f"positive-control: {pos_path}")
    print(f"within-subject:   {within_path}")

    pos = load_positive_control(pos_path)
    within = load_within_subject(within_path)
    rl_pids = [
        p.strip().zfill(3) if p.strip().isdigit() else p.strip()
        for p in (args.rl_pids or "").split(",")
        if p.strip()
    ]
    within = mark_rl_subset(within, rl_pids)
    merged = merge_pos_within(pos, within)
    if merged.empty:
        print("No overlapping participant IDs between the two CSVs.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.stem}_merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"merged table → {merged_path}  (n_rows={len(merged)}, n_pids={merged['pid'].nunique()})")
    print(f"x-axis: positive-control {merged['pos_metric_name'].iloc[0]}")
    print(f"y-axis: within-subject {merged['within_metric_name'].iloc[0]}")

    pdf = plot_poscontrol_vs_within(
        merged,
        args.out_dir / f"{args.stem}.pdf",
        annotate_pids=not args.no_annotate,
    )
    png = plot_poscontrol_vs_within(
        merged,
        args.out_dir / f"{args.stem}.png",
        annotate_pids=not args.no_annotate,
    )
    print("\nSaved figures:")
    print(f"  {pdf}")
    print(f"  {png}")

    # Print per-condition correlations for the terminal
    from scipy.stats import pearsonr, spearmanr

    print("\nPer-condition correlations:")
    for cond, sub in merged.groupby("condition", sort=False):
        x = sub["pos_metric"].to_numpy(float)
        y = sub["within_metric"].to_numpy(float)
        if len(x) < 3:
            print(f"  {cond}: n={len(x)} (too few)")
            continue
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        print(
            f"  {cond}: n={len(x)}  Pearson r={r_p:.3f} (p={p_p:.3g})  "
            f"Spearman ρ={r_s:.3f} (p={p_s:.3g})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
