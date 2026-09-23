#!/usr/bin/env python3
"""RW within-subject grid: window_s × temporal_shift.

Reports OOF AUC and macro-F1 (threshold 0 on LDA scores). Skips the circular
null so the grid finishes in one local run.

"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_test import load_cfg  # noqa: E402
from src.eval.cv import purged_oof  # noqa: E402
from src.eval.data import list_pids, pid_key  # noqa: E402
from src.eval.settings import channel_setup, cv_kwargs, win_kwargs  # noqa: E402
from src.eval.within_subject import _load_job  # noqa: E402

WINDOWS = [4.0, 6.0, 8.0]
SHIFTS = [0.0, 2.0, 4.0, 5.0, 6.0]


def _f1_from_scores(y, scores) -> float:
    pred = (np.asarray(scores) >= 0.0).astype(int)
    return float(f1_score(y, pred, average="macro"))


def main() -> int:
    cfg = load_cfg(REPO_ROOT / "configs/eval/within_subject.yaml")
    cfg["conditions"] = ["RW"]
    cfg["granularity"] = "binary"
    cfg["drop_suspect_channels"] = False
    cfg["step_s"] = 1.0
    cfg["embargo_s"] = 2.0
    cfg["n_null"] = 0

    processed = cfg["paths"]["processed"]
    labeled = cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    pids = list_pids(processed, labeled, "RW")
    print(f"RW candidates: {pids}")
    print(f"processed={processed}")
    print(f"labeled={labeled}")
    print(f"grid: windows={WINDOWS}  shifts={SHIFTS}  embargo={cfg['embargo_s']}")

    rows = []
    for window_s in WINDOWS:
        for shift in SHIFTS:
            cfg["window_s"] = window_s
            cfg["temporal_shift"] = shift
            win = win_kwargs(cfg)
            cv = cv_kwargs(cfg)
            print(f"\n== window={window_s} shift={shift} ==")
            for pid in pids:
                built = _load_job(pid, ["RW"], processed, labeled, channels, pairs, win)
                if built is None:
                    print(f"  skip {pid_key(pid)}: no windows")
                    continue
                F, y, T = built
                auc, pred, blk, ok = purged_oof(F, y, T, **cv)
                if pred is None or ok is None or not np.isfinite(auc):
                    print(f"  skip {pid_key(pid)}: non-finite metric")
                    continue
                y_ok, s_ok = y[ok], pred[ok]
                f1 = _f1_from_scores(y_ok, s_ok)
                try:
                    auc_chk = float(roc_auc_score(y_ok, s_ok))
                except Exception:
                    auc_chk = float(auc)
                row = dict(
                    pid=pid_key(pid),
                    condition="RW",
                    window_s=window_s,
                    temporal_shift=shift,
                    embargo_s=cfg["embargo_s"],
                    n=int(len(y)),
                    n_oof=int(ok.sum()),
                    frac1=float(np.mean(y)),
                    auc=float(auc_chk),
                    macro_f1=f1,
                    channels=ch_tag,
                )
                rows.append(row)
                print(
                    f"  {row['pid']} n={row['n']:4d} AUC={row['auc']:.3f} "
                    f"macroF1={row['macro_f1']:.3f} frac1={row['frac1']:.2f}"
                )

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"rw_window_shift_sweep_{stamp}.csv"
    df.to_csv(path, index=False)
    print(f"\nsaved → {path}")

    if df.empty:
        print("No rows.")
        return 1

    print("\n=== Per-participant stability across the grid ===")
    summary = (
        df.groupby("pid")
        .agg(
            n_settings=("macro_f1", "size"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            f1_min=("macro_f1", "min"),
            f1_max=("macro_f1", "max"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_min=("auc", "min"),
            n_above_f1_40=("macro_f1", lambda s: int((s >= 0.40).sum())),
            n_above_auc_60=("auc", lambda s: int((s >= 0.60).sum())),
        )
        .sort_values(["f1_mean", "auc_mean"], ascending=False)
    )
    print(summary.round(3).to_string())

    print("\n=== Mean metric by window × shift ===")
    grid = df.groupby(["window_s", "temporal_shift"])[["auc", "macro_f1"]].mean().unstack()
    print(grid.round(3).to_string())

    best = df.groupby(["window_s", "temporal_shift"])["auc"].mean().idxmax()
    print(f"\nBest mean-AUC setting: window={best[0]}  shift={best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
