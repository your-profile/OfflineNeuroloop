"""Within-participant purged CV + circular-shift null."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import block_bootstrap_ci, circular_null_p, purged_oof
from .data import list_pids, load_aligned, pid_key
from .features import build_windows, concat_runs
from .settings import channel_setup, cv_kwargs, expand_jobs, normalize_granularity, win_kwargs


def _load_job(pid, conds, processed, labeled, channels, pairs, win):
    runs = []
    for cond in conds:
        raw = load_aligned(
            pid, cond, processed, labeled, channels=channels, granularity=win["granularity"]
        )
        if raw is None:
            continue
        built = build_windows(*raw, channels=channels, pairs=pairs, **win)
        if built is not None:
            runs.append(built)
    if not runs:
        return None
    if len(runs) == 1:
        return runs[0]
    return concat_runs(runs)


def run(cfg: dict) -> pd.DataFrame:
    processed = cfg["paths"]["processed"]
    labeled = cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    win = win_kwargs(cfg)
    cv = cv_kwargs(cfg)
    n_null = cfg.get("n_null", 200)
    g = normalize_granularity(cfg.get("granularity", "binary"))
    metric_name = {"binary": "AUC", "discrete": "macroF1", "continuous": "Spearman"}[g]

    rows = []
    for job, conds in expand_jobs(cfg):
        # pids that have ALL conditions in the job (for pooled: both W and P)
        pid_sets = [set(list_pids(processed, labeled, c)) for c in conds]
        pids = sorted(set.intersection(*pid_sets)) if pid_sets else []
        print(f"\n== {job} [{g}] ({'+'.join(conds)}): {len(pids)} candidates ==")
        for pid in pids:
            built = _load_job(pid, conds, processed, labeled, channels, pairs, win)
            if built is None:
                continue
            F, y, T = built
            metric, pred, blk, ok = purged_oof(F, y, T, **cv)
            if not np.isfinite(metric):
                continue
            lo, hi = block_bootstrap_ci(y, pred, blk, ok, granularity=g)
            seed = hash(f"{pid_key(pid)}{job}{g}") % (2**31)
            null, p = circular_null_p(F, y, T, metric, n_null=n_null, seed=seed, **cv)
            row = dict(
                job=job,
                conds="+".join(conds),
                granularity=g,
                pid=pid_key(pid),
                n=len(y),
                metric=float(metric),
                lo=lo,
                hi=hi,
                null=null,
                p=p,
                channels=ch_tag,
            )
            if g == "binary":
                row["frac1"] = float(y.mean())
            elif g == "discrete":
                for c, cnt in zip(*np.unique(y, return_counts=True)):
                    row[f"frac_{int(c)}"] = float(cnt / len(y))
            else:
                row["y_mean"] = float(y.mean())
                row["y_std"] = float(y.std())
            rows.append(row)
            mark = " *" if p < 0.05 else ""
            print(
                f"  {row['pid']} n={row['n']:4d} {metric_name}={row['metric']:.3f} "
                f"null={row['null']:.3f} p={row['p']:.3f}{mark}"
            )

    df = pd.DataFrame(rows)
    if len(df):
        print(
            f"\n>>> mean {metric_name}={df.metric.mean():.3f}  mean null={df.null.mean():.3f}  "
            f"sig={int((df.p < 0.05).sum())}/{len(df)}"
        )
    return df
