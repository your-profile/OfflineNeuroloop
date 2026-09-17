"""Train on a focus set, test on other participants (same job / granularity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import fit_score
from .data import list_pids, load_aligned, pid_key
from .features import build_windows, concat_runs
from .settings import channel_setup, expand_jobs, normalize_granularity, win_kwargs


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
    out = runs[0] if len(runs) == 1 else concat_runs(runs)
    return out[0], out[1]


def run(cfg: dict) -> pd.DataFrame:
    focus = [pid_key(p) for p in cfg.get("focus_pids", ["010", "011", "018", "019"])]
    processed, labeled = cfg["paths"]["processed"], cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    win = win_kwargs(cfg)
    g = normalize_granularity(cfg.get("granularity", "binary"))

    # single job: prefer explicit condition, else first expand_jobs entry
    if cfg.get("condition"):
        jobs = [(cfg["condition"], [cfg["condition"]])]
    else:
        jobs = expand_jobs(cfg)
        if len(jobs) != 1:
            # default focus transfer is one domain/condition
            jobs = jobs[:1]
            print(f"focus_transfer using first job only: {jobs[0][0]}")

    rows = []
    for job, conds in jobs:
        pid_sets = [set(list_pids(processed, labeled, c)) for c in conds]
        pids = sorted(set.intersection(*pid_sets)) if pid_sets else []
        data = {}
        for pid in pids:
            built = _load_job(pid, conds, processed, labeled, channels, pairs, win)
            if built is not None:
                data[pid_key(pid)] = built

        missing = [p for p in focus if p not in data]
        if missing:
            raise RuntimeError(f"focus pids missing usable windows for {job}: {missing}")

        Xtr = np.vstack([data[p][0] for p in focus])
        ytr = np.concatenate([data[p][1] for p in focus])
        print(f"train on {focus} ({len(Xtr)} windows) → test other {job} [{g}]")

        for held in sorted(data):
            if held in focus:
                continue
            Xte, yte = data[held]
            if g != "continuous" and len(np.unique(yte)) < 2:
                continue
            primary, f1 = fit_score(Xtr, ytr, Xte, yte, granularity=g)
            rows.append(
                dict(
                    job=job,
                    conds="+".join(conds),
                    granularity=g,
                    held=held,
                    n_te=len(yte),
                    metric=float(primary) if np.isfinite(primary) else np.nan,
                    f1=float(f1) if np.isfinite(f1) else np.nan,
                    channels=ch_tag,
                )
            )
            print(f"  test {held}: metric={primary:.3f} F1={f1 if np.isfinite(f1) else float('nan'):.3f}")

    df = pd.DataFrame(rows)
    if len(df):
        print(f"mean metric={df.metric.mean():.3f}")
    return df
