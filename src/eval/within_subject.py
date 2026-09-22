"""Within-participant purged CV + circular-shift null.

Runs one decoder per participant (optionally pooling multiple conditions in a
job). Use ``participant_list`` to restrict who is evaluated and
``rl_participant_list`` to mark the subset used in the RL loop for figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cv import block_bootstrap_ci, circular_null_p, purged_oof
from .data import list_pids, load_aligned, pid_key
from .features import build_windows, concat_runs
from .settings import channel_setup, cv_kwargs, expand_jobs, normalize_granularity, win_kwargs


def _domain_from_cond(cond: str) -> str:
    return cond[0] if cond else ""


def _task_from_cond(cond: str) -> str:
    if not cond:
        return ""
    return "passive" if cond.endswith("W") else ("active" if cond.endswith("P") else "")


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


def _resolve_pid_filter(cfg: dict) -> list[str] | None:
    raw = cfg.get("participant_list") or cfg.get("focus_pids") or cfg.get("pids")
    if not raw:
        return None
    return [pid_key(p) for p in raw]


def _resolve_rl_pids(cfg: dict) -> set[str]:
    raw = cfg.get("rl_participant_list") or cfg.get("rl_pids")
    if not raw:
        return set()
    return {pid_key(p) for p in raw}


def run(cfg: dict) -> pd.DataFrame:
    processed = cfg["paths"]["processed"]
    labeled = cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    win = win_kwargs(cfg)
    cv = cv_kwargs(cfg)
    n_null = cfg.get("n_null", 200)
    g = normalize_granularity(cfg.get("granularity", "binary"))
    metric_name = {"binary": "AUC", "discrete": "macroF1", "continuous": "Spearman"}[g]
    pid_filter = _resolve_pid_filter(cfg)
    rl_pids = _resolve_rl_pids(cfg)

    rows = []
    for job, conds in expand_jobs(cfg):
        # pids that have ALL conditions in the job (for pooled: both W and P)
        pid_sets = [set(list_pids(processed, labeled, c)) for c in conds]
        pids = sorted(set.intersection(*pid_sets)) if pid_sets else []
        if pid_filter is not None:
            pids = [p for p in pids if pid_key(p) in pid_filter]
        print(f"\n== {job} [{g}] ({'+'.join(conds)}): {len(pids)} candidates ==")
        if pid_filter is not None:
            print(f"   participant_list filter → {pids}")
        for pid in pids:
            built = _load_job(pid, conds, processed, labeled, channels, pairs, win)
            if built is None:
                print(f"  skip {pid_key(pid)}: no windows")
                continue
            F, y, T = built
            metric, pred, blk, ok = purged_oof(F, y, T, **cv)
            if not np.isfinite(metric):
                print(f"  skip {pid_key(pid)}: non-finite metric")
                continue
            lo, hi = block_bootstrap_ci(y, pred, blk, ok, granularity=g)
            seed = hash(f"{pid_key(pid)}{job}{g}") % (2**31)
            null, p = circular_null_p(F, y, T, metric, n_null=n_null, seed=seed, **cv)
            pk = pid_key(pid)
            primary_cond = conds[0] if len(conds) == 1 else "+".join(conds)
            row = dict(
                model="within_subject",
                job=job,
                conds="+".join(conds),
                condition=primary_cond,
                domain=_domain_from_cond(conds[0]) if len(conds) == 1 else "mixed",
                task=_task_from_cond(conds[0]) if len(conds) == 1 else "pooled",
                granularity=g,
                metric_name=metric_name,
                pid=pk,
                n=len(y),
                metric=float(metric),
                lo=lo,
                hi=hi,
                null=null,
                p=p,
                significant=bool(p < 0.05),
                in_rl_subset=bool(pk in rl_pids) if rl_pids else False,
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
            rl_tag = " [RL]" if row["in_rl_subset"] else ""
            print(
                f"  {row['pid']}{rl_tag} n={row['n']:4d} {metric_name}={row['metric']:.3f} "
                f"null={row['null']:.3f} p={row['p']:.3f}{mark}"
            )

    df = pd.DataFrame(rows)
    if len(df):
        print(
            f"\n>>> mean {metric_name}={df.metric.mean():.3f}  mean null={df.null.mean():.3f}  "
            f"sig={int((df.p < 0.05).sum())}/{len(df)}"
        )
        if rl_pids and "in_rl_subset" in df.columns:
            sub = df[df.in_rl_subset]
            rest = df[~df.in_rl_subset]
            if len(sub) and len(rest):
                print(
                    f">>> RL subset mean={sub.metric.mean():.3f} (n={len(sub)})  "
                    f"others mean={rest.metric.mean():.3f} (n={len(rest)})"
                )
    return df
