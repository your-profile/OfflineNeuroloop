"""Leave-one-participant-out multi-subject decoding."""

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
    return out[0], out[1]  # F, y


def run(cfg: dict) -> pd.DataFrame:
    processed, labeled = cfg["paths"]["processed"], cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    win = win_kwargs(cfg)
    n_null = cfg.get("n_null", 100)
    g = normalize_granularity(cfg.get("granularity", "binary"))
    metric_name = {"binary": "AUC", "discrete": "macroF1", "continuous": "Spearman"}[g]
    focus = cfg.get("participant_list") or cfg.get("focus_pids") or cfg.get("pids")
    if focus:
        focus = [pid_key(p) for p in focus]
    rl_raw = cfg.get("rl_participant_list") or cfg.get("rl_pids") or []
    rl_pids = {pid_key(p) for p in rl_raw}

    rows = []
    for job, conds in expand_jobs(cfg):
        pid_sets = [set(list_pids(processed, labeled, c)) for c in conds]
        pids = sorted(set.intersection(*pid_sets)) if pid_sets else []
        data = {}
        for pid in pids:
            built = _load_job(pid, conds, processed, labeled, channels, pairs, win)
            if built is not None:
                data[pid_key(pid)] = built

        pool = sorted(data)
        if focus:
            pool = [p for p in pool if p in focus]
        if len(pool) < 3:
            print(f"\n== {job}: skip (n={len(pool)}) ==")
            continue
        print(f"\n== {job} [{g}]: LOPO multi-subject over {pool} ==")
        for held in pool:
            others = [q for q in pool if q != held]
            Xtr = np.vstack([data[q][0] for q in others])
            ytr = np.concatenate([data[q][1] for q in others])
            Xte, yte = data[held]
            primary, f1 = fit_score(Xtr, ytr, Xte, yte, granularity=g)
            rng = np.random.default_rng(hash(held + job + g) % (2**31))
            nulls = []
            for _ in range(n_null):
                sh = int(rng.integers(20, max(len(yte) - 20, 40)))
                a, _ = fit_score(Xtr, ytr, Xte, np.roll(yte, sh), granularity=g)
                if np.isfinite(a):
                    nulls.append(a)
            null = float(np.mean(nulls)) if nulls else np.nan
            p = ((np.array(nulls) >= primary).sum() + 1) / (len(nulls) + 1) if nulls else np.nan
            primary_cond = conds[0] if len(conds) == 1 else "+".join(conds)
            rows.append(
                dict(
                    model="lopo_multisubject",
                    job=job,
                    conds="+".join(conds),
                    condition=primary_cond,
                    granularity=g,
                    metric_name=metric_name,
                    pid=held,
                    held=held,
                    n_train_subjects=len(others),
                    n=len(yte),
                    n_te=len(yte),
                    metric=float(primary) if np.isfinite(primary) else np.nan,
                    f1=float(f1) if np.isfinite(f1) else np.nan,
                    null=null,
                    p=float(p),
                    significant=bool(p < 0.05) if np.isfinite(p) else False,
                    in_rl_subset=bool(held in rl_pids) if rl_pids else False,
                    channels=ch_tag,
                )
            )
            print(
                f"  hold {held}: metric={primary:.3f} F1={f1 if np.isfinite(f1) else float('nan'):.3f} "
                f"null={null:.3f} p={p:.3f}"
            )
        sub = [r for r in rows if r["job"] == job]
        print(f"  mean metric={np.nanmean([r['metric'] for r in sub]):.3f}")
    return pd.DataFrame(rows)
