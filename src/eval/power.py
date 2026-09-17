"""Minimum detectable effect / power curve under the within-subject pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import channels as ch
from .cv import purged_oof
from .data import list_pids, load_aligned
from .features import build_windows


def _hrf_kernel(hz: float):
    t = np.arange(0, 20, 1 / hz)
    h = (t**2) * np.exp(-t / 1.2)
    return h / h.sum()


def _inject(ts, Z, tl, y, amp, rng, rate, n_resp):
    yi = np.interp(ts, tl, y.astype(float))
    Zs = Z.copy()
    if amp > 0:
        k = _hrf_kernel(rate)
        drive = np.convolve(yi, k, mode="full")[: len(yi)]
        drive = (drive - drive.mean()) / (drive.std() + 1e-9)
        chans = rng.choice(Z.shape[1], n_resp, replace=False)
        for c in chans:
            Zs[:, c] = Zs[:, c] + amp * drive
    return ts, Zs, tl, y


def run(cfg: dict) -> pd.DataFrame:
    processed, labeled = cfg["paths"]["processed"], cfg["paths"]["labeled"]
    conds = cfg.get("conditions", ch.CONDS)
    amps = cfg.get("amplitudes", [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    n_null = cfg.get("n_null", 100)
    n_resp = cfg.get("n_response_channels", 4)
    rate = cfg.get("rate_hz", 5.2)
    win = dict(
        window_s=cfg.get("window_s", 8.0),
        step_s=cfg.get("step_s", 1.0),
        rate=rate,
        ambig_lo=cfg.get("ambig_lo", 0.25),
        ambig_hi=cfg.get("ambig_hi", 0.75),
        min_windows=cfg.get("min_windows", 40),
        min_per_class=cfg.get("min_per_class", 15),
    )
    cv = dict(n_folds=cfg.get("n_folds", 8), window_s=win["window_s"], embargo_s=cfg.get("embargo_s", 4.0))

    datasets = []
    for cond in conds:
        for pid in list_pids(processed, labeled, cond):
            raw = load_aligned(pid, cond, processed, labeled)
            if raw is None:
                continue
            if build_windows(*raw, **win) is None:
                continue
            datasets.append((pid, cond, raw))
    print(f"{len(datasets)} participant-conditions usable\n")
    print(f"{'amp':>6} {'mean AUC':>9} {'median AUC':>11} {'power':>8}")

    out = []
    for amp in amps:
        aucs, det, tot = [], 0, 0
        for i, (pid, cond, raw) in enumerate(datasets):
            rng = np.random.default_rng(1000 + i)
            inj = _inject(*raw, amp, rng, rate, n_resp)
            built = build_windows(*inj, **win)
            if built is None:
                continue
            F, y, T = built
            a, *_ = purged_oof(F, y, T, **cv)
            if not np.isfinite(a):
                continue
            rs = np.random.default_rng(2000 + i)
            nulls = []
            for _ in range(n_null):
                sh = int(rs.integers(30, max(len(y) - 30, 60)))
                v, *_ = purged_oof(F, np.roll(y, sh), T, **cv)
                if np.isfinite(v):
                    nulls.append(v)
            if not nulls:
                continue
            p = (np.sum(np.asarray(nulls) >= a) + 1) / (len(nulls) + 1)
            aucs.append(a)
            tot += 1
            det += int(p < 0.05)
        pw = det / max(tot, 1)
        out.append(
            dict(
                amp=amp,
                mean_auc=float(np.mean(aucs)) if aucs else np.nan,
                med_auc=float(np.median(aucs)) if aucs else np.nan,
                power=pw,
                n=tot,
                n_detected=det,
            )
        )
        print(
            f"{amp:6.2f} {np.mean(aucs):9.3f} {np.median(aucs):11.3f} {pw:8.2f}  ({det}/{tot})"
        )
    return pd.DataFrame(out)
