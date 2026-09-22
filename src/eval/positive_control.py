"""Positive control: Play (P) vs Watch (W) decoding with confound checks."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .channels import CHANNELS_8, PAIRS_8
from .data import load_run_only


def _windows(X, el, hz, window_s: float = 10.0, standardize_run: bool = False):
    Y = X.copy()
    if standardize_run:
        for j in range(Y.shape[1]):
            Y[:, j] -= np.polyval(np.polyfit(el, Y[:, j], 1), el)
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    w = max(int(round(window_s * hz)), 10)
    F, T = [], []
    for s in range(0, len(Y) - w + 1, w):
        seg = Y[s : s + w]
        tt = el[s : s + w] - el[s]
        mean = seg.mean(0)
        sd = seg.std(0)
        sl = np.array([np.polyfit(tt, seg[:, j], 1)[0] for j in range(seg.shape[1])])
        hs, hd = [], []
        for a, b in PAIRS_8:
            ia, ib = CHANNELS_8.index(a), CHANNELS_8.index(b)
            hs.append(mean[ia] + mean[ib])
            hd.append(mean[ia] - mean[ib])
        F.append(np.concatenate([mean, sd, sl, hs, hd]))
        T.append(el[s] / max(el[-1], 1e-9))
    if not F:
        return None
    return np.nan_to_num(np.asarray(F, np.float32)), np.asarray(T), float(el[-1])


def _clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))


def run(cfg: dict) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed"])
    window_s = cfg.get("window_s", 10.0)
    runs, runs_z = {}, {}
    for f in sorted(glob.glob(str(processed / "*_processed_*.csv"))):
        b = Path(f).name
        pid = b.split("_")[0]
        cond = b.split("_processed_")[1].replace(".csv", "")
        dom, mode = cond[0], cond[1]
        try:
            X, el, hz = load_run_only(f)
            r = _windows(X, el, hz, window_s=window_s, standardize_run=False)
            rz = _windows(X, el, hz, window_s=window_s, standardize_run=True)
        except Exception:
            continue
        if r is None:
            continue
        runs[(pid, dom, mode)] = r
        runs_z[(pid, dom, mode)] = rz

    pids = sorted({k[0] for k in runs})
    print(f"{len(runs)} runs, {len(pids)} participants")

    rows = []

    # LOPO window-level Play vs Watch
    for held in pids:
        tr = [k for k in runs if k[0] != held]
        te = [k for k in runs if k[0] == held]
        if len({k[2] for k in te}) < 2 or len({k[2] for k in tr}) < 2:
            continue
        Xtr = np.vstack([runs[k][0] for k in tr])
        ytr = np.concatenate([[k[2] == "P"] * len(runs[k][0]) for k in tr]).astype(int)
        Xte = np.vstack([runs[k][0] for k in te])
        yte = np.concatenate([[k[2] == "P"] * len(runs[k][0]) for k in te]).astype(int)
        m = _clf()
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        pred = m.predict(Xte)
        auc = roc_auc_score(yte, proba)
        f1 = f1_score(yte, pred, average="macro")
        rows.append(
            dict(test="lopo_window", held=held, n=len(yte), auc=float(auc), f1=float(f1))
        )
        print(f"  LOPO window {held}: AUC={auc:.3f}  macroF1={f1:.3f}")

    # LOPO after per-run detrend+zscore
    aucs_z = []
    for held in pids:
        tr = [k for k in runs_z if k[0] != held]
        te = [k for k in runs_z if k[0] == held]
        if len({k[2] for k in te}) < 2 or len({k[2] for k in tr}) < 2:
            continue
        Xtr = np.vstack([runs_z[k][0] for k in tr])
        ytr = np.concatenate([[k[2] == "P"] * len(runs_z[k][0]) for k in tr]).astype(int)
        Xte = np.vstack([runs_z[k][0] for k in te])
        yte = np.concatenate([[k[2] == "P"] * len(runs_z[k][0]) for k in te]).astype(int)
        m = _clf()
        m.fit(Xtr, ytr)
        aucs_z.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    if aucs_z:
        print(f"  LOPO detrended mean AUC={np.mean(aucs_z):.3f}")
        rows.append(dict(test="lopo_window_detrend", held="mean", n=len(aucs_z), auc=float(np.mean(aucs_z))))

    # run-level LOPO
    keys = list(runs)
    RF = np.array([runs[k][0].mean(0) for k in keys])
    RY = np.array([k[2] == "P" for k in keys], int)
    RPID = np.array([k[0] for k in keys])
    pred = np.full(len(keys), np.nan)
    for held in pids:
        tr, te = RPID != held, RPID == held
        if te.sum() == 0 or len(np.unique(RY[tr])) < 2:
            continue
        m = _clf()
        m.fit(RF[tr], RY[tr])
        pred[te] = m.predict_proba(RF[te])[:, 1]
    ok = np.isfinite(pred)
    if ok.sum() and len(np.unique(RY[ok])) > 1:
        auc = roc_auc_score(RY[ok], pred[ok])
        print(f"  LOPO run-level AUC={auc:.3f} (n_runs={ok.sum()})")
        rows.append(dict(test="lopo_run", held="all", n=int(ok.sum()), auc=float(auc)))

    return pd.DataFrame(rows)
