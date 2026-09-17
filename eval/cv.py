"""Purged CV and models for binary / discrete / continuous evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .settings import normalize_granularity


def make_model(granularity: str = "binary"):
    g = normalize_granularity(granularity)
    if g == "continuous":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return LDA(solver="lsqr", shrinkage="auto")


def _score(y, pred_or_decision, granularity: str):
    """Return primary metric (higher is better)."""
    g = normalize_granularity(granularity)
    if g == "continuous":
        if np.std(y) == 0 or np.std(pred_or_decision) == 0:
            return np.nan
        r = spearmanr(y, pred_or_decision).correlation
        return float(r) if np.isfinite(r) else np.nan
    if g == "discrete":
        # pred_or_decision is class predictions for F1 path; for OOF we pass preds
        return float(f1_score(y, pred_or_decision, average="macro"))
    # binary: decision scores → AUC
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, pred_or_decision))


def _predict_oof_fold(model, Fte, granularity: str):
    g = normalize_granularity(granularity)
    if g == "continuous":
        return model.predict(Fte)
    if g == "discrete":
        return model.predict(Fte).astype(float)
    # binary: decision_function (signed score)
    if hasattr(model, "decision_function"):
        d = model.decision_function(Fte)
        return d if np.ndim(d) == 1 else d[:, 1] if d.shape[1] == 2 else d.ravel()
    return model.predict_proba(Fte)[:, 1]


def purged_oof(
    F,
    y,
    T,
    n_folds: int = 8,
    window_s: float = 8.0,
    embargo_s: float = 4.0,
    granularity: str = "binary",
):
    """Contiguous test blocks; drop train windows within window+embargo of the test span.

    Returns (metric, pred, blk, ok). Metric is AUC / macro-F1 / Spearman rho.
    """
    g = normalize_granularity(granularity)
    n = len(y)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    pred = np.full(n, np.nan)
    blk = np.full(n, -1)
    gap = window_s + embargo_s

    for k in range(n_folds):
        te = np.zeros(n, bool)
        te[edges[k] : edges[k + 1]] = True
        if te.sum() < 5:
            continue
        lo, hi = T[te].min() - gap, T[te].max() + gap
        tr = (~te) & ((T < lo) | (T > hi))
        if tr.sum() < 30:
            continue
        if g != "continuous":
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
        else:
            if np.std(y[tr]) == 0 or np.std(y[te]) == 0:
                continue
        m = make_model(g)
        m.fit(F[tr], y[tr])
        pred[te] = _predict_oof_fold(m, F[te], g)
        blk[te] = k

    ok = np.isfinite(pred)
    if ok.sum() < 20:
        return np.nan, None, None, None
    if g == "continuous":
        metric = _score(y[ok], pred[ok], g)
    elif g == "discrete":
        metric = _score(y[ok], pred[ok], g)
    else:
        if len(np.unique(y[ok])) < 2:
            return np.nan, None, None, None
        metric = _score(y[ok], pred[ok], g)
    return metric, pred, blk, ok


def circular_null_p(F, y, T, metric, n_null: int = 200, seed: int = 0, **cv):
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        sh = int(rng.integers(30, max(len(y) - 30, 60)))
        a, *_ = purged_oof(F, np.roll(y, sh), T, **cv)
        if np.isfinite(a):
            nulls.append(a)
    if not nulls:
        return np.nan, np.nan
    nulls = np.asarray(nulls)
    # higher is better for AUC/F1/rho
    p = (np.sum(nulls >= metric) + 1) / (len(nulls) + 1)
    return float(nulls.mean()), float(p)


def block_bootstrap_ci(y, pred, blk, ok, granularity: str = "binary", n_boot: int = 400, seed: int = 0):
    bs = [b for b in np.unique(blk[ok]) if b >= 0]
    if len(bs) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    g = normalize_granularity(granularity)
    for _ in range(n_boot):
        pick = rng.choice(bs, len(bs), replace=True)
        idx = np.concatenate([np.where((blk == b) & ok)[0] for b in pick])
        if g != "continuous" and len(np.unique(y[idx])) < 2:
            continue
        if g == "continuous" and (np.std(y[idx]) == 0 or np.std(pred[idx]) == 0):
            continue
        v = _score(y[idx], pred[idx], g)
        if np.isfinite(v):
            out.append(v)
    if len(out) < 50:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fit_score(Xtr, ytr, Xte, yte, granularity: str = "binary"):
    g = normalize_granularity(granularity)
    m = make_model(g)
    m.fit(Xtr, ytr)
    if g == "continuous":
        pred = m.predict(Xte)
        return _score(yte, pred, g), np.nan
    pred = m.predict(Xte)
    f1 = float(f1_score(yte, pred, average="macro"))
    if g == "binary" and len(np.unique(yte)) > 1:
        d = m.decision_function(Xte)
        auc = float(roc_auc_score(yte, d))
        return auc, f1
    if g == "discrete" and len(np.unique(yte)) > 1:
        try:
            proba = m.predict_proba(Xte)
            auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = np.nan
        return auc, f1
    return np.nan, f1
