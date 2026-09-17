"""Eval-compatible decoder: robust scale + window features + shrinkage LDA/Ridge.

Matches ``src/eval`` (within-subject tests): same channels/pairs features,
ambiguity handling via ``build_windows``, and purged embargo in seconds on
window timestamps — not a random row holdout and not ``temporal_shift``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.eval.channels import CHANNELS_8, PAIRS_8
from src.eval.cv import fit_score, make_model
from src.eval.data import load_aligned, pid_key
from src.eval.features import _featurize_seg, build_windows
from src.eval.settings import normalize_granularity
from src.models.decoder_bank import normalize_pid


@dataclass
class RobustWindowDecoder:
    """Sklearn-like wrapper: ``predict`` on feature rows, ``predict_raw`` on TxC."""

    model: Any
    med: np.ndarray
    mad: np.ndarray
    channels: list[str]
    pairs: list[tuple[str, str]]
    granularity: str = "binary"
    holdout_metric: float | None = None
    n_train: int = 0
    n_holdout: int = 0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.med) / self.mad, -10.0, 10.0)

    def featurize_raw(self, X: np.ndarray, sample_period_s: float = 1.0 / 5.2) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("raw window must be [T, C] with T>=2")
        Z = self._scale(X)
        # elapsed seconds (eval uses real time, not sample index)
        tt = np.arange(len(Z), dtype=float) * float(sample_period_s)
        C = min(Z.shape[1], len(self.channels))
        feat = _featurize_seg(Z[:, :C], tt, self.channels[:C], self.pairs)
        return feat

    def predict_raw(self, X: np.ndarray, sample_period_s: float = 1.0 / 5.2):
        f = self.featurize_raw(X, sample_period_s=sample_period_s)[None, :]
        return self.model.predict(f)

    def predict_proba_raw(self, X: np.ndarray, sample_period_s: float = 1.0 / 5.2):
        f = self.featurize_raw(X, sample_period_s=sample_period_s)[None, :]
        return self.model.predict_proba(f)

    def predict(self, F):
        return self.model.predict(np.asarray(F))

    def predict_proba(self, F):
        return self.model.predict_proba(np.asarray(F))

    @property
    def classes_(self):
        return getattr(self.model, "classes_", None)


def _robust_stats(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(Z, axis=0)
    mad = 1.4826 * np.median(np.abs(Z - med), axis=0)
    mad[mad == 0] = 1.0
    return med, mad


def _episode_spans(task_df: pd.DataFrame, pid: str) -> list[tuple[tuple, pd.Timestamp, pd.Timestamp]]:
    """Return [( (participantKey, episode), t_start, t_end ), ...] for this pid."""
    df = task_df.copy()
    df["participantKey"] = df["participantKey"].astype(str)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype("Int64")
    df = df[df["participantKey"].map(normalize_pid) == normalize_pid(pid)]
    if df.empty:
        return []
    spans = []
    for key, g in df.groupby(["participantKey", "episode"], sort=False):
        t = pd.to_datetime(g["time"], utc=True)
        spans.append((key, t.min(), t.max()))
    return spans


def _assign_windows_to_episodes(T_elapsed: np.ndarray, t0: pd.Timestamp, spans) -> np.ndarray:
    """Map each window start (elapsed s) to episode key index, or -1."""
    abs_t = pd.to_datetime(t0) + pd.to_timedelta(T_elapsed, unit="s")
    out = np.full(len(T_elapsed), -1, dtype=int)
    for i, (key, lo, hi) in enumerate(spans):
        m = (abs_t >= lo) & (abs_t <= hi)
        out[m] = i
    return out


def _embargo_train_mask(T: np.ndarray, te: np.ndarray, gap_s: float) -> np.ndarray:
    """Keep train indices that are outside [te_min - gap, te_max + gap].

    For a time-ordered episode split the holdout block is contiguous, so this
    matches eval purged-CV embargo semantics (seconds on window timestamps).
    """
    tr = ~te
    if not te.any() or not tr.any():
        return tr
    lo, hi = float(T[te].min()) - gap_s, float(T[te].max()) + gap_s
    keep = tr & ((T < lo) | (T > hi))
    return keep


def train_subject_eval_decoder(
    pid,
    conditions: list[str],
    processed_dir: str,
    labeled_dir: str,
    task_df: pd.DataFrame,
    decoder_keys: set[tuple],
    agent_keys: set[tuple],
    *,
    granularity: str = "binary",
    window_s: float = 8.0,
    step_s: float = 1.0,
    embargo_s: float = 4.0,
    rate_hz: float = 5.2,
    channels: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    seed: int = 0,
) -> tuple[RobustWindowDecoder | None, dict]:
    """Fit one subject decoder on decoder-episode windows; score agent-episode holdout.

    Embargo drops train windows within ``window_s + embargo_s`` seconds of any
    holdout window (same units as eval purged CV).
    """
    channels = channels or CHANNELS_8
    pairs = pairs or PAIRS_8
    g = normalize_granularity(granularity)
    pid = pid_key(pid)
    gap = float(window_s) + float(embargo_s)

    win = dict(
        granularity=g,
        window_s=window_s,
        step_s=step_s,
        rate=rate_hz,
        ambig_lo=0.25,
        ambig_hi=0.75,
        min_majority=0.5,
        min_windows=20,
        min_per_class=8,
        min_std=1e-6,
    )

    spans = _episode_spans(task_df, pid)
    if not spans:
        return None, {"error": "no episode spans"}

    F_all, y_all, T_all = [], [], []
    ep_idx_all = []
    Zu_pool = []
    t0_by_cond = {}
    skip_reasons: list[str] = []

    for cond in conditions:
        raw, reason = load_aligned(
            pid,
            cond,
            processed_dir,
            labeled_dir,
            channels=channels,
            granularity=g,
            robust_scale=True,
            return_reason=True,
        )
        if raw is None:
            skip_reasons.append(f"{cond}: {reason}")
            continue
        ts, Z, tl, y = raw
        from pathlib import Path
        from src.eval.data import _resolve_data_file

        sp = _resolve_data_file(Path(processed_dir), pid, cond, "processed")
        if sp is None:
            skip_reasons.append(f"{cond}: processed file vanished after load")
            continue
        s = pd.read_csv(sp)
        s["time"] = pd.to_datetime(s["time"], utc=True)
        t0 = s["time"].iloc[0]
        t0_by_cond[cond] = t0
        built = build_windows(ts, Z, tl, y, channels=channels, pairs=pairs, **win)
        if built is None:
            skip_reasons.append(
                f"{cond}: build_windows returned None "
                f"(need >= {win['min_windows']} windows / >= {win['min_per_class']} per class)"
            )
            continue
        F, yw, Tw = built

        raw_u, _ = load_aligned(
            pid,
            cond,
            processed_dir,
            labeled_dir,
            channels=channels,
            granularity=g,
            robust_scale=False,
            return_reason=True,
        )
        if raw_u is not None:
            Zu_pool.append(raw_u[1])

        assign = _assign_windows_to_episodes(Tw, t0, spans)
        F_all.append(F)
        y_all.append(yw)
        T_all.append(Tw)
        ep_idx_all.append(assign)

    if not F_all or not Zu_pool:
        detail = "; ".join(skip_reasons) if skip_reasons else "unknown"
        return None, {
            "error": "no windows",
            "detail": detail,
            "processed_dir": str(processed_dir),
            "labeled_dir": str(labeled_dir),
            "conditions": list(conditions),
            "pid": pid,
        }

    med_ref, mad_ref = _robust_stats(np.vstack(Zu_pool))

    F = np.vstack(F_all)
    y = np.concatenate(y_all)
    T = np.concatenate(T_all)
    ep_i = np.concatenate(ep_idx_all)

    # map span index → episode key
    span_keys = [sp[0] for sp in spans]
    # normalize keys for membership (participantKey str, episode int)
    def _as_key(k):
        return (str(k[0]), int(k[1]))

    decoder_keys_n = {_as_key(k) for k in decoder_keys}
    agent_keys_n = {_as_key(k) for k in agent_keys}

    is_dec = np.zeros(len(y), dtype=bool)
    is_te = np.zeros(len(y), dtype=bool)
    for i, si in enumerate(ep_i):
        if si < 0:
            continue
        key = _as_key(span_keys[si])
        if key in decoder_keys_n:
            is_dec[i] = True
        if key in agent_keys_n:
            is_te[i] = True

    # train pool = decoder episodes, minus embargo vs holdout windows
    tr = is_dec & _embargo_train_mask(T, is_te, gap) if is_te.any() else is_dec
    te = is_te

    if tr.sum() < 15:
        return None, {"error": f"too few train windows after embargo ({tr.sum()})"}
    if g != "continuous" and len(np.unique(y[tr])) < 2:
        return None, {"error": "train windows lack both classes"}

    model = make_model(g)
    model.fit(F[tr], y[tr])

    report: dict[str, Any] = {
        "n_train": int(tr.sum()),
        "n_holdout": int(te.sum()),
        "n_embargo_dropped": int(is_dec.sum() - tr.sum()),
        "gap_s": gap,
    }

    holdout_metric = None
    if te.sum() >= 10 and (g == "continuous" or len(np.unique(y[te])) >= 2):
        primary, _extra = fit_score(F[tr], y[tr], F[te], y[te], granularity=g)
        holdout_metric = primary
        report["holdout_metric"] = float(primary) if primary == primary else None
        report["metric_name"] = {"binary": "AUC", "discrete": "macroF1", "continuous": "Spearman"}[g]
    else:
        report["holdout_metric"] = None
        report["note"] = "holdout too small or single-class"

    dec = RobustWindowDecoder(
        model=model,
        med=med_ref,
        mad=mad_ref,
        channels=list(channels),
        pairs=list(pairs),
        granularity=g,
        holdout_metric=holdout_metric,
        n_train=int(tr.sum()),
        n_holdout=int(te.sum()),
    )
    return dec, report
