"""Window features for binary / discrete / continuous labels."""

from __future__ import annotations

import numpy as np

from .channels import CHANNELS_8, PAIRS_8
from .settings import normalize_granularity


def _map_labels(ts, tl, y, continuous: bool):
    """Map label timestamps onto the signal grid."""
    if continuous:
        return np.interp(ts, tl, y.astype(float))
    # nearest (left) label — avoids fractional discrete/binary values
    idx = np.searchsorted(tl, ts, side="right") - 1
    idx = np.clip(idx, 0, len(tl) - 1)
    return y[idx].astype(float)


def _featurize_seg(seg, tt, channels, pairs):
    mean = seg.mean(0)
    sd = seg.std(0)
    sl = np.array([np.polyfit(tt, seg[:, j], 1)[0] for j in range(seg.shape[1])])
    hs, hd = [], []
    for a, b in pairs:
        ia, ib = channels.index(a), channels.index(b)
        hs.append(mean[ia] + mean[ib])
        hd.append(mean[ia] - mean[ib])
    return np.concatenate([mean, sd, sl, hs, hd])


def build_windows(
    ts,
    Z,
    tl,
    y,
    channels: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    granularity: str = "binary",
    window_s: float = 8.0,
    step_s: float = 1.0,
    rate: float = 5.2,
    ambig_lo: float = 0.25,
    ambig_hi: float = 0.75,
    min_majority: float = 0.5,
    min_windows: int = 40,
    min_per_class: int = 15,
    min_std: float = 1e-6,
):
    """Build overlapping windows. Returns (F, y, T) or None.

    binary: mean of labels in window; drop ambiguous; class = mean >= 0.5
    discrete: majority vote; drop if plurality < min_majority
    continuous: mean of continuous_optimal in window
    """
    channels = channels or CHANNELS_8
    pairs = pairs or PAIRS_8
    g = normalize_granularity(granularity)
    w = int(round(window_s * rate))
    st = max(int(round(step_s * rate)), 1)
    yi = _map_labels(ts, tl, y, continuous=(g == "continuous"))

    F, L, T = [], [], []
    for s0 in range(0, len(Z) - w + 1, st):
        seg = Z[s0 : s0 + w]
        tt = ts[s0 : s0 + w] - ts[s0]
        labs = yi[s0 : s0 + w]

        if g == "continuous":
            lab = float(labs.mean())
        elif g == "discrete":
            vals, counts = np.unique(np.round(labs).astype(int), return_counts=True)
            maj = vals[np.argmax(counts)]
            if counts.max() / len(labs) < min_majority:
                continue
            lab = int(maj)
        else:  # binary
            m = float(labs.mean())
            if ambig_lo < m < ambig_hi:
                continue
            lab = int(m >= 0.5)

        F.append(_featurize_seg(seg, tt, channels, pairs))
        L.append(lab)
        T.append(ts[s0])

    if len(F) < min_windows:
        return None
    y_arr = np.asarray(L)
    if g == "continuous":
        if float(np.std(y_arr)) < min_std:
            return None
    else:
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2 or counts.min() < min_per_class:
            return None
    return np.nan_to_num(np.asarray(F, dtype=np.float64)), y_arr, np.asarray(T)


def concat_runs(windows_list, gap_s: float = 1e6):
    """Concatenate several (F, y, T) runs with a time gap so purge CV does not bridge runs."""
    if not windows_list:
        return None
    Fs, ys, Ts = [], [], []
    offset = 0.0
    for F, y, T in windows_list:
        Fs.append(F)
        ys.append(y)
        Ts.append(T - T.min() + offset)
        offset = float(Ts[-1].max()) + gap_s
    return np.vstack(Fs), np.concatenate(ys), np.concatenate(Ts)
