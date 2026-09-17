"""Event-related switch-locked decoding with TIME-MATCHED controls.

For each to-suboptimal switch, pick a stable control epoch whose midpoint is as
close as possible in time (within the same run). This removes the 'switches happen
late/early' confound that inflated naive AUCs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .cv import make_model
from .data import list_pids, load_aligned, pid_key
from .features import _featurize_seg
from .settings import channel_setup, expand_jobs


def _nearest_labels(ts, tl, y):
    idx = np.searchsorted(tl, ts, side="right") - 1
    return y[np.clip(idx, 0, len(tl) - 1)].astype(int)


def _switches(yi):
    return np.where(np.diff(yi) != 0)[0] + 1


def _epoch_feat(Z, ts, t0, t1, channels, pairs):
    m = (ts >= t0) & (ts < t1)
    if m.sum() < 5:
        return None
    return _featurize_seg(Z[m], ts[m] - ts[m][0], channels, pairs)


def build_epochs(ts, Z, tl, y, channels, pairs, lag_s, tau_s, switch_dir, margin_s, seed):
    yi = _nearest_labels(ts, tl, y)
    keep = []
    for i in _switches(yi):
        prev, cur = int(yi[i - 1]), int(yi[i])
        if switch_dir == "to_suboptimal" and not (prev == 1 and cur == 0):
            continue
        if switch_dir == "to_optimal" and not (prev == 0 and cur == 1):
            continue
        keep.append(i)
    if len(keep) < 3:
        return None

    sw_times = ts[np.asarray(keep)]
    dur = lag_s + tau_s
    rng = np.random.default_rng(seed)

    # candidate control starts
    step = max(int(round(0.5 / max(float(np.median(np.diff(ts))), 0.05))), 1)
    cands = []  # (t0, tmid)
    for i in range(0, len(ts) - 1, step):
        t0 = float(ts[i])
        t1 = t0 + dur
        if t1 > ts[-1]:
            break
        if any((t0 - margin_s) <= float(sw) <= (t1 + margin_s) for sw in sw_times):
            continue
        m = (ts >= t0) & (ts < t1)
        if m.sum() < 5 or len(np.unique(yi[m])) != 1:
            continue
        cands.append((t0, t0 + 0.5 * dur))
    if len(cands) < 3:
        return None
    cands = np.asarray(cands, float)

    F_sw, F_c, t_sw_mid, t_c_mid = [], [], [], []
    used_ctrl = set()
    for t_sw in sw_times:
        t0, t1 = float(t_sw) + lag_s, float(t_sw) + lag_s + tau_s
        if t1 > ts[-1] or t0 < ts[0]:
            continue
        feat = _epoch_feat(Z, ts, t0, t1, channels, pairs)
        if feat is None:
            continue
        target_mid = 0.5 * (t0 + t1)
        # nearest unused control by midpoint distance
        dists = np.abs(cands[:, 1] - target_mid)
        order = np.argsort(dists)
        chosen = None
        for j in order:
            key = round(cands[j, 0], 3)
            if key in used_ctrl:
                continue
            # require reasonably close: within 30% of run or 40s
            run_len = float(ts[-1] - ts[0])
            if dists[j] > max(40.0, 0.3 * run_len):
                break
            chosen = j
            break
        if chosen is None:
            continue
        c0 = float(cands[chosen, 0])
        cfeat = _epoch_feat(Z, ts, c0, c0 + dur, channels, pairs)
        if cfeat is None:
            continue
        used_ctrl.add(round(c0, 3))
        F_sw.append(feat)
        F_c.append(cfeat)
        t_sw_mid.append(target_mid)
        t_c_mid.append(float(cands[chosen, 1]))

    n = len(F_sw)
    if n < 3:
        return None
    F = np.nan_to_num(np.vstack([F_sw, F_c]))
    lab = np.array([1] * n + [0] * n)
    tmid = np.array(t_sw_mid + t_c_mid, float)
    tmid = (tmid - ts[0]) / max(float(ts[-1] - ts[0]), 1e-9)
    # pairing quality: mean |t_sw - t_ctrl| in normalized units
    pair_dt = float(np.mean(np.abs(np.array(t_sw_mid) - np.array(t_c_mid))))
    return F, lab, tmid, pair_dt, n


def cv_auc(F, y, n_splits=5, seed=0):
    if len(y) < 8 or len(np.unique(y)) < 2:
        return np.nan
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan)
    for tr, te in skf.split(F, y):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_model("binary")
        m.fit(F[tr], y[tr])
        scores[te] = m.decision_function(F[te])
    ok = np.isfinite(scores)
    if ok.sum() < 6 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], scores[ok]))


def time_only_auc(tmid, y):
    if len(np.unique(y)) < 2 or np.std(tmid) == 0:
        return np.nan
    return float(roc_auc_score(y, tmid))


def perm_p(F, y, auc, n_null=80, seed=0):
    rng = np.random.default_rng(seed)
    nulls = []
    for i in range(n_null):
        yp = y.copy()
        rng.shuffle(yp)
        a = cv_auc(F, yp, seed=10_000 + i)
        if np.isfinite(a):
            nulls.append(a)
    if not nulls:
        return np.nan, np.nan
    nulls = np.asarray(nulls)
    return float(nulls.mean()), float((np.sum(nulls >= auc) + 1) / (len(nulls) + 1))


def run(cfg: dict) -> pd.DataFrame:
    processed, labeled = cfg["paths"]["processed"], cfg["paths"]["labeled"]
    channels, pairs, ch_tag = channel_setup(cfg)
    lags = cfg.get("lags_s", [4.0])
    taus = cfg.get("taus_s", [6.0])
    dirs = cfg.get("switch_dirs", ["to_suboptimal"])
    margin = cfg.get("margin_s", 4.0)
    n_null = cfg.get("n_null", 80)
    primary_lag = cfg.get("primary_lag_s", lags[0])
    primary_tau = cfg.get("primary_tau_s", taus[0])
    primary_dir = cfg.get("primary_dir", dirs[0])

    rows = []
    for _job, conds in expand_jobs(cfg):
        for cond in conds:
            pids = list_pids(processed, labeled, cond)
            print(f"\n== {cond}: {len(pids)} candidates ==", flush=True)
            for lag in lags:
                for tau in taus:
                    for sdir in dirs:
                        for pid in pids:
                            raw = load_aligned(
                                pid, cond, processed, labeled, channels=channels, granularity="binary"
                            )
                            if raw is None:
                                continue
                            ts, Z, tl, y = raw
                            built = build_epochs(
                                ts, Z, tl, y, channels, pairs, lag, tau, sdir, margin,
                                seed=hash(f"{pid}{cond}{lag}{tau}{sdir}") % (2**31),
                            )
                            if built is None:
                                continue
                            F, lab, tmid, pair_dt, n = built
                            auc = cv_auc(F, lab, seed=hash(pid + cond) % (2**31))
                            if not np.isfinite(auc):
                                continue
                            t_auc = time_only_auc(tmid, lab)
                            null, p = perm_p(F, lab, auc, n_null=n_null,
                                             seed=hash(f"n{pid}{cond}{lag}") % (2**31))
                            is_primary = (
                                abs(lag - primary_lag) < 1e-9
                                and abs(tau - primary_tau) < 1e-9
                                and sdir == primary_dir
                            )
                            row = dict(
                                cond=cond, pid=pid_key(pid), lag_s=lag, tau_s=tau,
                                switch_dir=sdir, n_switch=n, n_total=2 * n,
                                auc=auc, time_auc=t_auc, pair_dt_s=pair_dt,
                                null=null, p=p, primary=is_primary, channels=ch_tag,
                            )
                            rows.append(row)
                            if is_primary:
                                mark = " *" if p < 0.05 else ""
                                print(
                                    f"  {row['pid']} n={n}+{n} AUC={auc:.3f} "
                                    f"timeAUC={t_auc:.3f} Δt={pair_dt:.1f}s "
                                    f"null={null:.3f} p={p:.3f}{mark}",
                                    flush=True,
                                )

    df = pd.DataFrame(rows)
    if not len(df):
        print("no usable participants/epochs")
        return df

    prim = df[df.primary].copy()
    if len(prim):
        prim["excess_vs_time"] = prim.auc - prim.time_auc
        print(
            f"\n>>> PRIMARY lag={primary_lag}s tau={primary_tau}s {primary_dir} "
            f"(time-matched controls)\n"
            f"    n={len(prim)} mean AUC={prim.auc.mean():.3f}  "
            f"mean timeAUC={prim.time_auc.mean():.3f}\n"
            f"    mean null={prim.null.mean():.3f}  sig={int((prim.p < 0.05).sum())}/{len(prim)}\n"
            f"    mean (AUC-timeAUC)={prim.excess_vs_time.mean():.3f}  "
            f"|AUC>time+0.05|={(prim.excess_vs_time > 0.05).sum()}/{len(prim)}",
            flush=True,
        )
        for cond, g in prim.groupby("cond"):
            print(
                f"    {cond}: AUC={g.auc.mean():.3f} time={g.time_auc.mean():.3f} "
                f"sig={(g.p < 0.05).sum()}/{len(g)}",
                flush=True,
            )

    if len(lags) * len(taus) * len(dirs) > 1:
        print("\n>>> grid mean AUC <<<", flush=True)
        g = df.groupby(["lag_s", "tau_s", "switch_dir"]).agg(
            mean_auc=("auc", "mean"), mean_time=("time_auc", "mean"),
            n=("auc", "count"), n_sig=("p", lambda s: int((s < 0.05).sum())),
        )
        print(g.round(3).to_string(), flush=True)
    return df
