#!/usr/bin/env python3
"""RW decoder selection using the RL episode holdout.

Early episodes train the LDA; later episodes are the agent/label set
(same split as trial.py). Sweeps window / step / temporal_shift.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trial import (  # noqa: E402
    _episode_table,
    _keys,
    _resolve_data_roots,
    _split_per_subject,
)
from src.eval.channels import CHANNELS_8, PAIRS_8  # noqa: E402
from src.eval.cv import make_model  # noqa: E402
from src.eval.data import _resolve_data_file, list_pids, load_aligned, pid_key  # noqa: E402
from src.eval.features import build_windows  # noqa: E402
from src.models.eval_compatible import (  # noqa: E402
    _assign_windows_to_episodes,
    _embargo_train_mask,
    _episode_spans,
)
from src.neural.loader import DataLoader  # noqa: E402

WINDOWS = [4.0, 6.0, 8.0]
STEPS = [1.0, 1.5]
SHIFTS = [0.0, 2.0, 4.0, 5.0, 6.0]
EMBARGO_S = 2.0
FRACTION = 0.5


def _as_key(k):
    return (str(k[0]), int(k[1]))


def eval_pid(pid, task_df, dec_keys, agent_keys, processed, labeled, window_s, step_s, shift):
    raw = load_aligned(
        pid, "RW", processed, labeled, channels=CHANNELS_8, granularity="binary", robust_scale=True
    )
    if raw is None:
        return None, "no aligned data"
    ts, Z, tl, y = raw
    sp = _resolve_data_file(Path(processed), pid, "RW", "processed")
    if sp is None:
        return None, "no processed file"
    s = pd.read_csv(sp)
    s["time"] = pd.to_datetime(s["time"], utc=True)
    t0 = s["time"].iloc[0]
    win = dict(
        granularity="binary",
        window_s=window_s,
        step_s=step_s,
        rate=5.2,
        ambig_lo=0.25,
        ambig_hi=0.75,
        min_majority=0.5,
        min_windows=20,
        min_per_class=8,
        min_std=1e-6,
        temporal_shift=shift,
    )
    built = build_windows(ts, Z, tl, y, channels=CHANNELS_8, pairs=PAIRS_8, **win)
    if built is None:
        return None, "build_windows None"
    F, yw, Tw = built
    spans = _episode_spans(task_df, pid)
    if not spans:
        return None, "no episode spans"
    assign = _assign_windows_to_episodes(Tw, t0, spans)
    span_keys = [spn[0] for spn in spans]
    is_dec = np.zeros(len(yw), dtype=bool)
    is_te = np.zeros(len(yw), dtype=bool)
    for i, si in enumerate(assign):
        if si < 0:
            continue
        key = _as_key(span_keys[si])
        if key in dec_keys:
            is_dec[i] = True
        if key in agent_keys:
            is_te[i] = True
    gap = float(window_s) + float(EMBARGO_S)
    tr = is_dec & _embargo_train_mask(Tw, is_te, gap) if is_te.any() else is_dec
    te = is_te
    if tr.sum() < 15:
        return None, f"too few train ({tr.sum()})"
    if len(np.unique(yw[tr])) < 2:
        return None, "train single class"
    if te.sum() < 8:
        return None, f"too few holdout ({te.sum()})"
    if len(np.unique(yw[te])) < 2:
        return None, "holdout single class"
    model = make_model("binary")
    model.fit(F[tr], yw[tr])
    pred = model.predict(F[te])
    scores = model.decision_function(F[te])
    return dict(
        pid=pid,
        window_s=window_s,
        step_s=step_s,
        temporal_shift=shift,
        n_windows=int(len(yw)),
        n_train=int(tr.sum()),
        n_holdout=int(te.sum()),
        train_frac1=float(yw[tr].mean()),
        holdout_frac1=float(yw[te].mean()),
        auc=float(roc_auc_score(yw[te], scores)),
        macro_f1=float(f1_score(yw[te], pred, average="macro")),
    ), None


def main() -> int:
    data = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData"
    processed, labeled, task = _resolve_data_roots(data)
    pids = list_pids(processed, labeled, "RW")
    print(f"RW pids: {pids}")

    loader = DataLoader(
        fnirs_data_source_path=processed,
        task_data_source_path=task,
        labeled_data_source_path=labeled,
        participant_list=[int(p) if p.isdigit() else p for p in pids],
        conditions_list=["RW"],
    )
    task_df = loader.load_task()
    episode_ids = _episode_table(task_df)
    dec_eps, agent_eps = _split_per_subject(episode_ids, FRACTION, 42, task_df=task_df)
    print("episode split (decoder / agent):")
    for pid, g in dec_eps.groupby("pid"):
        print(f"  {pid}: {len(g)} / {len(agent_eps[agent_eps.pid == pid])}")

    rows, skips = [], []
    for window_s in WINDOWS:
        for step_s in STEPS:
            for shift in SHIFTS:
                print(f"\n== window={window_s} step={step_s} shift={shift} ==")
                for pid in pids:
                    grp = dec_eps[dec_eps.pid == pid]
                    agrp = agent_eps[agent_eps.pid == pid]
                    if grp.empty or agrp.empty:
                        skips.append((pid, window_s, step_s, shift, "no episode split"))
                        continue
                    rec, err = eval_pid(
                        pid,
                        task_df,
                        {_as_key(k) for k in _keys(grp)},
                        {_as_key(k) for k in _keys(agrp)},
                        processed,
                        labeled,
                        window_s,
                        step_s,
                        shift,
                    )
                    if rec is None:
                        print(f"  skip {pid}: {err}")
                        skips.append((pid, window_s, step_s, shift, err))
                        continue
                    rows.append(rec)
                    print(
                        f"  {pid} train={rec['n_train']:3d} hold={rec['n_holdout']:3d} "
                        f"AUC={rec['auc']:.3f} F1={rec['macro_f1']:.3f}"
                    )

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"rw_rl_split_sweep_{stamp}.csv"
    df.to_csv(path, index=False)
    print(f"\nsaved → {path}")
    if df.empty:
        return 1

    print("\n=== Mean holdout by window × step × shift ===")
    grid = (
        df.groupby(["window_s", "step_s", "temporal_shift"])[["auc", "macro_f1", "n_holdout"]]
        .mean()
        .round(3)
    )
    print(grid.to_string())

    setting = (
        df.groupby(["window_s", "step_s", "temporal_shift"])
        .agg(auc=("auc", "mean"), f1=("macro_f1", "mean"), n=("pid", "nunique"))
        .sort_values(["f1", "auc"], ascending=False)
    )
    print("\n=== Settings ranked by mean holdout F1 ===")
    print(setting.head(8).round(3).to_string())

    print("\n=== Per-pid mean across all settings (stability) ===")
    stab = (
        df.groupby("pid")
        .agg(
            n=("macro_f1", "size"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
            f1_min=("macro_f1", "min"),
            auc_mean=("auc", "mean"),
            auc_min=("auc", "min"),
            n_f1_ge45=("macro_f1", lambda s: int((s >= 0.45).sum())),
            n_auc_ge60=("auc", lambda s: int((s >= 0.60).sum())),
        )
        .sort_values(["f1_mean", "auc_mean"], ascending=False)
    )
    print(stab.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
