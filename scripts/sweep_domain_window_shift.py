#!/usr/bin/env python3
"""RL-holdout grid: window × step × temporal_shift for Lunar / Flappy / Robot.

Early episodes train the LDA; later episodes score holdout macro-F1 / AUC
(same split as trial.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trial import (  # noqa: E402
    _episode_table,
    _holdout_f1,
    _keys,
    _resolve_data_roots,
    _split_per_subject,
)
import src.utils as utils  # noqa: E402
from src.eval.data import list_pids  # noqa: E402
from src.eval.settings import channel_setup  # noqa: E402
from src.models.decoder_bank import normalize_pid  # noqa: E402
from src.models.eval_compatible import train_subject_eval_decoder  # noqa: E402
from src.neural.loader import DataLoader  # noqa: E402

DEFAULT_DATA = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData"
WINDOWS = [4.0, 6.0, 8.0]
STEPS = [1.0, 1.5]
SHIFTS = [0.0, 2.0, 4.0, 5.0, 6.0]


def _pid_int(pid: str):
    return int(pid) if str(pid).isdigit() else pid


def sweep_one(domain: str, task: str, args) -> pd.DataFrame:
    conditions = utils.get_conditions(domain, task)
    processed, labeled, task_dir = _resolve_data_roots(args.data_path)
    pid_sets = [set(normalize_pid(p) for p in list_pids(processed, labeled, c)) for c in conditions]
    pids = sorted(set.union(*pid_sets)) if pid_sets else []
    if task.lower() == "pooled" and len(pid_sets) > 1:
        pids = sorted(set.intersection(*pid_sets))
    print(f"\n{'=' * 72}")
    print(f"{domain} {task} conditions={conditions} pids={pids}")
    if not pids:
        return pd.DataFrame()

    channels, pairs, ch_tag = channel_setup({"drop_suspect_channels": args.drop_suspect})
    loader = DataLoader(
        fnirs_data_source_path=processed,
        task_data_source_path=task_dir,
        labeled_data_source_path=labeled,
        participant_list=[_pid_int(p) for p in pids],
        conditions_list=conditions,
    )
    task_df = loader.load_task()
    episode_ids = _episode_table(task_df)
    dec_eps, agent_eps = _split_per_subject(
        episode_ids, args.fraction, args.seed, task_df=task_df
    )

    rows: list[dict] = []
    for window_s in args.windows:
        for step_s in args.steps:
            for shift in args.shifts:
                print(f"\n== {domain} {task} window={window_s} step={step_s} shift={shift} ==")
                for pid in pids:
                    grp = dec_eps[dec_eps.pid == pid]
                    agrp = agent_eps[agent_eps.pid == pid]
                    if grp.empty:
                        print(f"  skip {pid}: no decoder episodes")
                        continue
                    clf, report = train_subject_eval_decoder(
                        pid,
                        conditions,
                        processed,
                        labeled,
                        task_df,
                        decoder_keys=_keys(grp),
                        agent_keys=_keys(agrp) if len(agrp) else set(),
                        granularity="binary",
                        window_s=window_s,
                        step_s=step_s,
                        embargo_s=args.embargo_s,
                        rate_hz=5.2,
                        temporal_shift=shift,
                        channels=list(channels),
                        pairs=list(pairs),
                        seed=args.seed,
                    )
                    if clf is None:
                        err = report.get("error", report) if isinstance(report, dict) else report
                        print(f"  skip {pid}: {err}")
                        continue
                    f1 = _holdout_f1(report)
                    auc = report.get("holdout_auc")
                    rec = dict(
                        domain=domain,
                        task=task,
                        conditions="+".join(conditions),
                        pid=pid,
                        window_s=window_s,
                        step_s=step_s,
                        temporal_shift=shift,
                        embargo_s=args.embargo_s,
                        channels=ch_tag,
                        n_train=report.get("n_train"),
                        n_holdout=report.get("n_holdout"),
                        holdout_f1=f1,
                        holdout_auc=auc,
                    )
                    rows.append(rec)
                    print(
                        f"  {pid} train={rec['n_train']} hold={rec['n_holdout']} "
                        f"F1={f1} AUC={auc}"
                    )
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA))
    p.add_argument("--domain", nargs="+", default=["Lunar", "Flappy"])
    p.add_argument("--task", nargs="+", default=["Passive", "Pooled"])
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--embargo-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-suspect", action="store_true", default=True)
    p.add_argument("--keep-suspect", action="store_true")
    args = p.parse_args()
    if args.keep_suspect:
        args.drop_suspect = False
    args.windows = WINDOWS
    args.steps = STEPS
    args.shifts = SHIFTS

    frames = []
    for domain in args.domain:
        for task in args.task:
            frames.append(sweep_one(domain, task, args))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_dir = REPO_ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(d.lower() for d in args.domain)
    path = out_dir / f"{tag}_window_shift_sweep_{stamp}.csv"
    df.to_csv(path, index=False)
    print(f"\nsaved → {path}")
    if df.empty:
        return 1

    for (domain, task), g in df.groupby(["domain", "task"]):
        setting = (
            g.groupby(["window_s", "step_s", "temporal_shift"])
            .agg(
                n=("pid", "nunique"),
                f1=("holdout_f1", "mean"),
                auc=("holdout_auc", "mean"),
                n_f1_ge55=("holdout_f1", lambda s: int((s >= 0.55).sum())),
            )
            .sort_values(["f1", "auc"], ascending=False)
        )
        print(f"\n=== {domain} {task}: settings ranked by mean holdout F1 ===")
        print(setting.head(8).round(3).to_string())
        best = setting.index[0]
        print(
            f"Best: window={best[0]} step={best[1]} shift={best[2]} "
            f"mean F1={setting.iloc[0]['f1']:.3f} mean AUC={setting.iloc[0]['auc']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
