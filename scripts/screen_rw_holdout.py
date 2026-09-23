#!/usr/bin/env python3
"""Screen RW participants at a 0.65 decoder-episode holdout.

Fits the same eval-compatible LDA as trial.py (early episodes train, later
episodes score). Keeps anyone with holdout macro-F1 >= 0.55 and writes a CSV
of used/skipped IDs with scores.

python scripts/screen_rw_holdout.py

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
from src.eval.channels import CHANNELS_8, PAIRS_8  # noqa: E402
from src.eval.data import list_pids  # noqa: E402
from src.models.decoder_bank import normalize_pid  # noqa: E402
from src.models.eval_compatible import train_subject_eval_decoder  # noqa: E402
from src.neural.loader import DataLoader  # noqa: E402

DEFAULT_DATA = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData"


def _pid_int(pid: str):
    return int(pid) if str(pid).isdigit() else pid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA))
    p.add_argument("--fraction", type=float, default=0.65)
    p.add_argument("--min-f1", type=float, default=0.55)
    p.add_argument("--window-s", type=float, default=6.0)
    p.add_argument("--step-s", type=float, default=1.0)
    p.add_argument("--shift-s", type=float, default=6.0)
    p.add_argument("--embargo-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--condition", default="RW")
    p.add_argument(
        "--pids",
        default="",
        help="Comma-separated participant IDs. Default: all with RW files.",
    )
    args = p.parse_args()

    processed, labeled, task = _resolve_data_roots(args.data_path)
    all_pids = [normalize_pid(pid) for pid in list_pids(processed, labeled, args.condition)]
    if args.pids.strip():
        want = {normalize_pid(x.strip()) for x in args.pids.split(",") if x.strip()}
        pids = [pid for pid in all_pids if pid in want]
    else:
        pids = all_pids
    if not pids:
        print(f"No {args.condition} participants found under {processed}")
        return 1

    print(
        f"Screen {args.condition} pids={pids} | decoder_fraction={args.fraction} "
        f"min_f1={args.min_f1} window={args.window_s} step={args.step_s} "
        f"shift={args.shift_s} embargo={args.embargo_s}"
    )

    loader = DataLoader(
        fnirs_data_source_path=processed,
        task_data_source_path=task,
        labeled_data_source_path=labeled,
        participant_list=[_pid_int(pid) for pid in pids],
        conditions_list=[args.condition],
    )
    task_df = loader.load_task()
    episode_ids = _episode_table(task_df)
    dec_eps, agent_eps = _split_per_subject(
        episode_ids, args.fraction, args.seed, task_df=task_df
    )

    rows: list[dict] = []
    for pid in pids:
        grp = dec_eps[dec_eps.pid == pid]
        agrp = agent_eps[agent_eps.pid == pid]
        n_dec, n_rl = int(len(grp)), int(len(agrp))
        print(f"\n== pid={pid} decoder_eps={n_dec} rl_eps={n_rl} ==")
        if grp.empty:
            print("  skip: no decoder episodes")
            rows.append(
                dict(
                    pid=pid,
                    used=False,
                    reason="no decoder episodes",
                    decoder_fraction=args.fraction,
                    n_decoder_episodes=n_dec,
                    n_rl_episodes=n_rl,
                    holdout_f1=None,
                    holdout_auc=None,
                )
            )
            continue

        clf, report = train_subject_eval_decoder(
            pid,
            [args.condition],
            processed,
            labeled,
            task_df,
            decoder_keys=_keys(grp),
            agent_keys=_keys(agrp) if n_rl else set(),
            granularity="binary",
            window_s=args.window_s,
            step_s=args.step_s,
            embargo_s=args.embargo_s,
            rate_hz=5.2,
            temporal_shift=args.shift_s,
            channels=list(CHANNELS_8),
            pairs=list(PAIRS_8),
            seed=args.seed,
        )
        if clf is None:
            err = report.get("error", report) if isinstance(report, dict) else report
            print(f"  skip: {err}")
            rows.append(
                dict(
                    pid=pid,
                    used=False,
                    reason=str(err),
                    decoder_fraction=args.fraction,
                    n_decoder_episodes=n_dec,
                    n_rl_episodes=n_rl,
                    holdout_f1=None,
                    holdout_auc=None,
                )
            )
            continue

        f1 = _holdout_f1(report)
        auc = report.get("holdout_auc")
        keep = f1 is not None and f1 >= args.min_f1
        reason = "kept" if keep else f"holdout_f1 {f1} < {args.min_f1}"
        print(
            f"  train={report.get('n_train')} holdout={report.get('n_holdout')} "
            f"F1={f1} AUC={auc} → {'KEEP' if keep else 'DROP'}"
        )
        rows.append(
            dict(
                pid=pid,
                used=keep,
                reason=reason,
                decoder_fraction=args.fraction,
                min_holdout_f1=args.min_f1,
                n_decoder_episodes=n_dec,
                n_rl_episodes=n_rl,
                n_train=report.get("n_train"),
                n_holdout=report.get("n_holdout"),
                n_embargo_dropped=report.get("n_embargo_dropped"),
                holdout_f1=f1,
                holdout_auc=auc,
                window_s=args.window_s,
                step_s=args.step_s,
                temporal_shift=args.shift_s,
                embargo_s=args.embargo_s,
            )
        )

    out_dir = REPO_ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"decoder_holdout_screen_Robot_Passive_{stamp}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

    kept = df[df["used"]].copy()
    print(f"\nSaved screen → {path}")
    print(f"Kept {len(kept)} / {len(df)} (F1 >= {args.min_f1} at fraction={args.fraction})")
    if not kept.empty:
        show = kept[["pid", "holdout_f1", "holdout_auc", "n_train", "n_holdout", "n_rl_episodes"]]
        print(show.to_string(index=False))
        ints = []
        for pid in kept["pid"]:
            ints.append(int(pid) if str(pid).isdigit() else pid)
        print(f"\nparticipant_list: {ints}")
    else:
        print("No participants passed the holdout gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
