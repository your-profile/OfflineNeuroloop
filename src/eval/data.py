"""Load fixed processed fNIRS + labels from LabeledData."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .channels import CHANNELS_8
from .settings import LABEL_COLS, normalize_granularity


def pid_key(pid) -> str:
    s = str(pid)
    return s.zfill(3) if s.isdigit() else s


def list_pids(processed_dir: str | Path, labeled_dir: str | Path, cond: str) -> list[str]:
    processed_dir, labeled_dir = Path(processed_dir), Path(labeled_dir)
    proc_names = (
        list(glob.glob(str(processed_dir / f"*_processed_{cond}.csv")))
        + list(glob.glob(str(processed_dir / f"*_{cond}_processed.csv")))
    )
    lab_names = (
        list(glob.glob(str(labeled_dir / f"*_{cond}_LabeledData.csv")))
        + list(glob.glob(str(labeled_dir / f"*_LabeledData_{cond}.csv")))
    )
    a = {Path(p).name.split("_")[0] for p in proc_names}
    b = {Path(p).name.split("_")[0] for p in lab_names}
    return sorted(a & b)


def _resolve_data_file(folder: Path, pid: str, cond: str, kind: str) -> Path | None:
    """Find processed/labeled CSV
    """
    pid = pid_key(pid)
    folder = Path(folder)
    if kind == "processed":
        candidates = [
            folder / f"{pid}_processed_{cond}.csv",
            folder / f"{pid}_{cond}_processed.csv",
        ]
        patterns = [
            f"{pid}*processed*{cond}*.csv",
            f"{pid}*{cond}*processed*.csv",
        ]
    else:
        candidates = [
            folder / f"{pid}_{cond}_LabeledData.csv",
            folder / f"{pid}_LabeledData_{cond}.csv",
        ]
        patterns = [
            f"{pid}*{cond}*LabeledData*.csv",
            f"{pid}*LabeledData*{cond}*.csv",
        ]

    for path in candidates:
        if path.exists():
            return path
    for pattern in patterns:
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_aligned(
    pid,
    cond: str,
    processed_dir: str | Path,
    labeled_dir: str | Path,
    channels: list[str] | None = None,
    robust_scale: bool = True,
    granularity: str = "binary",
    *,
    return_reason: bool = False,
):
    """
    y dtype: int for binary/discrete, float for continuous.
    """
    channels = channels or CHANNELS_8
    g = normalize_granularity(granularity)
    col = LABEL_COLS[g]
    pid = pid_key(pid)
    processed_dir, labeled_dir = Path(processed_dir), Path(labeled_dir)

    sp = _resolve_data_file(processed_dir, pid, cond, "processed")
    lp = _resolve_data_file(labeled_dir, pid, cond, "labeled")
    if sp is None or lp is None:
        missing = []
        if sp is None:
            missing.append(
                f"processed missing under {processed_dir} (want {pid}_processed_{cond}.csv)"
            )
        if lp is None:
            missing.append(
                f"labeled missing under {labeled_dir} (want {pid}_{cond}_LabeledData.csv)"
            )
        reason = "; ".join(missing)
        return (None, reason) if return_reason else None

    s = pd.read_csv(sp)
    s["time"] = pd.to_datetime(s["time"], utc=True)
    s = s.sort_values("time").reset_index(drop=True)
    t0 = s["time"].iloc[0]
    s["el"] = (s["time"] - t0).dt.total_seconds()

    l = pd.read_csv(lp, index_col=0)
    l["time"] = pd.to_datetime(l["time"], utc=True)
    l = l.sort_values("time").reset_index(drop=True)
    l["el"] = (l["time"] - t0).dt.total_seconds()
    l = l[(l["el"] >= s["el"].iloc[0]) & (l["el"] <= s["el"].iloc[-1])].reset_index(drop=True)
    if col not in l.columns:
        reason = f"label col {col!r} not in {lp.name} (cols={list(l.columns)[:12]})"
        return (None, reason) if return_reason else None
    if len(l) < 100:
        reason = f"too few labels after time align ({len(l)}) in {lp.name}"
        return (None, reason) if return_reason else None

    missing_ch = [c for c in channels if c not in s.columns]
    if missing_ch:
        reason = f"missing channels {missing_ch} in {sp.name}"
        return (None, reason) if return_reason else None

    X = pd.DataFrame(s[channels].to_numpy(float)).interpolate().bfill().ffill().to_numpy()
    if robust_scale:
        med = np.median(X, 0)
        mad = 1.4826 * np.median(np.abs(X - med), 0)
        mad[mad == 0] = 1.0
        Z = np.clip((X - med) / mad, -10, 10)
    else:
        Z = X

    y = pd.to_numeric(l[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(y)
    y = y[ok]
    if g != "continuous":
        y = y.astype(int)
    data = (s["el"].to_numpy(), Z, l["el"].to_numpy()[ok], y)
    return (data, "ok") if return_reason else data


def load_run_only(path: str | Path, channels: list[str] | None = None):
    """Load a single processed run"""
    channels = channels or CHANNELS_8
    d = pd.read_csv(path)
    t = pd.to_datetime(d["time"], utc=True)
    el = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    X = pd.DataFrame(d[channels].to_numpy(float)).interpolate().bfill().ffill().to_numpy()
    hz = len(el) / max(el[-1], 1e-9)
    return X, el, hz
