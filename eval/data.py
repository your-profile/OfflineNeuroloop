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
    a = {Path(p).name.split("_")[0] for p in glob.glob(str(processed_dir / f"*_processed_{cond}.csv"))}
    b = {Path(p).name.split("_")[0] for p in glob.glob(str(labeled_dir / f"*_{cond}_LabeledData.csv"))}
    return sorted(a & b)


def load_aligned(
    pid,
    cond: str,
    processed_dir: str | Path,
    labeled_dir: str | Path,
    channels: list[str] | None = None,
    robust_scale: bool = True,
    granularity: str = "binary",
):
    """Return (ts, Z, tl, y) aligned in elapsed seconds, or None.

    y dtype: int for binary/discrete, float for continuous.
    """
    channels = channels or CHANNELS_8
    g = normalize_granularity(granularity)
    col = LABEL_COLS[g]
    pid = pid_key(pid)
    processed_dir, labeled_dir = Path(processed_dir), Path(labeled_dir)

    sp = processed_dir / f"{pid}_processed_{cond}.csv"
    lp = labeled_dir / f"{pid}_{cond}_LabeledData.csv"
    if not sp.exists() or not lp.exists():
        return None

    s = pd.read_csv(sp)
    s["time"] = pd.to_datetime(s["time"])
    s = s.sort_values("time").reset_index(drop=True)
    t0 = s["time"].iloc[0]
    s["el"] = (s["time"] - t0).dt.total_seconds()

    l = pd.read_csv(lp, index_col=0)
    l["time"] = pd.to_datetime(l["time"])
    l = l.sort_values("time").reset_index(drop=True)
    l["el"] = (l["time"] - t0).dt.total_seconds()
    l = l[(l["el"] >= s["el"].iloc[0]) & (l["el"] <= s["el"].iloc[-1])].reset_index(drop=True)
    if len(l) < 100 or col not in l.columns:
        return None

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
    return s["el"].to_numpy(), Z, l["el"].to_numpy()[ok], y


def load_run_only(path: str | Path, channels: list[str] | None = None):
    """Load a single processed run: (X, elapsed, hz)."""
    channels = channels or CHANNELS_8
    d = pd.read_csv(path)
    t = pd.to_datetime(d["time"])
    el = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    X = pd.DataFrame(d[channels].to_numpy(float)).interpolate().bfill().ffill().to_numpy()
    hz = len(el) / max(el[-1], 1e-9)
    return X, el, hz
