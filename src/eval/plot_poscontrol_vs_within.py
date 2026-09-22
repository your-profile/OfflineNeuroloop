"""Correlate positive-control Play-vs-Watch scores with within-subject decoding.

Per participant: x = positive-control LOPO-window score (macro-F1 preferred;
AUC if F1 is missing from older CSVs). y = within-subject metric for each
condition from ``within_subject`` results (macro-F1 when granularity is
discrete; AUC when binary).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _pid_key(x) -> str:
    s = str(x)
    return s.zfill(3) if s.isdigit() else s


def load_positive_control(path: Path | str) -> pd.DataFrame:
    """Return one row per participant from ``lopo_window`` rows."""
    df = pd.read_csv(path)
    if "test" in df.columns:
        df = df[df["test"].astype(str) == "lopo_window"].copy()
    pid_col = "held" if "held" in df.columns else "pid"
    df = df.rename(columns={pid_col: "pid"})
    df["pid"] = df["pid"].map(_pid_key)
    # Prefer macro-F1; fall back to AUC for older positive_control CSVs.
    if "f1" in df.columns and df["f1"].notna().any():
        df["pos_metric"] = pd.to_numeric(df["f1"], errors="coerce")
        df["pos_metric_name"] = "macro-F1"
    elif "auc" in df.columns:
        df["pos_metric"] = pd.to_numeric(df["auc"], errors="coerce")
        df["pos_metric_name"] = "AUC"
    else:
        raise ValueError(f"{path}: need an 'f1' or 'auc' column")
    return df[["pid", "pos_metric", "pos_metric_name", "n"]].dropna(subset=["pos_metric"])


def load_within_subject(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pid" not in df.columns and "held" in df.columns:
        df = df.rename(columns={"held": "pid"})
    if "condition" not in df.columns:
        if "conds" in df.columns:
            df["condition"] = df["conds"]
        elif "job" in df.columns:
            df["condition"] = df["job"]
        else:
            raise ValueError(f"{path}: need condition / conds / job column")
    df["pid"] = df["pid"].map(_pid_key)
    df["within_metric"] = pd.to_numeric(df["metric"], errors="coerce")
    if "metric_name" in df.columns and df["metric_name"].notna().any():
        name = str(df["metric_name"].dropna().iloc[0])
    else:
        g = str(df.get("granularity", pd.Series(["binary"])).iloc[0]).lower()
        name = {"binary": "AUC", "discrete": "macroF1", "continuous": "Spearman"}.get(g, "metric")
    df["within_metric_name"] = name
    return df.dropna(subset=["within_metric"])


def merge_pos_within(pos: pd.DataFrame, within: pd.DataFrame) -> pd.DataFrame:
    cols = ["pid", "condition", "within_metric", "within_metric_name"]
    for opt in ("significant", "in_rl_subset", "p", "n"):
        if opt in within.columns:
            cols.append(opt)
    merged = within[cols].merge(pos[["pid", "pos_metric", "pos_metric_name"]], on="pid", how="inner")
    return merged


def _corr_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 3:
        return f"n={len(x)}"
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    return f"n={len(x)}\nPearson r={r_p:.2f} (p={p_p:.3g})\nSpearman ρ={r_s:.2f} (p={p_s:.3g})"


def plot_poscontrol_vs_within(
    merged: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
    annotate_pids: bool = True,
) -> Path:
    """One scatter panel per within-subject condition."""
    conditions = list(dict.fromkeys(merged["condition"].astype(str)))
    n = len(conditions)
    if n == 0:
        raise ValueError("No overlapping participants/conditions to plot")

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    pos_name = str(merged["pos_metric_name"].iloc[0])
    within_name = str(merged["within_metric_name"].iloc[0])

    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        sub = merged[merged["condition"].astype(str) == cond]
        x = sub["pos_metric"].to_numpy(float)
        y = sub["within_metric"].to_numpy(float)

        rl = sub["in_rl_subset"].astype(bool) if "in_rl_subset" in sub.columns else pd.Series(False, index=sub.index)
        colors = ["#c44e52" if flag else "#4c72b0" for flag in rl]
        ax.scatter(x, y, c=colors, s=45, edgecolors="k", linewidths=0.4, zorder=3)

        if len(x) >= 2:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(min(x.min(), y.min(), 0.0), max(x.max(), y.max(), 1.0), 50)
            ax.plot(xs, np.polyval(coef, xs), color="0.35", lw=1.2, zorder=2)

        if annotate_pids:
            for _, row in sub.iterrows():
                ax.annotate(
                    str(row["pid"]),
                    (row["pos_metric"], row["within_metric"]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                    color="0.25",
                )

        ax.text(
            0.03,
            0.97,
            _corr_text(x, y),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )
        ax.set_title(str(cond))
        ax.set_xlabel(f"Positive control {pos_name}")
        ax.set_ylabel(f"Within-subject {within_name}")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="0.7", ls="--", lw=0.8)
        ax.axvline(0.5, color="0.7", ls="--", lw=0.8)
        ax.grid(alpha=0.25)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    if "in_rl_subset" in merged.columns and merged["in_rl_subset"].any():
        from matplotlib.patches import Patch

        fig.legend(
            handles=[
                Patch(facecolor="#c44e52", label="RL subset"),
                Patch(facecolor="#4c72b0", label="Other participants"),
            ],
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(
        title
        or f"Positive-control {pos_name} vs within-subject {within_name} by condition",
        y=1.06 if n > 1 else 1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
