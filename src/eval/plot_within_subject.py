"""Figures for within-subject (and optional LOPO) decoding results.

Designed for the paper justification of the RL participant subset:
highlight ``in_rl_subset`` subjects against the full cohort per condition.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DOMAIN_NAME = {"R": "Robot", "F": "Flappy", "L": "Lunar"}
TASK_NAME = {"passive": "Passive", "active": "Active", "pooled": "Pooled"}


def _metric_label(df: pd.DataFrame) -> str:
    if "metric_name" in df.columns and df["metric_name"].notna().any():
        return str(df["metric_name"].dropna().iloc[0])
    g = str(df["granularity"].iloc[0]).lower() if "granularity" in df.columns else "binary"
    return {"binary": "AUC", "discrete": "macro-F1", "continuous": "Spearman"}.get(g, "metric")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pid" not in df.columns and "held" in df.columns:
        df["pid"] = df["held"].map(lambda x: str(x).zfill(3) if str(x).isdigit() else str(x))
    if "condition" not in df.columns and "conds" in df.columns:
        df["condition"] = df["conds"]
    if "condition" not in df.columns and "job" in df.columns:
        df["condition"] = df["job"]
    if "in_rl_subset" not in df.columns:
        df["in_rl_subset"] = False
    else:
        df["in_rl_subset"] = df["in_rl_subset"].astype(bool)
    if "significant" not in df.columns and "p" in df.columns:
        df["significant"] = df["p"] < 0.05
    if "domain" not in df.columns:
        df["domain"] = df["condition"].astype(str).str[0]
    if "task" not in df.columns:
        df["task"] = df["condition"].astype(str).map(
            lambda c: "passive" if c.endswith("W") else ("active" if c.endswith("P") else "pooled")
        )
    df["pid"] = df["pid"].map(lambda x: str(x).zfill(3) if str(x).isdigit() else str(x))
    df["facet"] = df.apply(
        lambda r: f"{DOMAIN_NAME.get(r['domain'], r['domain'])} {TASK_NAME.get(r['task'], r['condition'])}",
        axis=1,
    )
    return df


def mark_rl_subset(df: pd.DataFrame, rl_pids: list | None) -> pd.DataFrame:
    df = _ensure_columns(df)
    if rl_pids:
        keys = {str(p).zfill(3) if str(p).isdigit() else str(p) for p in rl_pids}
        df["in_rl_subset"] = df["pid"].isin(keys)
    return df


def plot_per_participant_by_condition(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
    chance: float | None = 0.5,
) -> Path:
    """One panel per condition: participant bars with CI, RL subset highlighted."""
    df = _ensure_columns(df)
    metric = _metric_label(df)
    conditions = list(df["condition"].unique())
    n = len(conditions)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        sub = df[df["condition"] == cond].sort_values("metric", ascending=True)
        y = np.arange(len(sub))
        colors = ["#c44e52" if rl else "#4c72b0" for rl in sub["in_rl_subset"]]
        xerr = None
        if "lo" in sub.columns and "hi" in sub.columns:
            lo = sub["metric"] - sub["lo"]
            hi = sub["hi"] - sub["metric"]
            xerr = np.vstack([lo.clip(lower=0), hi.clip(lower=0)])
        ax.barh(y, sub["metric"], xerr=xerr, color=colors, alpha=0.85, ecolor="#333333", capsize=2)
        for yi, (_, row) in enumerate(sub.iterrows()):
            if row.get("significant"):
                ax.plot(row["metric"] + 0.02, yi, marker="*", color="k", markersize=7)
        ax.set_yticks(y)
        labels = [
            f"{pid}{' †' if rl else ''}"
            for pid, rl in zip(sub["pid"], sub["in_rl_subset"])
        ]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(str(cond))
        if chance is not None and metric.upper() in ("AUC",):
            ax.axvline(chance, color="0.5", ls="--", lw=1)
        ax.set_xlim(0, 1.05 if metric.upper() in ("AUC", "MACROF1", "MACRO-F1") else None)
        ax.grid(axis="x", alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="#c44e52", label="RL subset"),
        Patch(facecolor="#4c72b0", label="Other participants"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title or f"Within-subject {metric} by participant and condition", y=1.06, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_condition_summary(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    title: str | None = None,
    chance: float | None = 0.5,
) -> Path:
    """Strip + mean per condition; RL subset vs others overlaid."""
    df = _ensure_columns(df)
    metric = _metric_label(df)
    conditions = list(df["condition"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(conditions)), 4.2))

    rng = np.random.default_rng(0)
    for i, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        for rl_flag, color, marker in [
            (False, "#4c72b0", "o"),
            (True, "#c44e52", "D"),
        ]:
            pts = sub[sub["in_rl_subset"] == rl_flag]
            if pts.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(pts))
            ax.scatter(
                np.full(len(pts), i) + jitter,
                pts["metric"],
                c=color,
                marker=marker,
                s=45 if rl_flag else 30,
                alpha=0.85,
                zorder=3,
                edgecolors="k",
                linewidths=0.4,
            )
        ax.errorbar(
            i,
            sub["metric"].mean(),
            yerr=sub["metric"].std(ddof=1) / max(np.sqrt(len(sub)), 1) if len(sub) > 1 else 0,
            fmt="s",
            color="k",
            markersize=6,
            capsize=3,
            zorder=4,
        )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_xlabel("Condition")
    if chance is not None and metric.upper() == "AUC":
        ax.axhline(chance, color="0.5", ls="--", lw=1, label="chance")
    ax.set_title(title or f"Within-subject {metric} across conditions")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#c44e52", markersize=8, label="RL subset"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0", markersize=7, label="Other"),
        Line2D([0], [0], marker="s", color="k", markersize=6, label="Condition mean ± SEM"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rl_vs_others(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    condition: str | None = None,
    title: str | None = None,
    chance: float | None = 0.5,
) -> Path:
    """Box/strip comparing RL-selected participants vs the rest (paper justification)."""
    df = _ensure_columns(df)
    metric = _metric_label(df)
    if condition:
        df = df[df["condition"] == condition]
    if df.empty or not df["in_rl_subset"].any():
        raise ValueError("Need rows with in_rl_subset=True for RL vs others plot")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    groups = [("Other participants", False, "#4c72b0"), ("RL subset", True, "#c44e52")]
    data, positions, colors = [], [], []
    for i, (label, flag, color) in enumerate(groups):
        vals = df.loc[df["in_rl_subset"] == flag, "metric"].dropna().to_numpy()
        data.append(vals)
        positions.append(i)
        colors.append(color)
        if len(vals):
            jitter = np.random.default_rng(1).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, c=color, s=40, zorder=3, edgecolors="k", linewidths=0.4)

    bp = ax.boxplot(data, positions=positions, widths=0.45, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(metric)
    cond_txt = f" ({condition})" if condition else ""
    ax.set_title(title or f"Within-subject {metric}: RL subset vs others{cond_txt}")
    if chance is not None and metric.upper() == "AUC":
        ax.axhline(chance, color="0.5", ls="--", lw=1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(
    df: pd.DataFrame,
    out_dir: Path | str,
    *,
    stem: str = "within_subject",
    rl_pids: list | None = None,
    rl_vs_condition: str | None = "RW",
) -> list[Path]:
    """Write the standard paper figure set; return saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = mark_rl_subset(df, rl_pids)
    paths = []
    paths.append(
        plot_per_participant_by_condition(df, out_dir / f"{stem}_by_participant.pdf")
    )
    paths.append(plot_condition_summary(df, out_dir / f"{stem}_condition_summary.pdf"))
    # Prefer an existing condition for the RL-vs-others panel
    cond = rl_vs_condition
    if cond is None or cond not in set(df["condition"].astype(str)):
        # fall back to first condition that has both RL and non-RL points
        cond = None
        for c in df["condition"].unique():
            sub = df[df["condition"] == c]
            if sub["in_rl_subset"].any() and (~sub["in_rl_subset"]).any():
                cond = c
                break
    if cond is not None and df["in_rl_subset"].any():
        paths.append(
            plot_rl_vs_others(df, out_dir / f"{stem}_rl_vs_others.pdf", condition=cond)
        )
    # also save PNG previews
    for p in list(paths):
        png = p.with_suffix(".png")
        if p.suffix == ".pdf":
            # re-save via reading is awkward; duplicate call for png
            pass
    # regenerate png companions
    plot_per_participant_by_condition(df, out_dir / f"{stem}_by_participant.png")
    plot_condition_summary(df, out_dir / f"{stem}_condition_summary.png")
    if cond is not None and df["in_rl_subset"].any():
        plot_rl_vs_others(df, out_dir / f"{stem}_rl_vs_others.png", condition=cond)
        paths.append(out_dir / f"{stem}_rl_vs_others.png")
    paths.append(out_dir / f"{stem}_by_participant.png")
    paths.append(out_dir / f"{stem}_condition_summary.png")
    return paths
