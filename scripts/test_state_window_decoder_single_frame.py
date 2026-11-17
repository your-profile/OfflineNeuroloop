#!/usr/bin/env python3
"""Test & visualize single-frame State -> Window Features -> Existing Decoder.

Evaluates ``src/models/state_window_decoder_single_frame.py`` (single-frame
``np.concatenate([observation, desired_goal])`` input of shape ``[N, 28]``).

Does NOT write any CSV files. Saves three PNG figures under
``results/figures/state_window_decoder_single_frame/``:
  1. ``*_stage1_feature_prediction.png``
  2. ``*_stage2_decoder_comparison.png``
  3. ``*_holdout_timeseries_preview.png``

Examples:
  python scripts/test_state_window_decoder_single_frame.py --data-path /path/to/ParticipantData
  python scripts/test_state_window_decoder_single_frame.py --data-path /path/to/ParticipantData --show
  python scripts/test_state_window_decoder_single_frame.py --synthetic --show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trial import (  # noqa: E402
    _episode_table,
    _keys,
    _resolve_data_roots,
    _split_per_subject,
)
from src.eval.cv import make_model  # noqa: E402
from src.eval.data import list_pids  # noqa: E402
from src.eval.settings import channel_setup, normalize_granularity  # noqa: E402
from src.models.decoder_bank import normalize_pid  # noqa: E402
from src.models.eval_compatible import (  # noqa: E402
    RobustWindowDecoder,
    _embargo_train_mask,
    _robust_stats,
    train_subject_eval_decoder,
)
from src.models.state_window_decoder_single_frame import (  # noqa: E402
    StateToWindowDecoder,
    SubjectStateDecoderBank,
    build_subject_state_window_dataset,
    format_state_representation,
)
from src.neural.loader import DataLoader  # noqa: E402

DEFAULT_DATA = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData"

PALETTE = {
    "real": "#2b5c8f",
    "ridge": "#55a868",
    "mlp": "#c44e52",
}


def _pid_int(pid: str):
    return int(pid) if str(pid).isdigit() else pid


def _as_key(k) -> tuple[str, int]:
    return (str(k[0]), int(k[1]))


def _feature_standardized_mae_vector(Z_true: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    """Column-wise MAE on standardized window features."""
    return np.mean(np.abs(Z_true - Z_pred), axis=0)


def _top_decoder_feature_indices(clf: Any, n_features: int, k: int = 2) -> np.ndarray:
    """Return indices of the top-k features weighted most strongly by Stage-2 ``clf``."""
    inner = getattr(clf, "model", clf)
    if hasattr(inner, "named_steps") and "ridge" in inner.named_steps:
        inner = inner.named_steps["ridge"]
    coef = getattr(inner, "coef_", None)
    if coef is not None:
        w = np.abs(np.asarray(coef, dtype=float))
        if w.ndim == 2:
            w = w.mean(axis=0)
        if w.size == n_features:
            return np.argsort(w)[::-1][:k]
    return np.arange(min(k, n_features))


def _feature_group_slices(n_channels: int, n_pairs: int) -> list[tuple[str, slice]]:
    """Map ``_featurize_seg`` columns to named feature groups."""
    c, p = int(n_channels), int(n_pairs)
    groups = [
        ("Mean", slice(0, c)),
        ("Std", slice(c, 2 * c)),
        ("Slope", slice(2 * c, 3 * c)),
    ]
    if p > 0:
        groups.append(("Pair Sum", slice(3 * c, 3 * c + p)))
        groups.append(("Pair Diff", slice(3 * c + p, 3 * c + 2 * p)))
    return groups


def _feature_names(channels: list[str], pairs: list[tuple[str, str]]) -> list[str]:
    return (
        [f"mean({ch})" for ch in channels]
        + [f"std({ch})" for ch in channels]
        + [f"slope({ch})" for ch in channels]
        + [f"sum({a},{b})" for a, b in pairs]
        + [f"diff({a},{b})" for a, b in pairs]
    )


def _continuous_signal(clf: Any, F: np.ndarray, granularity: str) -> np.ndarray:
    """Extract a 1D score/probability trace from ``clf`` for plotting."""
    g = normalize_granularity(granularity)
    if g == "continuous":
        return np.asarray(clf.predict(F), dtype=float).ravel()
    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(F), dtype=float)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    return np.asarray(clf.predict(F), dtype=float).ravel()


def _score_clf_predictions(
    clf: Any,
    F: np.ndarray,
    y_true: np.ndarray,
    granularity: str,
) -> dict[str, float | None]:
    """Score any fitted Stage-2 decoder on feature matrix ``F`` against ``y_true``."""
    g = normalize_granularity(granularity)
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(clf.predict(F)).ravel()

    if g == "continuous":
        yt_f, yp_f = yt.astype(float), yp.astype(float)
        if yt_f.size >= 2 and np.std(yt_f) > 0 and np.std(yp_f) > 0:
            rho_val, _ = spearmanr(yt_f, yp_f)
            rho = float(rho_val) if np.isfinite(rho_val) else None
        else:
            rho = None
        return {
            "spearman": rho,
            "r2": float(r2_score(yt_f, yp_f)) if yt_f.size >= 2 else None,
            "mse": float(mean_squared_error(yt_f, yp_f)),
            "acc": None,
            "f1": None,
            "auc": None,
        }

    yt_i, yp_i = yt.astype(int), yp.astype(int)
    acc = float(accuracy_score(yt_i, yp_i))
    f1 = float(f1_score(yt_i, yp_i, average="macro", zero_division=0))
    auc: float | None = None
    if len(np.unique(yt_i)) > 1:
        inner = getattr(clf, "model", clf)
        try:
            if g == "binary":
                if hasattr(inner, "decision_function"):
                    auc = float(roc_auc_score(yt_i, inner.decision_function(F)))
                elif hasattr(clf, "predict_proba"):
                    auc = float(roc_auc_score(yt_i, clf.predict_proba(F)[:, 1]))
            else:
                proba = clf.predict_proba(F)
                auc = float(roc_auc_score(yt_i, proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = None

    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "spearman": None,
        "r2": None,
        "mse": None,
    }


def _score_decoder_fidelity(
    clf: Any,
    F_true: np.ndarray,
    F_hat: np.ndarray,
    granularity: str,
) -> dict[str, float | None]:
    """Compare ``clf(F_hat)`` directly against ``clf(F_true)`` on the same holdout windows."""
    g = normalize_granularity(granularity)
    pred_real = np.asarray(clf.predict(F_true)).ravel()
    pred_surr = np.asarray(clf.predict(F_hat)).ravel()

    if g == "continuous":
        if len(pred_real) >= 2 and np.std(pred_real) > 0 and np.std(pred_surr) > 0:
            r_val, _ = pearsonr(pred_real.astype(float), pred_surr.astype(float))
            r = float(r_val) if np.isfinite(r_val) else None
        else:
            r = None
        return {
            "clf_agreement_acc": None,
            "clf_agreement_f1": None,
            "clf_score_pearson": r,
        }

    pr_i, ps_i = pred_real.astype(int), pred_surr.astype(int)
    agree_acc = float(accuracy_score(pr_i, ps_i))
    agree_f1 = float(f1_score(pr_i, ps_i, average="macro", zero_division=0))
    score_corr: float | None = None
    try:
        if hasattr(clf, "predict_proba"):
            p_real = np.asarray(clf.predict_proba(F_true))
            p_surr = np.asarray(clf.predict_proba(F_hat))
            v1, v2 = p_real.ravel(), p_surr.ravel()
            if np.std(v1) > 1e-12 and np.std(v2) > 1e-12:
                r_val, _ = pearsonr(v1, v2)
                score_corr = float(r_val) if np.isfinite(r_val) else None
    except Exception:
        score_corr = None

    return {
        "clf_agreement_acc": agree_acc,
        "clf_agreement_f1": agree_f1,
        "clf_score_pearson": score_corr,
    }


def evaluate_subject(
    pid: str,
    conditions: list[str],
    processed_dir: str,
    labeled_dir: str,
    task_df: pd.DataFrame,
    decoder_keys: set[tuple],
    agent_keys: set[tuple],
    *,
    regressors: list[str],
    granularity: str,
    window_s: float,
    step_s: float,
    embargo_s: float,
    rate_hz: float,
    temporal_shift: float,
    channels: list[str],
    pairs: list[tuple[str, str]],
    ridge_alpha: float,
    hidden_sizes: tuple[int, ...],
    min_holdout_f1: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Fit existing decoder + single-frame state->window decoders for one subject."""
    existing_clf, existing_rep = train_subject_eval_decoder(
        pid,
        conditions,
        processed_dir,
        labeled_dir,
        task_df,
        decoder_keys=decoder_keys,
        agent_keys=agent_keys,
        granularity=granularity,
        window_s=window_s,
        step_s=step_s,
        embargo_s=embargo_s,
        rate_hz=rate_hz,
        temporal_shift=temporal_shift,
        channels=channels,
        pairs=pairs,
        seed=seed,
    )
    if existing_clf is None:
        err = existing_rep.get("error", str(existing_rep)) if isinstance(existing_rep, dict) else str(existing_rep)
        return [], {}, f"existing decoder failed: {err}"

    bundle, status = build_subject_state_window_dataset(
        pid,
        conditions,
        processed_dir,
        labeled_dir,
        task_df,
        granularity=granularity,
        window_s=window_s,
        step_s=step_s,
        rate_hz=rate_hz,
        temporal_shift=temporal_shift,
        channels=channels,
        pairs=pairs,
    )
    if bundle is None:
        return [], {}, status.get("error", "no state-window bundle")

    S = bundle["S"]
    F = bundle["F"]
    y = bundle["y"]
    T = bundle["T"]
    ep_i = bundle["ep_i"]
    spans = bundle["spans"]
    g = bundle["granularity"]
    gap = float(window_s) + float(embargo_s)

    span_keys = [sp[0] for sp in spans]
    dec_keys_n = {_as_key(k) for k in decoder_keys}
    agt_keys_n = {_as_key(k) for k in agent_keys}

    is_dec = np.zeros(len(y), dtype=bool)
    is_te = np.zeros(len(y), dtype=bool)
    for i, si in enumerate(ep_i):
        if si < 0:
            continue
        key = _as_key(span_keys[si])
        if key in dec_keys_n:
            is_dec[i] = True
        if key in agt_keys_n:
            is_te[i] = True

    tr = is_dec & _embargo_train_mask(T, is_te, gap) if is_te.any() else is_dec
    te = is_te

    if tr.sum() < 15:
        return [], {}, f"too few train state-windows after embargo ({int(tr.sum())})"
    if te.sum() < 8:
        return [], {}, f"too few holdout state-windows ({int(te.sum())})"
    if g != "continuous" and len(np.unique(y[te])) < 2:
        return [], {}, "holdout windows have only a single class"

    real_metrics = _score_clf_predictions(existing_clf, F[te], y[te], g)
    if g != "continuous" and min_holdout_f1 > 0.0:
        rf1 = real_metrics["f1"]
        if rf1 is None or rf1 < min_holdout_f1:
            rf1_str = f"{rf1:.3f}" if rf1 is not None else "None"
            return [], {}, f"holdout_f1 {rf1_str} < min_holdout_f1={min_holdout_f1:.2f}"

    group_slices = _feature_group_slices(len(channels), len(pairs))

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {
        "pid": normalize_pid(pid),
        "F_te": F[te],
        "y_te": y[te],
        "T_te": T[te],
        "channels": list(channels),
        "pairs": list(pairs),
        "existing_clf": existing_clf,
        "real_trace": _continuous_signal(existing_clf, F[te], g),
        "by_reg": {},
    }

    for reg_type in regressors:
        state_dec = StateToWindowDecoder(
            regressor_type=reg_type,
            ridge_alpha=ridge_alpha,
            hidden_layer_sizes=hidden_sizes,
            seed=seed,
            granularity=g,
            neural_clf=existing_clf,
        )
        state_dec.fit(S[tr], F[tr])

        F_hat_te = state_dec.predict_features(S[te])
        feat_metrics = state_dec.score_feature_predictions(F[te], F_hat_te)

        surr_metrics = _score_clf_predictions(existing_clf, F_hat_te, y[te], g)
        fidelity = _score_decoder_fidelity(existing_clf, F[te], F_hat_te, g)

        assert np.allclose(state_dec.predict(S[te]), existing_clf.predict(F_hat_te))

        f_mu = F[tr].mean(axis=0)
        f_sd = F[tr].std(axis=0)
        f_sd[f_sd < 1e-8] = 1.0
        Z_true = (F[te] - f_mu) / f_sd
        Z_hat = (F_hat_te - f_mu) / f_sd
        col_mae = _feature_standardized_mae_vector(Z_true, Z_hat)

        group_maes = {}
        for g_name, sl in group_slices:
            vals = col_mae[sl]
            group_maes[f"mae_{g_name}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        artifacts["by_reg"][reg_type] = {
            "decoder": state_dec,
            "F_hat_te": F_hat_te,
            "Z_true": Z_true,
            "Z_hat": Z_hat,
            "col_mae": col_mae,
            "cos": feat_metrics["feature_cosine_sim"],
            "pred_trace": _continuous_signal(existing_clf, F_hat_te, g),
        }

        row = dict(
            pid=normalize_pid(pid),
            conditions="+".join(conditions),
            granularity=g,
            window_s=float(window_s),
            temporal_shift=float(temporal_shift),
            regressor=reg_type,
            state_dim=int(S.shape[1]),
            feature_dim=int(F.shape[1]),
            n_train=int(tr.sum()),
            n_holdout=int(te.sum()),
            feature_r2=feat_metrics["feature_r2"],
            feature_cosine_sim=feat_metrics["feature_cosine_sim"],
            feature_mse=feat_metrics["feature_mse"],
            feature_mae=feat_metrics["feature_mae"],
            pred_feat_f1=surr_metrics["f1"],
            pred_feat_auc=surr_metrics["auc"],
            pred_feat_acc=surr_metrics["acc"],
            pred_feat_spearman=surr_metrics["spearman"],
            pred_feat_r2=surr_metrics["r2"],
            real_fnirs_f1=real_metrics["f1"],
            real_fnirs_auc=real_metrics["auc"],
            real_fnirs_acc=real_metrics["acc"],
            real_fnirs_spearman=real_metrics["spearman"],
            real_fnirs_r2=real_metrics["r2"],
            clf_agreement_acc=fidelity["clf_agreement_acc"],
            clf_agreement_f1=fidelity["clf_agreement_f1"],
            clf_score_pearson=fidelity["clf_score_pearson"],
            **group_maes,
        )
        rows.append(row)

    return rows, artifacts, None


def _plot_stage1_features(
    df: pd.DataFrame,
    subject_artifacts: list[dict[str, Any]],
    regressors: list[str],
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 1: Stage-1 single-frame state -> window feature prediction quality."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    pids = sorted(df["pid"].unique())
    x = np.arange(len(pids))
    width = min(0.8 / max(len(regressors), 1), 0.55)
    for idx, reg in enumerate(regressors):
        sub = df[df["regressor"] == reg].set_index("pid").reindex(pids)
        offset = (idx - (len(regressors) - 1) / 2) * width
        color = PALETTE.get(reg, f"C{idx}")
        ax.bar(
            x + offset,
            sub["feature_cosine_sim"],
            width=width,
            color=color,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.5,
            label=f"{reg.upper()} (Cosine sim)",
        )
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=35 if len(pids) > 6 else 0, fontsize=8.5)
    ax.set_ylabel("Feature Cosine Similarity")
    ax.set_xlabel("Participant")
    ax.set_title("A. Stage-1 Feature Cosine Similarity by Subject")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    ax = axes[1]
    group_cols = [c for c in df.columns if c.startswith("mae_")]
    group_labels = [c.replace("mae_", "") for c in group_cols]
    gx = np.arange(len(group_cols))
    for idx, reg in enumerate(regressors):
        sub = df[df["regressor"] == reg]
        means = [sub[c].mean() for c in group_cols]
        sems = [
            sub[c].std(ddof=1) / np.sqrt(max(sub[c].notna().sum(), 1))
            if sub[c].notna().sum() > 1
            else 0.0
            for c in group_cols
        ]
        offset = (idx - (len(regressors) - 1) / 2) * width
        ax.bar(
            gx + offset,
            means,
            yerr=sems,
            width=width,
            color=PALETTE.get(reg, f"C{idx}"),
            alpha=0.85,
            capsize=3,
            edgecolor="k",
            linewidth=0.5,
            label=reg.upper(),
        )
    ax.set_xticks(gx)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylabel("Standardized MAE ± SEM (lower = better)")
    ax.set_title("B. Prediction Error by Window Feature Type")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    ax = axes[2]
    primary_reg = regressors[0]
    z_t_list, z_p_list = [], []
    for art in subject_artifacts:
        if primary_reg in art["by_reg"]:
            z_t_list.append(art["by_reg"][primary_reg]["Z_true"].ravel())
            z_p_list.append(art["by_reg"][primary_reg]["Z_hat"].ravel())
    if z_t_list:
        zt = np.concatenate(z_t_list)
        zp = np.concatenate(z_p_list)
        mask = np.isfinite(zt) & np.isfinite(zp)
        zt, zp = zt[mask], zp[mask]
        rng = np.random.default_rng(0)
        if len(zt) > 3500:
            pick = rng.choice(len(zt), size=3500, replace=False)
            zt_plot, zp_plot = zt[pick], zp[pick]
        else:
            zt_plot, zp_plot = zt, zp
        ax.scatter(
            zt_plot,
            zp_plot,
            s=10,
            alpha=0.25,
            color=PALETTE.get(primary_reg, "#55a868"),
            edgecolors="none",
        )
        lim = float(np.clip(np.percentile(np.abs(np.concatenate([zt_plot, zp_plot])), 99), 1.5, 6.0))
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="Ideal y = x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        mae_all = float(np.mean(np.abs(zt - zp))) if len(zt) else float("nan")
        ax.set_title(f"C. True vs Predicted Features ({primary_reg.upper()}, z-MAE={mae_all:.2f})")
    ax.set_xlabel("True Window Feature (z-score)")
    ax.set_ylabel("State-Predicted Window Feature (z-score)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(f"{title_prefix} — Single-Frame Stage 1: State → fNIRS Window Feature Prediction", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_stage2_decoder(
    df: pd.DataFrame,
    regressors: list[str],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 2: Downstream classification on predicted features vs real fNIRS."""
    g = normalize_granularity(granularity)
    if g == "continuous":
        pred_col, real_col, metric_label, chance = (
            "pred_feat_spearman",
            "real_fnirs_spearman",
            "Spearman ρ",
            0.0,
        )
    else:
        pred_col, real_col, metric_label, chance = (
            "pred_feat_f1",
            "real_fnirs_f1",
            "Holdout Macro-F1",
            1.0 / 3.0 if g == "discrete" else 0.5,
        )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    pids = sorted(df["pid"].unique())
    x = np.arange(len(pids))
    n_bars = 1 + len(regressors)
    width = min(0.8 / n_bars, 0.32)

    real_by_pid = df.groupby("pid")[real_col].first().reindex(pids)
    ax.bar(
        x - ((n_bars - 1) / 2) * width,
        real_by_pid,
        width=width,
        color=PALETTE["real"],
        alpha=0.9,
        edgecolor="k",
        linewidth=0.5,
        label="Real fNIRS → Existing Decoder",
    )
    for idx, reg in enumerate(regressors, start=1):
        sub = df[df["regressor"] == reg].set_index("pid").reindex(pids)
        offset = (idx - (n_bars - 1) / 2) * width
        ax.bar(
            x + offset,
            sub[pred_col],
            width=width,
            color=PALETTE.get(reg, f"C{idx}"),
            alpha=0.85,
            edgecolor="k",
            linewidth=0.5,
            label=f"Single-Frame State ({reg.upper()}) → Existing Decoder",
        )
    ax.axhline(chance, color="0.45", ls="--", lw=1, label=f"Chance ({chance:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=35 if len(pids) > 6 else 0, fontsize=8.5)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Participant")
    if g != "continuous":
        ax.set_ylim(0.0, 1.05)
    ax.set_title(f"A. {metric_label} by Participant")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="best")

    ax = axes[1]
    lo_val = 0.0 if g != "continuous" else -0.6
    hi_val = 1.02 if g != "continuous" else 1.02
    for idx, reg in enumerate(regressors):
        sub = df[df["regressor"] == reg].dropna(subset=[real_col, pred_col])
        ax.scatter(
            sub[real_col],
            sub[pred_col],
            s=55,
            color=PALETTE.get(reg, f"C{idx}"),
            alpha=0.88,
            edgecolors="k",
            linewidths=0.6,
            label=f"{reg.upper()} (mean={sub[pred_col].mean():.3f})",
        )
        for _, r_row in sub.iterrows():
            ax.annotate(
                str(r_row["pid"]),
                (r_row[real_col], r_row[pred_col]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                alpha=0.75,
            )
    ax.plot([lo_val, hi_val], [lo_val, hi_val], "k--", lw=1.1, label="Parity with Real fNIRS")
    ax.axhline(chance, color="0.65", ls=":", lw=1)
    ax.axvline(chance, color="0.65", ls=":", lw=1)
    ax.set_xlim(lo_val, hi_val)
    ax.set_ylim(lo_val, hi_val)
    ax.set_xlabel(f"Real fNIRS → Existing Decoder ({metric_label})")
    ax.set_ylabel(f"Single-Frame State → F_hat → Existing Decoder ({metric_label})")
    ax.set_title("B. Surrogate vs Real fNIRS Parity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    ax = axes[2]
    fidelity_metrics = (
        [("clf_score_pearson", "Output Pearson r")]
        if g == "continuous"
        else [("clf_agreement_acc", "Class Agreement"), ("clf_score_pearson", "Proba Pearson r")]
    )
    fx = np.arange(len(fidelity_metrics))
    rng = np.random.default_rng(1)
    box_step = min(0.75 / max(len(regressors), 1), 0.34)
    box_w = box_step * 0.78
    for idx, reg in enumerate(regressors):
        sub = df[df["regressor"] == reg]
        offset = (idx - (len(regressors) - 1) / 2) * box_step
        color = PALETTE.get(reg, f"C{idx}")
        for m_i, (col, _) in enumerate(fidelity_metrics):
            vals = sub[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            pos = fx[m_i] + offset
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
            for median in bp["medians"]:
                median.set_color("k")
            jitter = rng.uniform(-box_w * 0.25, box_w * 0.25, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                c=color,
                s=34,
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
                label=reg.upper() if m_i == 0 else None,
            )
    ax.set_xticks(fx)
    ax.set_xticklabels([lbl for _, lbl in fidelity_metrics])
    ax.set_ylabel("Fidelity to Existing Decoder(Real F)")
    ax.set_ylim(-0.2 if g == "continuous" else 0.0, 1.05)
    ax.set_title("C. Fidelity to Existing Neural Decoder")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    fig.suptitle(
        f"{title_prefix} — Single-Frame Stage 2: Predicted Window Features Inside Existing Decoder",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_pooled_roc(
    subject_artifacts: list[dict[str, Any]],
    regressors: list[str],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path | None:
    """Standalone 1:1 square Pooled Holdout ROC figure (single plot, not a panel)."""
    g = normalize_granularity(granularity)
    if g != "binary":
        return None

    y_all = np.concatenate([a["y_te"].astype(int) for a in subject_artifacts])
    if len(np.unique(y_all)) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    p_real_all = np.concatenate([a["real_trace"] for a in subject_artifacts])
    series = [("real", "Real fNIRS → Existing Decoder", PALETTE["real"], p_real_all)]
    for reg in regressors:
        p_r = np.concatenate([a["by_reg"][reg]["pred_trace"] for a in subject_artifacts])
        series.append((reg, f"Single-Frame ({reg.upper()}) → Existing Decoder", PALETTE.get(reg, "#55a868"), p_r))

    for _, label, color, p_vec in series:
        fpr, tpr, _ = roc_curve(y_all, p_vec)
        auc_val = float(roc_auc_score(y_all, p_vec))
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.65, label="Chance (AUC = 0.500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate (Pooled Holdout Windows)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate", fontsize=10.5)
    ax.set_title(
        f"{title_prefix}\nPooled Holdout ROC (n={len(subject_artifacts)} Screened Subjects, N={len(y_all)} Windows)",
        fontsize=11,
        pad=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_cohort_diagnostics(
    df: pd.DataFrame,
    subject_artifacts: list[dict[str, Any]],
    regressors: list[str],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 3: Pooled probability separation and screened participants heatmap."""
    g = normalize_granularity(granularity)
    primary_reg = regressors[0]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, max(4.8, 0.42 * df["pid"].nunique() + 1.8)),
        gridspec_kw={"width_ratios": [1.05, 1.25]},
    )

    # Panel A: Pooled Class-Conditional Probability Separation Across Screened Holdout Windows
    ax = axes[0]
    y_all = np.concatenate([a["y_te"] for a in subject_artifacts])
    if g != "continuous":
        classes = sorted(np.unique(y_all.astype(int)))
        series_list = [("real", "Real fNIRS", PALETTE["real"])] + [
            (reg, f"Single-Frame ({reg.upper()})", PALETTE.get(reg, f"C{i}"))
            for i, reg in enumerate(regressors)
        ]
        cx = np.arange(len(classes))
        step_w = min(0.76 / len(series_list), 0.28)
        for s_idx, (s_key, s_lbl, color) in enumerate(series_list):
            p_pool = (
                np.concatenate([a["real_trace"] for a in subject_artifacts])
                if s_key == "real"
                else np.concatenate([a["by_reg"][s_key]["pred_trace"] for a in subject_artifacts])
            )
            offset = (s_idx - (len(series_list) - 1) / 2) * step_w
            for c_i, cls_val in enumerate(classes):
                vals = p_pool[y_all.astype(int) == cls_val]
                if len(vals) == 0:
                    continue
                pos = cx[c_i] + offset
                bp = ax.boxplot(
                    [vals],
                    positions=[pos],
                    widths=step_w * 0.82,
                    patch_artist=True,
                    showfliers=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.45)
                for med in bp["medians"]:
                    med.set_color("k")
                ax.scatter(
                    [pos],
                    [float(np.mean(vals))],
                    color=color,
                    marker="D",
                    s=30,
                    edgecolors="k",
                    linewidths=0.6,
                    zorder=4,
                    label=s_lbl if c_i == 0 else None,
                )
        ax.axhline(0.5, color="0.4", ls="--", lw=1.0, label="0.50 Threshold")
        ax.set_xticks(cx)
        cls_names = (
            ["True Optimal (y=0)", "True Suboptimal (y=1)"]
            if classes == [0, 1]
            else [f"True y={c}" for c in classes]
        )
        ax.set_xticklabels(cls_names)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Predicted P(class=1) Across Screened Subjects")
        ax.set_title(f"A. Pooled Probability Separation (N={len(y_all)} windows)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    else:
        p_real = np.concatenate([a["real_trace"] for a in subject_artifacts])
        p_surr = np.concatenate([a["by_reg"][primary_reg]["pred_trace"] for a in subject_artifacts])
        ax.boxplot(
            [np.abs(y_all - p_real), np.abs(y_all - p_surr)],
            positions=[0, 1],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Real fNIRS", f"Single-Frame ({primary_reg.upper()})"])
        ax.set_ylabel("|y_true - y_pred| Across All Windows")
        ax.set_title("A. Pooled Absolute Error Distribution")
        ax.grid(axis="y", alpha=0.3)

    # Panel B: Screened Participants Multi-Metric Heatmap
    ax = axes[1]
    sub = df[df["regressor"] == primary_reg].copy()
    sort_col = "real_fnirs_spearman" if g == "continuous" else "real_fnirs_f1"
    sub = sub.sort_values(sort_col, ascending=False).set_index("pid")
    sorted_pids = list(sub.index)

    if g == "continuous":
        col_defs = [
            ("Real ρ", sub["real_fnirs_spearman"]),
            ("State ρ", sub["pred_feat_spearman"]),
            ("Score r", sub["clf_score_pearson"]),
            ("Cos Sim", sub["feature_cosine_sim"]),
        ]
        vmin, vmax = -0.4, 0.8
    else:
        col_defs = [
            ("Real F1", sub["real_fnirs_f1"]),
            ("State F1", sub["pred_feat_f1"]),
            ("Real AUC", sub["real_fnirs_auc"]),
            ("State AUC", sub["pred_feat_auc"]),
            ("Clf Agree", sub["clf_agreement_acc"]),
            ("Cos Sim", sub["feature_cosine_sim"]),
        ]
        vmin, vmax = 0.20, 0.85

    mat = np.column_stack([s.to_numpy(dtype=float) for _, s in col_defs])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    for r_i in range(mat.shape[0]):
        for c_j in range(mat.shape[1]):
            v = mat[r_i, c_j]
            txt = f"{v:.2f}" if np.isfinite(v) else "n/a"
            ax.text(
                c_j,
                r_i,
                txt,
                ha="center",
                va="center",
                fontsize=8.0,
                color="k",
                fontweight="bold" if (g != "continuous" and c_j in (1, 3, 4) and np.isfinite(v) and v >= 0.60) else "normal",
            )

    ax.set_xticks(np.arange(len(col_defs)))
    ax.set_xticklabels([c[0] for c in col_defs], rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(sorted_pids)))
    ax.set_yticklabels([str(p) for p in sorted_pids], fontsize=8.5)
    ax.set_title(f"B. Screened Participants Heatmap (n={len(sorted_pids)})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title_prefix} — Single-Frame Screened Cohort Diagnostics (n={len(sorted_pids)} Participants)",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def generate_all_visualizations(
    df: pd.DataFrame,
    subject_artifacts: list[dict[str, Any]],
    regressors: list[str],
    granularity: str,
    out_dir: Path,
    stem: str,
    title_prefix: str,
    show: bool = False,
) -> list[Path]:
    paths = [
        _plot_stage1_features(
            df,
            subject_artifacts,
            regressors,
            out_dir / f"{stem}_stage1_feature_prediction.png",
            title_prefix,
        ),
        _plot_stage2_decoder(
            df,
            regressors,
            granularity,
            out_dir / f"{stem}_stage2_decoder_comparison.png",
            title_prefix,
        ),
        _plot_cohort_diagnostics(
            df,
            subject_artifacts,
            regressors,
            granularity,
            out_dir / f"{stem}_cohort_diagnostics.png",
            title_prefix,
        ),
    ]
    roc_fig = _plot_pooled_roc(
        subject_artifacts,
        regressors,
        granularity,
        out_dir / f"{stem}_pooled_roc.png",
        title_prefix,
    )
    if roc_fig is not None:
        paths.append(roc_fig)
    if show:
        plt.show()
    else:
        plt.close("all")
    return paths


def run_synthetic_smoke_test(
    regressors: list[str],
    out_dir: Path,
    granularity: str = "binary",
    seed: int = 42,
    show: bool = False,
) -> int:
    """Self-contained synthetic single-frame Fetch evaluation + visualizations (no CSV saved)."""
    print("=" * 76)
    print("SYNTHETIC SINGLE-FRAME FETCH TEST: [observation(25), desired_goal(3)] -> Window Features -> Existing Decoder")
    print("=" * 76)

    channels, pairs, _ = channel_setup({"drop_suspect_channels": False})
    n_channels = len(channels)
    group_slices = _feature_group_slices(len(channels), len(pairs))
    win_steps = int(round(6.0 * 5.2))
    obs_dim, goal_dim = 25, 3
    g = normalize_granularity(granularity)

    all_rows: list[dict[str, Any]] = []
    subject_artifacts: list[dict[str, Any]] = []
    synth_pids = ["010", "011", "018", "019", "020", "022"]

    for s_idx, pid in enumerate(synth_pids):
        sub_rng = np.random.default_rng(seed + s_idx * 17)
        n_samples = 280
        split = int(0.65 * n_samples)
        noise_scale = 0.28 + 0.08 * s_idx

        observations = sub_rng.normal(size=(n_samples, obs_dim)).astype(np.float32)
        desired_goals = sub_rng.normal(size=(n_samples, goal_dim)).astype(np.float32)
        concat_states = format_state_representation(observations, desired_goals)

        dist = np.linalg.norm(observations[:, :3] - desired_goals, axis=1)
        latent_opt = (np.median(dist) - dist) / (np.std(dist) + 1e-8)

        W_state = sub_rng.normal(scale=0.35, size=(obs_dim + goal_dim, n_channels))
        t_grid = np.arange(win_steps, dtype=float) / 5.2
        raw_windows = []
        for i in range(n_samples):
            base = concat_states[i] @ W_state
            slope = 0.25 * latent_opt[i] * np.array([1, 1, 0.5, 0.5, -1, -1, -0.5, -0.5])[:n_channels]
            win = (
                base[None, :]
                + t_grid[:, None] * slope[None, :]
                + sub_rng.normal(scale=noise_scale, size=(win_steps, n_channels))
            )
            raw_windows.append(win)

        med, mad = _robust_stats(np.vstack(raw_windows))
        if g == "continuous":
            y = latent_opt + sub_rng.normal(scale=0.25, size=n_samples)
        elif g == "discrete":
            edges = np.quantile(latent_opt, [1 / 3, 2 / 3])
            y = np.digitize(latent_opt, edges).astype(int)
        else:
            y = (latent_opt > 0.0).astype(int)

        dummy_dec = RobustWindowDecoder(
            model=None, med=med, mad=mad, channels=list(channels), pairs=list(pairs), granularity=g
        )
        F = np.vstack([dummy_dec.featurize_raw(w) for w in raw_windows])

        stage2_model = make_model(g)
        stage2_model.fit(F[:split], y[:split])
        existing_clf = RobustWindowDecoder(
            model=stage2_model,
            med=med,
            mad=mad,
            channels=list(channels),
            pairs=list(pairs),
            granularity=g,
            n_train=split,
            n_holdout=n_samples - split,
        )

        real_metrics = _score_clf_predictions(existing_clf, F[split:], y[split:], g)
        f_mu = F[:split].mean(axis=0)
        f_sd = F[:split].std(axis=0)
        f_sd[f_sd < 1e-8] = 1.0

        art: dict[str, Any] = {
            "pid": pid,
            "F_te": F[split:],
            "y_te": y[split:],
            "T_te": np.arange(n_samples - split, dtype=float),
            "channels": list(channels),
            "pairs": list(pairs),
            "existing_clf": existing_clf,
            "real_trace": _continuous_signal(existing_clf, F[split:], g),
            "by_reg": {},
        }

        for reg_type in regressors:
            dec = StateToWindowDecoder(
                regressor_type=reg_type,
                ridge_alpha=10.0,
                hidden_layer_sizes=(128, 64),
                seed=seed,
                granularity=g,
                neural_clf=existing_clf,
            )
            dec.fit(observations[:split], F[:split], desired_goal=desired_goals[:split])

            single_dict = {"observation": observations[split], "desired_goal": desired_goals[split]}
            y1, _, f1_vec = dec.predict_with_features(single_dict)
            assert np.allclose(f1_vec, dec.predict_features(observations[split], desired_goals[split]))
            assert y1.shape == (1,)

            F_hat_te = dec.predict_features(observations[split:], desired_goals[split:])
            feat_m = dec.score_feature_predictions(F[split:], F_hat_te)
            Z_true = (F[split:] - f_mu) / f_sd
            Z_hat = (F_hat_te - f_mu) / f_sd
            col_mae = _feature_standardized_mae_vector(Z_true, Z_hat)
            surr_m = _score_clf_predictions(existing_clf, F_hat_te, y[split:], g)
            fid_m = _score_decoder_fidelity(existing_clf, F[split:], F_hat_te, g)

            group_maes = {
                f"mae_{g_name}": float(np.nanmean(col_mae[sl])) for g_name, sl in group_slices
            }
            art["by_reg"][reg_type] = {
                "decoder": dec,
                "F_hat_te": F_hat_te,
                "Z_true": Z_true,
                "Z_hat": Z_hat,
                "col_mae": col_mae,
                "cos": feat_m["feature_cosine_sim"],
                "pred_trace": _continuous_signal(existing_clf, F_hat_te, g),
            }

            all_rows.append(
                dict(
                    pid=pid,
                    conditions="SYNTH_FETCH",
                    granularity=g,
                    regressor=reg_type,
                    state_dim=obs_dim + goal_dim,
                    feature_dim=F.shape[1],
                    n_train=split,
                    n_holdout=n_samples - split,
                    feature_r2=feat_m["feature_r2"],
                    feature_cosine_sim=feat_m["feature_cosine_sim"],
                    feature_mse=feat_m["feature_mse"],
                    feature_mae=feat_m["feature_mae"],
                    pred_feat_f1=surr_m["f1"],
                    pred_feat_auc=surr_m["auc"],
                    pred_feat_acc=surr_m["acc"],
                    pred_feat_spearman=surr_m["spearman"],
                    pred_feat_r2=surr_m["r2"],
                    real_fnirs_f1=real_metrics["f1"],
                    real_fnirs_auc=real_metrics["auc"],
                    real_fnirs_acc=real_metrics["acc"],
                    real_fnirs_spearman=real_metrics["spearman"],
                    real_fnirs_r2=real_metrics["r2"],
                    clf_agreement_acc=fid_m["clf_agreement_acc"],
                    clf_agreement_f1=fid_m["clf_agreement_f1"],
                    clf_score_pearson=fid_m["clf_score_pearson"],
                    **group_maes,
                )
            )
        subject_artifacts.append(art)

    first_reg = regressors[0]
    bank = SubjectStateDecoderBank(
        {a["pid"]: a["by_reg"][first_reg]["decoder"] for a in subject_artifacts},
        mode="ensemble",
        granularity=g,
    )
    assert bank.predict_features(observations[:4], desired_goals[:4]).shape == (4, F.shape[1])

    df = pd.DataFrame(all_rows)
    _print_summary(df, g)

    saved_figs = generate_all_visualizations(
        df=df,
        subject_artifacts=subject_artifacts,
        regressors=regressors,
        granularity=g,
        out_dir=out_dir,
        stem=f"single_frame_synthetic_fetch_{g}",
        title_prefix=f"Synthetic Fetch Single-Frame ({g.capitalize()})",
        show=show,
    )
    print("\nSaved visualizations (no CSV written):")
    for p in saved_figs:
        print(f"  {p}")
    return 0


def _parse_float_grid(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty float grid: {raw!r}")
    return vals


def _plot_window_shift_sweep(
    sweep_df: pd.DataFrame,
    regressors: list[str],
    granularity: str,
    windows: list[float],
    shifts: list[float],
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Plot 2D (window_s x temporal_shift) heatmaps for Real fNIRS and each Single-Frame regressor."""
    g = normalize_granularity(granularity)
    if g == "continuous":
        m1_pred, m1_real, m1_lbl = "pred_feat_spearman", "real_fnirs_spearman", "Spearman ρ"
        m2_pred, m2_real, m2_lbl = "clf_score_pearson", "real_fnirs_r2", "Score r / Real R²"
        vmin, vmax = -0.3, 0.8
    else:
        m1_pred, m1_real, m1_lbl = "pred_feat_f1", "real_fnirs_f1", "Holdout Macro-F1"
        m2_pred, m2_real, m2_lbl = "pred_feat_auc", "real_fnirs_auc", "Holdout AUC"
        vmin, vmax = 0.35, 0.85

    col_specs = [("real", "Real fNIRS → Existing Decoder")] + [
        (reg, f"Single-Frame ({reg.upper()}) → Existing Decoder") for reg in regressors
    ]
    n_cols = len(col_specs)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.6 * n_cols, max(5.6, 1.4 * len(windows) + 3.0)), squeeze=False)

    for c_idx, (key, col_title) in enumerate(col_specs):
        if key == "real":
            base = sweep_df.drop_duplicates(subset=["window_s", "temporal_shift", "pid"])
            c1, c2 = m1_real, m2_real
        else:
            base = sweep_df[sweep_df["regressor"] == key]
            c1, c2 = m1_pred, m2_pred

        mat1 = np.full((len(windows), len(shifts)), np.nan, dtype=float)
        mat2 = np.full((len(windows), len(shifts)), np.nan, dtype=float)
        cnts = np.zeros((len(windows), len(shifts)), dtype=int)
        ag_mat = np.full((len(windows), len(shifts)), np.nan, dtype=float)

        for wi, w_val in enumerate(windows):
            for si, s_val in enumerate(shifts):
                cell = base[(base["window_s"] == w_val) & (base["temporal_shift"] == s_val)]
                cnts[wi, si] = int(cell["pid"].nunique())
                if len(cell):
                    if cell[c1].notna().any():
                        mat1[wi, si] = float(cell[c1].mean())
                    if cell[c2].notna().any():
                        mat2[wi, si] = float(cell[c2].mean())
                    if key != "real" and "clf_agreement_acc" in cell.columns and cell["clf_agreement_acc"].notna().any():
                        ag_mat[wi, si] = float(cell["clf_agreement_acc"].mean())

        valid_mask = np.isfinite(mat1) & (cnts >= 2)
        if not valid_mask.any():
            valid_mask = np.isfinite(mat1)
        best_rc = None
        if valid_mask.any():
            masked = np.where(valid_mask, mat1, -np.inf)
            best_rc = np.unravel_index(int(np.argmax(masked)), mat1.shape)

        ax0 = axes[0, c_idx]
        im0 = ax0.imshow(mat1, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        for wi in range(len(windows)):
            for si in range(len(shifts)):
                v, n_k = mat1[wi, si], cnts[wi, si]
                star = " ★" if best_rc == (wi, si) else ""
                txt = f"{v:.2f}{star}\n(n={n_k})" if np.isfinite(v) else "—"
                ax0.text(si, wi, txt, ha="center", va="center", fontsize=8.0, fontweight="bold" if best_rc == (wi, si) else "normal")
        ax0.set_xticks(range(len(shifts)))
        ax0.set_xticklabels([f"{s:g}s" for s in shifts])
        ax0.set_yticks(range(len(windows)))
        ax0.set_yticklabels([f"{w:g}s" for w in windows])
        ax0.set_xlabel("Temporal Shift")
        if c_idx == 0:
            ax0.set_ylabel(f"Window Size\n({m1_lbl})")
        ax0.set_title(f"{col_title}\n{m1_lbl}")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[1, c_idx]
        im1 = ax1.imshow(mat2, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=min(vmax + 0.08, 0.95))
        for wi in range(len(windows)):
            for si in range(len(shifts)):
                v, ag = mat2[wi, si], ag_mat[wi, si]
                if not np.isfinite(v):
                    txt = "—"
                elif key != "real" and np.isfinite(ag):
                    txt = f"{v:.2f}\nag={ag:.2f}"
                else:
                    txt = f"{v:.2f}"
                ax1.text(si, wi, txt, ha="center", va="center", fontsize=8.0)
        ax1.set_xticks(range(len(shifts)))
        ax1.set_xticklabels([f"{s:g}s" for s in shifts])
        ax1.set_yticks(range(len(windows)))
        ax1.set_yticklabels([f"{w:g}s" for w in windows])
        ax1.set_xlabel("Temporal Shift")
        if c_idx == 0:
            ax1.set_ylabel(f"Window Size\n({m2_lbl})")
        ax1.set_title(f"{m2_lbl} (+ Agree)" if (key != "real" and g != "continuous") else m2_lbl)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix} — Window Size × Temporal Shift Sweep (★ = Best {m1_lbl})", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _print_sweep_leaderboard(sweep_df: pd.DataFrame, granularity: str) -> tuple[float, float]:
    g = normalize_granularity(granularity)
    primary_col = "pred_feat_spearman" if g == "continuous" else "pred_feat_f1"
    sec_col = "clf_score_pearson" if g == "continuous" else "pred_feat_auc"
    real_col = "real_fnirs_spearman" if g == "continuous" else "real_fnirs_f1"

    agg = (
        sweep_df.groupby(["window_s", "temporal_shift"])
        .agg(
            n_kept=("pid", "nunique"),
            pred_metric=(primary_col, "mean"),
            real_metric=(real_col, "mean"),
            sec_metric=(sec_col, "mean"),
            agree=("clf_agreement_acc", "mean"),
            feat_mae=("feature_mae", "mean"),
        )
        .sort_values(["pred_metric", "sec_metric"], ascending=False)
    )
    print("\n" + "=" * 84)
    print(f"SINGLE-FRAME SWEEP LEADERBOARD (ranked by {primary_col}):")
    print("=" * 84)
    print(agg.round(4).to_string())

    reset = agg.reset_index()
    pool = reset[reset["n_kept"] >= 2] if (reset["n_kept"] >= 2).any() else reset
    best_row = pool.iloc[0]
    best_w, best_s = float(best_row["window_s"]), float(best_row["temporal_shift"])
    print(
        f"\n>>> Best (window_s, shift_s) = ({best_w:g}s, {best_s:g}s) "
        f"[n_kept={int(best_row['n_kept'])}, {primary_col}={best_row['pred_metric']:.4f}]"
    )
    return best_w, best_s


def _print_summary(df: pd.DataFrame, granularity: str) -> None:
    g = normalize_granularity(granularity)
    print("\n" + "=" * 76)
    print("SINGLE-FRAME SUMMARY ACROSS PARTICIPANTS (Holdout Episodes)")
    print("=" * 76)

    stage1_cols = ["feature_cosine_sim", "feature_mae", "feature_mse", "feature_r2"]
    print("\n1) Stage-1 Window Feature Prediction Quality (mean across subjects):")
    print(df.groupby("regressor")[stage1_cols].mean().round(4).to_string())

    if g == "continuous":
        stage2_cols = [
            "pred_feat_spearman",
            "real_fnirs_spearman",
            "pred_feat_r2",
            "real_fnirs_r2",
            "clf_score_pearson",
        ]
    else:
        stage2_cols = [
            "pred_feat_f1",
            "real_fnirs_f1",
            "pred_feat_auc",
            "real_fnirs_auc",
            "clf_agreement_acc",
            "clf_score_pearson",
        ]

    print("\n2) Stage-2 Existing Decoder Performance (Predicted Features vs Real fNIRS):")
    print(df.groupby("regressor")[stage2_cols].mean().round(4).to_string())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA))
    p.add_argument("--conditions", default="RW", help="Comma-separated conditions, e.g. RW or RW,RP")
    p.add_argument("--pids", default="", help="Comma-separated participant IDs (default: all available)")
    p.add_argument("--granularity", default="binary", choices=["binary", "ternary", "discrete", "continuous"])
    p.add_argument(
        "--min-holdout-f1",
        type=float,
        default=0.55,
        help="Drop participants whose real-fNIRS Stage-2 holdout macro-F1 is below this threshold (default: 0.55)",
    )
    p.add_argument("--fraction", type=float, default=0.65, help="Fraction of early episodes used to fit decoders")
    p.add_argument(
        "--windows",
        default="4.0,6.0,8.0",
        help="Comma-separated window durations (s) to sweep (default: 4.0,6.0,8.0)",
    )
    p.add_argument(
        "--shifts",
        default="0.0,2.0,4.0,5.0,6.0",
        help="Comma-separated temporal shifts (s) to sweep (default: 0.0,2.0,4.0,5.0,6.0)",
    )
    p.add_argument("--window-s", type=float, default=None, help="Single window_s override (disables --windows grid)")
    p.add_argument("--step-s", type=float, default=1.0)
    p.add_argument("--shift-s", type=float, default=None, help="Single shift_s override (disables --shifts grid)")
    p.add_argument("--embargo-s", type=float, default=2.0)
    p.add_argument("--rate-hz", type=float, default=5.2)
    p.add_argument("--drop-suspect-channels", action="store_true", help="Use intensity-only channels")
    p.add_argument(
        "--regressors",
        default="ridge,mlp",
        help="Comma-separated Stage-1 regressors (default: 'ridge,mlp')",
    )
    p.add_argument("--hidden-sizes", default="128,64", help="Comma-separated MLP hidden layer sizes")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "state_window_decoder_single_frame",
        help="Directory where PNG figures are written",
    )
    p.add_argument("--show", action="store_true", help="Also open interactive matplotlib windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="Run synthetic Fetch test + visualizations")
    args = p.parse_args(argv)

    regressors = [r.strip().lower() for r in args.regressors.split(",") if r.strip()]
    hidden_sizes = tuple(int(h.strip()) for h in args.hidden_sizes.split(",") if h.strip())

    if args.synthetic:
        return run_synthetic_smoke_test(
            regressors=regressors,
            out_dir=args.out_dir,
            granularity=args.granularity,
            seed=args.seed,
            show=args.show,
        )

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    processed, labeled, task = _resolve_data_roots(args.data_path)

    pid_sets = [set(normalize_pid(x) for x in list_pids(processed, labeled, c)) for c in conditions]
    all_pids = sorted(set.intersection(*pid_sets)) if pid_sets else []
    if args.pids.strip():
        want = {normalize_pid(x.strip()) for x in args.pids.split(",") if x.strip()}
        pids = [pid for pid in all_pids if pid in want]
    else:
        pids = all_pids

    if not pids:
        print(f"No participants found for conditions={conditions} under {processed}")
        return 1

    windows = [float(args.window_s)] if args.window_s is not None else _parse_float_grid(args.windows)
    shifts = [float(args.shift_s)] if args.shift_s is not None else _parse_float_grid(args.shifts)
    is_sweep = (len(windows) * len(shifts)) > 1

    channels, pairs, ch_tag = channel_setup({"drop_suspect_channels": args.drop_suspect_channels})
    print(
        f"Evaluating Single-Frame State -> Window Features -> Existing Decoder | "
        f"conditions={conditions} granularity={args.granularity} "
        f"windows={windows} shifts={shifts} channels={ch_tag} "
        f"embargo={args.embargo_s}s min_holdout_f1={args.min_holdout_f1}"
    )

    loader = DataLoader(
        fnirs_data_source_path=processed,
        task_data_source_path=task,
        labeled_data_source_path=labeled,
        participant_list=[_pid_int(pid) for pid in pids],
        conditions_list=conditions,
    )
    task_df = loader.load_task()
    episode_ids = _episode_table(task_df)
    dec_eps, agent_eps = _split_per_subject(
        episode_ids, args.fraction, args.seed, task_df=task_df
    )

    sweep_rows: list[dict[str, Any]] = []
    artifacts_by_setting: dict[tuple[float, float], list[dict[str, Any]]] = {}

    for win_s in windows:
        for shift_s in shifts:
            print(f"\n== Grid: window_s={win_s:g}s  shift_s={shift_s:g}s ==")
            setting_arts: list[dict[str, Any]] = []
            for pid in pids:
                grp = dec_eps[dec_eps.pid == pid]
                agrp = agent_eps[agent_eps.pid == pid]
                if grp.empty or agrp.empty:
                    if not is_sweep:
                        print(f"  skip pid={pid}: insufficient episode split (dec={len(grp)}, holdout={len(agrp)})")
                    continue

                subj_rows, art, err = evaluate_subject(
                    pid,
                    conditions,
                    processed,
                    labeled,
                    task_df,
                    decoder_keys=_keys(grp),
                    agent_keys=_keys(agrp),
                    regressors=regressors,
                    granularity=args.granularity,
                    window_s=win_s,
                    step_s=args.step_s,
                    embargo_s=args.embargo_s,
                    rate_hz=args.rate_hz,
                    temporal_shift=shift_s,
                    channels=list(channels),
                    pairs=list(pairs),
                    ridge_alpha=args.ridge_alpha,
                    hidden_sizes=hidden_sizes,
                    min_holdout_f1=args.min_holdout_f1,
                    seed=args.seed,
                )
                if err is not None:
                    if not is_sweep:
                        print(f"  drop pid={pid}: {err}")
                    continue

                setting_arts.append(art)
                for row in subj_rows:
                    sweep_rows.append(row)
                    if normalize_granularity(args.granularity) == "continuous":
                        print(
                            f"  pid={row['pid']} [{row['regressor']:5s}] "
                            f"cos={row['feature_cosine_sim']:+.3f} mae={row['feature_mae']:.3f} | "
                            f"Spearman(pred_F)={row['pred_feat_spearman']} vs real_F={row['real_fnirs_spearman']}"
                        )
                    else:
                        print(
                            f"  pid={row['pid']} [{row['regressor']:5s}] "
                            f"cos={row['feature_cosine_sim']:+.3f} mae={row['feature_mae']:.3f} | "
                            f"F1(pred_F)={row['pred_feat_f1']:.3f} vs real_F={row['real_fnirs_f1']:.3f} "
                            f"| agree={row['clf_agreement_acc']:.3f}"
                        )

            if not setting_arts:
                print("  (0 participants passed min_holdout_f1 at this setting)")
            else:
                artifacts_by_setting[(win_s, shift_s)] = setting_arts

    if not sweep_rows:
        print("No subjects passed screening across any (window_s, shift_s) setting.")
        return 1

    sweep_df = pd.DataFrame(sweep_rows)
    cond_tag = "_".join(conditions)
    alpha_tag = f"alpha{args.ridge_alpha:g}".replace(".", "p")
    saved_figs: list[Path] = []

    if is_sweep:
        best_w, best_s = _print_sweep_leaderboard(sweep_df, args.granularity)
        sweep_fig = _plot_window_shift_sweep(
            sweep_df=sweep_df,
            regressors=regressors,
            granularity=args.granularity,
            windows=windows,
            shifts=shifts,
            out_path=args.out_dir / f"single_frame_{cond_tag}_{args.granularity}_{alpha_tag}_sweep.png",
            title_prefix=f"{'+'.join(conditions)} Single-Frame ({args.granularity.capitalize()}, α={args.ridge_alpha:g})",
        )
        saved_figs.append(sweep_fig)
    else:
        best_w, best_s = windows[0], shifts[0]

    best_df = sweep_df[
        (sweep_df["window_s"] == best_w) & (sweep_df["temporal_shift"] == best_s)
    ].copy()
    best_arts = artifacts_by_setting[(best_w, best_s)]
    print(f"\nDetailed breakdown for setting window_s={best_w:g}s, shift_s={best_s:g}s:")
    _print_summary(best_df, args.granularity)

    w_tag = f"w{best_w:g}s_sh{best_s:g}s".replace(".", "p")
    best_figs = generate_all_visualizations(
        df=best_df,
        subject_artifacts=best_arts,
        regressors=regressors,
        granularity=args.granularity,
        out_dir=args.out_dir,
        stem=f"single_frame_{cond_tag}_{args.granularity}_{w_tag}_{alpha_tag}",
        title_prefix=f"{'+'.join(conditions)} ({args.granularity.capitalize()}, win={best_w:g}s, shift={best_s:g}s, α={args.ridge_alpha:g})",
        show=args.show,
    )
    saved_figs.extend(best_figs)
    print("\nSaved visualizations (no CSV written):")
    for fig_path in saved_figs:
        print(f"  {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
