#!/usr/bin/env python3
"""Head-to-head comparison of ``state_window_decoder_single_frame`` vs ``state_window_decoder``.

Evaluates all models on the **exact same** per-participant episode split, purged
embargo mask, and trained Stage-2 ``RobustWindowDecoder`` (``existing_clf``):
  - ``single_frame``   — ``src/models/state_window_decoder_single_frame.py`` ([28] input)
  - ``window_flatten`` — ``src/models/state_window_decoder.py`` with ``window_mode="flatten"`` ([T*28] input)
  - ``window_summary`` — ``src/models/state_window_decoder.py`` with ``window_mode="summary"`` ([6*28] input)

Does NOT write any CSV files. Saves three side-by-side PNG figures under
``results/figures/compare_state_decoders/`` (and opens interactive windows with ``--show``):
  1. ``*_stage1_features.png``        — Feature cosine similarity, per-type MAE, and overall MAE
  2. ``*_stage2_classification.png``  — Per-subject F1 vs Real fNIRS, cohort F1, and decoder agreement
  3. ``*_cohort_diagnostics.png``     — Pooled ROC curves, class-conditional probability separation,
                                        and a full Participant x Metric annotated heatmap

Examples:
  python scripts/compare_state_window_decoders.py \
    --data-path /Users/maddiebrower/workspace/tufts/fNIRS2RL/Experiment/ParticipantData \
    --show

  # Compare only single_frame vs window_summary (or use separate Ridge alpha on windowed inputs):
  python scripts/compare_state_window_decoders.py \
    --data-path /Users/maddiebrower/workspace/tufts/fNIRS2RL/Experiment/ParticipantData \
    --models single_frame,window_summary \
    --ridge-alpha 10 --window-ridge-alpha 200 --show

  # Synthetic Fetch comparison (no ParticipantData required):
  python scripts/compare_state_window_decoders.py --synthetic --show
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
from src.models.state_window_decoder import (  # noqa: E402
    StateToWindowDecoder as WindowStateDecoder,
    build_subject_state_window_dataset as build_window_dataset,
)
from src.models.state_window_decoder_single_frame import (  # noqa: E402
    StateToWindowDecoder as SingleFrameStateDecoder,
    build_subject_state_window_dataset as build_single_frame_dataset,
)
from src.neural.loader import DataLoader  # noqa: E402

DEFAULT_DATA = "/Users/juliasantaniello/Desktop/fNIRS-2-RL/Experiment/ParticipantData"

MODEL_LABELS = {
    "single_frame": "Single-Frame Ridge (28d)",
    "single_frame_mlp": "Single-Frame MLP (28d)",
    "window_flatten": "Window-Flatten Ridge",
    "window_flatten_mlp": "Window-Flatten MLP",
    "window_summary": "Window-Summary Ridge (168d)",
    "window_summary_mlp": "Window-Summary MLP (168d)",
}

PALETTE = {
    "real": "#2b5c8f",
    "single_frame": "#e08214",
    "single_frame_mlp": "#b35806",
    "window_flatten": "#7570b3",
    "window_flatten_mlp": "#984ea3",
    "window_summary": "#2ca25f",
    "window_summary_mlp": "#c44e52",
}


def _pid_int(pid: str):
    return int(pid) if str(pid).isdigit() else pid


def _as_key(k) -> tuple[str, int]:
    return (str(k[0]), int(k[1]))


def _feature_standardized_mae_vector(Z_true: np.ndarray, Z_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(Z_true - Z_pred), axis=0)


def _top_decoder_feature_indices(clf: Any, n_features: int, k: int = 2) -> np.ndarray:
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


def evaluate_subject_comparison(
    pid: str,
    conditions: list[str],
    processed_dir: str,
    labeled_dir: str,
    task_df: pd.DataFrame,
    decoder_keys: set[tuple],
    agent_keys: set[tuple],
    *,
    models_to_run: list[str],
    granularity: str,
    window_s: float,
    step_s: float,
    embargo_s: float,
    rate_hz: float,
    temporal_shift: float,
    channels: list[str],
    pairs: list[tuple[str, str]],
    single_ridge_alpha: float,
    window_ridge_alpha: float,
    hidden_sizes: tuple[int, ...],
    min_holdout_f1: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Fit the shared Stage-2 decoder and compare Single-Frame vs Windowed Stage-1 models."""
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

    bundle_sf, status_sf = build_single_frame_dataset(
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
    if bundle_sf is None:
        return [], {}, status_sf.get("error", "no single-frame bundle")

    bundle_win, status_win = build_window_dataset(
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
    if bundle_win is None:
        return [], {}, status_win.get("error", "no windowed bundle")

    # Guarantee both modules aligned the exact same fNIRS windows and labels.
    assert np.allclose(bundle_sf["F"], bundle_win["F"]), f"Window feature mismatch on pid={pid}"
    assert np.array_equal(bundle_sf["y"], bundle_win["y"]), f"Label mismatch on pid={pid}"

    S_sf = bundle_sf["S"]    # [N, 28]
    S_win = bundle_win["S"]  # [N, window_steps, 28]
    F = bundle_win["F"]
    y = bundle_win["y"]
    T = bundle_win["T"]
    ep_i = bundle_win["ep_i"]
    spans = bundle_win["spans"]
    g = bundle_win["granularity"]
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
        return [], {}, f"too few train windows after embargo ({int(tr.sum())})"
    if te.sum() < 8:
        return [], {}, f"too few holdout windows ({int(te.sum())})"
    if g != "continuous" and len(np.unique(y[te])) < 2:
        return [], {}, "holdout windows have only a single class"

    real_metrics = _score_clf_predictions(existing_clf, F[te], y[te], g)
    if g != "continuous" and min_holdout_f1 > 0.0:
        rf1 = real_metrics["f1"]
        if rf1 is None or rf1 < min_holdout_f1:
            rf1_str = f"{rf1:.3f}" if rf1 is not None else "None"
            return [], {}, f"holdout_f1 {rf1_str} < min_holdout_f1={min_holdout_f1:.2f}"

    group_slices = _feature_group_slices(len(channels), len(pairs))
    f_mu = F[tr].mean(axis=0)
    f_sd = F[tr].std(axis=0)
    f_sd[f_sd < 1e-8] = 1.0
    Z_true = (F[te] - f_mu) / f_sd

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
        "real_f1": real_metrics["f1"],
        "by_model": {},
    }

    for m_key in models_to_run:
        if m_key in ("single_frame", "single_frame_mlp"):
            sf_reg = "mlp" if m_key == "single_frame_mlp" else "ridge"
            dec = SingleFrameStateDecoder(
                regressor_type=sf_reg,
                ridge_alpha=single_ridge_alpha,
                hidden_layer_sizes=hidden_sizes,
                seed=seed,
                granularity=g,
                neural_clf=existing_clf,
            )
            dec.fit(S_sf[tr], F[tr])
            F_hat_te = dec.predict_features(S_sf[te])
            assert np.allclose(dec.predict(S_sf[te]), existing_clf.predict(F_hat_te))
            input_dim = dec.state_dim
        elif m_key in ("window_flatten", "window_flatten_mlp", "window_summary", "window_summary_mlp"):
            w_mode = "flatten" if m_key.startswith("window_flatten") else "summary"
            reg_t = "mlp" if m_key.endswith("_mlp") else "ridge"
            dec = WindowStateDecoder(
                regressor_type=reg_t,
                window_s=window_s,
                rate_hz=rate_hz,
                window_mode=w_mode,
                ridge_alpha=window_ridge_alpha,
                hidden_layer_sizes=hidden_sizes,
                seed=seed,
                granularity=g,
                neural_clf=existing_clf,
            )
            dec.fit(S_win[tr], F[tr])
            F_hat_te = dec.predict_features(S_win[te])
            assert np.allclose(dec.predict(S_win[te]), existing_clf.predict(F_hat_te))
            input_dim = dec.state_dim
        else:
            raise ValueError(f"Unknown model key {m_key!r}")

        feat_metrics = dec.score_feature_predictions(F[te], F_hat_te)
        surr_metrics = _score_clf_predictions(existing_clf, F_hat_te, y[te], g)
        fidelity = _score_decoder_fidelity(existing_clf, F[te], F_hat_te, g)

        Z_hat = (F_hat_te - f_mu) / f_sd
        col_mae = _feature_standardized_mae_vector(Z_true, Z_hat)
        group_maes = {
            f"mae_{g_name}": float(np.nanmean(col_mae[sl])) for g_name, sl in group_slices
        }

        artifacts["by_model"][m_key] = {
            "decoder": dec,
            "F_hat_te": F_hat_te,
            "Z_true": Z_true,
            "Z_hat": Z_hat,
            "col_mae": col_mae,
            "cos": feat_metrics["feature_cosine_sim"],
            "pred_trace": _continuous_signal(existing_clf, F_hat_te, g),
        }

        rows.append(
            dict(
                pid=normalize_pid(pid),
                conditions="+".join(conditions),
                granularity=g,
                window_s=float(window_s),
                temporal_shift=float(temporal_shift),
                model=m_key,
                input_dim=int(input_dim),
                feature_dim=int(F.shape[1]),
                n_train=int(tr.sum()),
                n_holdout=int(te.sum()),
                feature_cosine_sim=feat_metrics["feature_cosine_sim"],
                feature_mae=feat_metrics["feature_mae"],
                feature_mse=feat_metrics["feature_mse"],
                feature_r2=feat_metrics["feature_r2"],
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
        )

    return rows, artifacts, None


def _label_for(
    df: pd.DataFrame,
    m_key: str,
    best_params: dict[str, tuple[float, float]] | None = None,
) -> str:
    if m_key == "real":
        base = "Real fNIRS"
    else:
        sub = df[df["model"] == m_key]
        if not sub.empty and "input_dim" in sub.columns:
            d = int(sub["input_dim"].iloc[0])
            nice = {
                "single_frame": f"Single-Frame Ridge ({d}d)",
                "single_frame_mlp": f"Single-Frame MLP ({d}d)",
                "window_flatten": f"Window-Flatten Ridge ({d}d)",
                "window_flatten_mlp": f"Window-Flatten MLP ({d}d)",
                "window_summary": f"Window-Summary Ridge ({d}d)",
                "window_summary_mlp": f"Window-Summary MLP ({d}d)",
            }
            base = nice.get(m_key, m_key)
        else:
            base = MODEL_LABELS.get(m_key, m_key)

    if best_params and m_key in best_params:
        w_v, s_v = best_params[m_key]
        return f"{base} [{w_v:g}s, sh={s_v:g}s]"
    return base


def _plot_stage1_comparison(
    state_df: pd.DataFrame,
    models_to_run: list[str],
    best_params: dict[str, tuple[float, float]],
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 1: Side-by-side Stage-1 feature prediction comparison (per-model best win/shift)."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

    pids = sorted(state_df["pid"].unique())
    x = np.arange(len(pids))
    width = min(0.82 / max(len(models_to_run), 1), 0.36)

    # Panel A: Per-subject Cosine Similarity
    ax = axes[0]
    for idx, m_key in enumerate(models_to_run):
        sub = state_df[state_df["model"] == m_key].set_index("pid").reindex(pids)
        offset = (idx - (len(models_to_run) - 1) / 2) * width
        ax.bar(
            x + offset,
            sub["feature_cosine_sim"],
            width=width,
            color=PALETTE.get(m_key, f"C{idx}"),
            alpha=0.88,
            edgecolor="k",
            linewidth=0.5,
            label=_label_for(state_df, m_key, best_params),
        )
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pids, rotation=35 if len(pids) > 6 else 0, fontsize=8.5)
    ax.set_ylabel("Feature Cosine Similarity (higher = better)")
    ax.set_xlabel("Participant")
    ax.set_title("A. Stage-1 Feature Cosine Similarity by Subject")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="best")

    # Panel B: Standardized MAE by Window Feature Type (lower = better)
    ax = axes[1]
    group_cols = [c for c in state_df.columns if c.startswith("mae_")]
    group_labels = [c.replace("mae_", "") for c in group_cols]
    gx = np.arange(len(group_cols))
    g_width = min(0.8 / max(len(models_to_run), 1), 0.32)
    for idx, m_key in enumerate(models_to_run):
        sub = state_df[state_df["model"] == m_key]
        means = [sub[c].mean() for c in group_cols]
        sems = [
            sub[c].std(ddof=1) / np.sqrt(max(sub[c].notna().sum(), 1))
            if sub[c].notna().sum() > 1
            else 0.0
            for c in group_cols
        ]
        offset = (idx - (len(models_to_run) - 1) / 2) * g_width
        ax.bar(
            gx + offset,
            means,
            yerr=sems,
            width=g_width,
            color=PALETTE.get(m_key, f"C{idx}"),
            alpha=0.88,
            capsize=3,
            edgecolor="k",
            linewidth=0.5,
            label=_label_for(state_df, m_key, best_params),
        )
    ax.set_xticks(gx)
    ax.set_xticklabels(group_labels, rotation=20, ha="right")
    ax.set_ylabel("Standardized MAE ± SEM (lower = better)")
    ax.set_title("B. Error by Window Feature Type")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, frameon=False, loc="best")

    # Panel C: Cohort distribution of overall Feature MAE across subjects
    ax = axes[2]
    rng = np.random.default_rng(0)
    positions = np.arange(len(models_to_run))
    for idx, m_key in enumerate(models_to_run):
        vals = state_df[state_df["model"] == m_key]["feature_mae"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        color = PALETTE.get(m_key, f"C{idx}")
        bp = ax.boxplot(
            [vals],
            positions=[idx],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("k")
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            c=color,
            s=36,
            edgecolors="k",
            linewidths=0.45,
            zorder=3,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([_label_for(state_df, m, best_params) for m in models_to_run], rotation=14, ha="right", fontsize=8.0)
    ax.set_ylabel("Holdout Feature MAE (lower = better)")
    ax.set_title("C. Overall Stage-1 Feature Error")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} — Stage-1 Feature Prediction (Per-Model Best Window & Shift)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_stage2_comparison(
    real_df: pd.DataFrame,
    state_df: pd.DataFrame,
    models_to_run: list[str],
    best_params: dict[str, tuple[float, float]],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 2: Stage-2 classification comparing Real fNIRS & State models at each model's best (win, shift)."""
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

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))

    # Panel A: Per-participant bars across union of screened subjects
    ax = axes[0]
    pids = sorted(set(real_df["pid"].unique()).union(state_df["pid"].unique()))
    x = np.arange(len(pids))
    n_bars = 1 + len(models_to_run)
    width = min(0.84 / n_bars, 0.28)

    real_by_pid = real_df.set_index("pid")[real_col].reindex(pids)
    ax.bar(
        x - ((n_bars - 1) / 2) * width,
        real_by_pid,
        width=width,
        color=PALETTE["real"],
        alpha=0.92,
        edgecolor="k",
        linewidth=0.5,
        label=_label_for(state_df, "real", best_params),
    )
    for idx, m_key in enumerate(models_to_run, start=1):
        sub = state_df[state_df["model"] == m_key].set_index("pid").reindex(pids)
        offset = (idx - (n_bars - 1) / 2) * width
        ax.bar(
            x + offset,
            sub[pred_col],
            width=width,
            color=PALETTE.get(m_key, f"C{idx}"),
            alpha=0.88,
            edgecolor="k",
            linewidth=0.5,
            label=_label_for(state_df, m_key, best_params),
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
    ax.legend(fontsize=7.0, frameon=False, loc="best")

    # Panel B: Screened Cohort Summary — Mean ± SEM at each model's own optimal (win, shift)
    ax = axes[1]
    series_keys = ["real"] + list(models_to_run)
    sx = np.arange(len(series_keys))
    xtick_lbls = []

    for s_i, s_key in enumerate(series_keys):
        if s_key == "real":
            vals = real_df[real_col].dropna().to_numpy(dtype=float)
        else:
            vals = state_df[state_df["model"] == s_key][pred_col].dropna().to_numpy(dtype=float)

        c = PALETTE.get(s_key, "#555555")
        mu = float(np.mean(vals)) if len(vals) else np.nan
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.bar(
            sx[s_i],
            mu,
            yerr=sem,
            width=0.52,
            color=c,
            alpha=0.88,
            capsize=3,
            edgecolor="k",
            linewidth=0.5,
        )
        xtick_lbls.append(f"{_label_for(state_df, s_key, best_params)}\n(n={len(vals)})")

    ax.axhline(chance, color="0.45", ls="--", lw=1)
    ax.set_xticks(sx)
    ax.set_xticklabels(xtick_lbls, rotation=14, ha="right", fontsize=8.0)
    ax.set_ylabel(f"Mean {metric_label} ± SEM")
    if g != "continuous":
        ax.set_ylim(0.0, 1.0)
    ax.set_title(f"B. Screened Cohort {metric_label} (Per-Model Best Params)")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Fidelity to Existing Decoder at each state model's own optimal (win, shift)
    ax = axes[2]
    fidelity_metrics = (
        [("clf_score_pearson", "Output Pearson r")]
        if g == "continuous"
        else [("clf_agreement_acc", "Class Agreement"), ("clf_score_pearson", "Proba Pearson r")]
    )
    fx = np.arange(len(fidelity_metrics))
    rng = np.random.default_rng(1)
    box_step = min(0.76 / max(len(models_to_run), 1), 0.30)
    box_w = box_step * 0.78

    for idx, m_key in enumerate(models_to_run):
        sub = state_df[state_df["model"] == m_key]
        offset = (idx - (len(models_to_run) - 1) / 2) * box_step
        color = PALETTE.get(m_key, f"C{idx}")
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
            for med in bp["medians"]:
                med.set_color("k")
            jitter = rng.uniform(-box_w * 0.22, box_w * 0.22, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                c=color,
                s=30,
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
                label=_label_for(state_df, m_key, best_params) if m_i == 0 else None,
            )
    ax.set_xticks(fx)
    ax.set_xticklabels([lbl for _, lbl in fidelity_metrics])
    ax.set_ylabel("Fidelity to Existing Decoder(Real F)")
    ax.set_ylim(-0.25 if g == "continuous" else -0.1, 1.05)
    ax.set_title("C. Fidelity to Real Neural Decoder")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.2, frameon=False, loc="best")

    fig.suptitle(
        f"{title_prefix} — Stage-2 Classification (Each Model at Its Optimal Window & Shift)",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_pooled_roc(
    state_df: pd.DataFrame,
    best_arts_by_key: dict[str, list[dict[str, Any]]],
    models_to_run: list[str],
    best_params: dict[str, tuple[float, float]],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path | None:
    """Standalone 1:1 square Pooled Holdout ROC figure using each model's own optimal (win, shift)."""
    g = normalize_granularity(granularity)
    if g != "binary":
        return None

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    plotted = False

    series_keys = ["real"] + list(models_to_run)
    for s_key in series_keys:
        arts = best_arts_by_key.get(s_key, [])
        if not arts:
            continue
        y_vec = np.concatenate([a["y_te"].astype(int) for a in arts])
        if len(np.unique(y_vec)) < 2:
            continue
        p_vec = (
            np.concatenate([a["real_trace"] for a in arts])
            if s_key == "real"
            else np.concatenate([a["by_model"][s_key]["pred_trace"] for a in arts])
        )
        fpr, tpr, _ = roc_curve(y_vec, p_vec)
        auc_val = float(roc_auc_score(y_vec, p_vec))
        lbl = _label_for(state_df, s_key, best_params)
        color = PALETTE.get(s_key, "#555555")
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.2,
            label=f"{lbl} (n={len(arts)}, AUC={auc_val:.3f})",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.65, label="Chance (AUC=0.500)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False Positive Rate (Pooled Holdout Windows)", fontsize=10.5)
    ax.set_ylabel("True Positive Rate", fontsize=10.5)
    ax.set_title(
        f"{title_prefix}\nPooled Holdout ROC (Per-Model Optimal Window & Shift)",
        fontsize=11,
        pad=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.0, frameon=True, facecolor="white", framealpha=0.92, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _plot_cohort_diagnostics(
    real_df: pd.DataFrame,
    state_df: pd.DataFrame,
    best_arts_by_key: dict[str, list[dict[str, Any]]],
    models_to_run: list[str],
    best_params: dict[str, tuple[float, float]],
    granularity: str,
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Figure 3: Per-model optimal probability separation and screened participants heatmap."""
    g = normalize_granularity(granularity)
    all_pids_set = set(real_df["pid"].unique()).union(state_df["pid"].unique())
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(13.0, 9.5 + 1.15 * len(models_to_run)), max(4.8, 0.42 * len(all_pids_set) + 1.8)),
        gridspec_kw={"width_ratios": [1.0, max(1.3, 0.48 * (1 + 2 * len(models_to_run)))]},
    )

    # Panel A: Class-Conditional Probability Separation (each model at its own optimal win/shift)
    ax = axes[0]
    series_list = [("real", _label_for(state_df, "real", best_params), PALETTE["real"])] + [
        (m, _label_for(state_df, m, best_params), PALETTE.get(m, f"C{i}"))
        for i, m in enumerate(models_to_run)
    ]
    if g != "continuous":
        all_y_vals = [
            a["y_te"].astype(int)
            for arts in best_arts_by_key.values()
            for a in arts
        ]
        classes = sorted(np.unique(np.concatenate(all_y_vals))) if all_y_vals else [0, 1]
        cx = np.arange(len(classes))
        step_w = min(0.78 / len(series_list), 0.24)
        for s_idx, (s_key, s_lbl, color) in enumerate(series_list):
            arts = best_arts_by_key.get(s_key, [])
            if not arts:
                continue
            y_k = np.concatenate([a["y_te"].astype(int) for a in arts])
            p_k = (
                np.concatenate([a["real_trace"] for a in arts])
                if s_key == "real"
                else np.concatenate([a["by_model"][s_key]["pred_trace"] for a in arts])
            )
            offset = (s_idx - (len(series_list) - 1) / 2) * step_w
            for c_i, cls_val in enumerate(classes):
                vals = p_k[y_k == cls_val]
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
                    s=28,
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
        ax.set_title("A. Pooled Probability Separation (Per-Model Best Params)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7.0, frameon=False, loc="best")
    else:
        for s_idx, (s_key, s_lbl, color) in enumerate(series_list):
            arts = best_arts_by_key.get(s_key, [])
            if not arts:
                continue
            y_k = np.concatenate([a["y_te"].astype(float) for a in arts])
            p_k = (
                np.concatenate([a["real_trace"] for a in arts])
                if s_key == "real"
                else np.concatenate([a["by_model"][s_key]["pred_trace"] for a in arts])
            )
            err = np.abs(y_k - p_k)
            bp = ax.boxplot([err], positions=[s_idx], widths=0.45, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
        ax.set_xticks(range(len(series_list)))
        ax.set_xticklabels([s[1] for s in series_list], rotation=18, ha="right", fontsize=8.0)
        ax.set_ylabel("|y_true - y_pred|")
        ax.set_title("A. Pooled Absolute Error Distribution")
        ax.grid(axis="y", alpha=0.3)

    # Panel B: Screened Participants Multi-Metric Heatmap (union of kept pids across models)
    ax = axes[1]
    val_col = "pred_feat_spearman" if g == "continuous" else "pred_feat_f1"
    real_col = "real_fnirs_spearman" if g == "continuous" else "real_fnirs_f1"
    agree_col = "clf_score_pearson" if g == "continuous" else "clf_agreement_acc"

    real_series = real_df.set_index("pid")[real_col]
    # Include any pid kept in a state model even if absent at Real's optimal (win, shift)
    combined_index = list(real_series.sort_values(ascending=False).index)
    for p_extra in sorted(all_pids_set):
        if p_extra not in combined_index:
            combined_index.append(p_extra)
    sorted_pids = combined_index

    short_names = {
        "single_frame": "SF-Ridge",
        "single_frame_mlp": "SF-MLP",
        "window_flatten": "Flat-Ridge",
        "window_flatten_mlp": "Flat-MLP",
        "window_summary": "Sum-Ridge",
        "window_summary_mlp": "Sum-MLP",
    }
    w_r, s_r = best_params["real"]
    r_hdr = f"Real F1\n({w_r:g}s,{s_r:g}s)" if g != "continuous" else f"Real ρ\n({w_r:g}s,{s_r:g}s)"
    col_defs: list[tuple[str, pd.Series]] = [(r_hdr, real_series.reindex(sorted_pids))]
    for m_key in models_to_run:
        sub = state_df[state_df["model"] == m_key].set_index("pid").reindex(sorted_pids)
        tag = short_names.get(m_key, m_key)
        wm, sm = best_params[m_key]
        m_lbl = f"{tag} F1\n({wm:g}s,{sm:g}s)" if g != "continuous" else f"{tag} ρ\n({wm:g}s,{sm:g}s)"
        col_defs.append((m_lbl, sub[val_col]))
    for m_key in models_to_run:
        sub = state_df[state_df["model"] == m_key].set_index("pid").reindex(sorted_pids)
        tag = short_names.get(m_key, m_key)
        wm, sm = best_params[m_key]
        a_lbl = f"{tag} Agree\n({wm:g}s,{sm:g}s)" if g != "continuous" else f"{tag} r\n({wm:g}s,{sm:g}s)"
        col_defs.append((a_lbl, sub[agree_col]))

    mat = np.column_stack([s.to_numpy(dtype=float) for _, s in col_defs])
    vmin, vmax = (-0.4, 0.8) if g == "continuous" else (0.25, 0.85)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#efefef")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for r_i in range(mat.shape[0]):
        for c_j in range(mat.shape[1]):
            v = mat[r_i, c_j]
            txt = f"{v:.2f}" if np.isfinite(v) else "—"
            ax.text(
                c_j,
                r_i,
                txt,
                ha="center",
                va="center",
                fontsize=8.0,
                color="k" if np.isfinite(v) else "0.5",
                fontweight="bold" if (g != "continuous" and c_j > 0 and np.isfinite(v) and v >= 0.60) else "normal",
            )

    ax.set_xticks(np.arange(len(col_defs)))
    ax.set_xticklabels([c[0] for c in col_defs], rotation=20, ha="right", fontsize=8.0)
    ax.set_yticks(np.arange(len(sorted_pids)))
    ax.set_yticklabels([str(p) for p in sorted_pids], fontsize=8.5)
    ax.set_title("B. Screened Participants Heatmap (— = below F1 gate at setting)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title_prefix} — Screened Cohort Diagnostics (Independent Model Hyperparameters)",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def generate_comparison_figures(
    real_df: pd.DataFrame,
    state_df: pd.DataFrame,
    best_arts_by_key: dict[str, list[dict[str, Any]]],
    models_to_run: list[str],
    best_params: dict[str, tuple[float, float]],
    granularity: str,
    out_dir: Path,
    stem: str,
    title_prefix: str,
    show: bool = False,
) -> list[Path]:
    paths = [
        _plot_stage1_comparison(
            state_df,
            models_to_run,
            best_params,
            out_dir / f"{stem}_stage1_features.png",
            title_prefix,
        ),
        _plot_stage2_comparison(
            real_df,
            state_df,
            models_to_run,
            best_params,
            granularity,
            out_dir / f"{stem}_stage2_classification.png",
            title_prefix,
        ),
        _plot_cohort_diagnostics(
            real_df,
            state_df,
            best_arts_by_key,
            models_to_run,
            best_params,
            granularity,
            out_dir / f"{stem}_cohort_diagnostics.png",
            title_prefix,
        ),
    ]
    roc_fig = _plot_pooled_roc(
        state_df,
        best_arts_by_key,
        models_to_run,
        best_params,
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


def _parse_float_grid(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty float grid: {raw!r}")
    return vals


def _plot_window_shift_sweep(
    sweep_df: pd.DataFrame,
    models_to_run: list[str],
    granularity: str,
    windows: list[float],
    shifts: list[float],
    out_path: Path,
    title_prefix: str,
) -> Path:
    """Plot 2D (window_s x temporal_shift) heatmaps for Real fNIRS and each state decoder."""
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
        (m, MODEL_LABELS.get(m, m)) for m in models_to_run
    ]
    n_cols = len(col_specs)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.6 * n_cols, max(5.6, 1.4 * len(windows) + 3.0)),
        squeeze=False,
    )

    for c_idx, (key, col_title) in enumerate(col_specs):
        if key == "real":
            # One row per (window_s, temporal_shift, pid)
            base = sweep_df.drop_duplicates(subset=["window_s", "temporal_shift", "pid"])
            c1, c2 = m1_real, m2_real
        else:
            base = sweep_df[sweep_df["model"] == key]
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

        # Find best cell (preferring n >= 2 if any exist)
        valid_mask = np.isfinite(mat1) & (cnts >= 2)
        if not valid_mask.any():
            valid_mask = np.isfinite(mat1)
        best_rc = None
        if valid_mask.any():
            masked = np.where(valid_mask, mat1, -np.inf)
            best_rc = np.unravel_index(int(np.argmax(masked)), mat1.shape)

        # Row 0: Primary metric (Macro-F1 / Spearman) + n_kept
        ax0 = axes[0, c_idx]
        im0 = ax0.imshow(mat1, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        for wi in range(len(windows)):
            for si in range(len(shifts)):
                v = mat1[wi, si]
                n_k = cnts[wi, si]
                star = " ★" if best_rc == (wi, si) else ""
                txt = f"{v:.2f}{star}\n(n={n_k})" if np.isfinite(v) else "—"
                ax0.text(
                    si,
                    wi,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold" if best_rc == (wi, si) else "normal",
                    color="k",
                )
        ax0.set_xticks(range(len(shifts)))
        ax0.set_xticklabels([f"{s:g}s" for s in shifts])
        ax0.set_yticks(range(len(windows)))
        ax0.set_yticklabels([f"{w:g}s" for w in windows])
        ax0.set_xlabel("Temporal Shift")
        if c_idx == 0:
            ax0.set_ylabel(f"Window Size\n({m1_lbl})")
        ax0.set_title(f"{col_title}\n{m1_lbl}")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Row 1: Secondary metric (Holdout AUC + Agreement)
        ax1 = axes[1, c_idx]
        im1 = ax1.imshow(mat2, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=min(vmax + 0.08, 0.95))
        for wi in range(len(windows)):
            for si in range(len(shifts)):
                v = mat2[wi, si]
                ag = ag_mat[wi, si]
                if not np.isfinite(v):
                    txt = "—"
                elif key != "real" and np.isfinite(ag):
                    txt = f"{v:.2f}\nag={ag:.2f}"
                else:
                    txt = f"{v:.2f}"
                ax1.text(si, wi, txt, ha="center", va="center", fontsize=8.0, color="k")
        ax1.set_xticks(range(len(shifts)))
        ax1.set_xticklabels([f"{s:g}s" for s in shifts])
        ax1.set_yticks(range(len(windows)))
        ax1.set_yticklabels([f"{w:g}s" for w in windows])
        ax1.set_xlabel("Temporal Shift")
        if c_idx == 0:
            ax1.set_ylabel(f"Window Size\n({m2_lbl})")
        sub_lbl = f"{m2_lbl} (+ Agree)" if (key != "real" and g != "continuous") else m2_lbl
        ax1.set_title(sub_lbl)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{title_prefix} — Hyperparameter Sweep: Window Size × Temporal Shift (★ = Best {m1_lbl})",
        fontsize=12.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def _select_best_setting_from_agg(agg_df: pd.DataFrame, primary: str, secondary: str) -> tuple[float, float, pd.Series]:
    reset = agg_df.reset_index()
    multi = reset[reset["n_kept"] >= 2]
    pool = multi if not multi.empty else reset
    best = pool.sort_values([primary, secondary], ascending=False).iloc[0]
    return float(best["window_s"]), float(best["temporal_shift"]), best


def _print_sweep_leaderboard(
    sweep_df: pd.DataFrame,
    models_to_run: list[str],
    granularity: str,
) -> dict[str, tuple[float, float]]:
    """Rank (window_s, temporal_shift) independently for Real fNIRS and each state model."""
    g = normalize_granularity(granularity)
    primary_col = "pred_feat_spearman" if g == "continuous" else "pred_feat_f1"
    sec_col = "clf_score_pearson" if g == "continuous" else "pred_feat_auc"
    real_col = "real_fnirs_spearman" if g == "continuous" else "real_fnirs_f1"
    real_sec_col = "real_fnirs_r2" if g == "continuous" else "real_fnirs_auc"

    print("\n" + "=" * 92)
    print("HYPERPARAMETER SWEEP LEADERBOARD: INDEPENDENT (WINDOW_S, SHIFT_S) PER MODEL")
    print("=" * 92)

    best_params: dict[str, tuple[float, float]] = {}

    # 1. Real fNIRS Stage-2 decoder leaderboard
    real_dedup = sweep_df.drop_duplicates(subset=["window_s", "temporal_shift", "pid"])
    real_agg = (
        real_dedup.groupby(["window_s", "temporal_shift"])
        .agg(
            n_kept=("pid", "nunique"),
            real_metric=(real_col, "mean"),
            real_sec=(real_sec_col, "mean"),
        )
        .sort_values(["real_metric", "real_sec"], ascending=False)
    )
    print(f"\n--- Model: real_fnirs (ranked by {real_col}) ---")
    print(real_agg.round(4).to_string())
    w_r, s_r, _ = _select_best_setting_from_agg(real_agg, "real_metric", "real_sec")
    best_params["real"] = (w_r, s_r)

    # 2. Independent leaderboard for each state model
    for model_name in models_to_run:
        m_df = sweep_df[sweep_df["model"] == model_name]
        agg = (
            m_df.groupby(["window_s", "temporal_shift"])
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
        print(f"\n--- Model: {model_name} (ranked by {primary_col}) ---")
        print(agg.round(4).to_string())
        w_m, s_m, _ = _select_best_setting_from_agg(agg, "pred_metric", "sec_metric")
        best_params[model_name] = (w_m, s_m)

    n_total_hp = 2 * len(best_params)
    print(f"\n>>> Selected Independent Best Hyperparameters ({n_total_hp} total):")
    for k in ["real"] + list(models_to_run):
        wk, sk = best_params[k]
        print(f"    {k:16s}: window_s={wk:g}s, shift_s={sk:g}s")
    return best_params


def _print_comparison_summary(
    real_df: pd.DataFrame,
    state_df: pd.DataFrame,
    best_params: dict[str, tuple[float, float]],
    granularity: str,
) -> None:
    g = normalize_granularity(granularity)
    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD COMPARISON AT EACH MODEL'S OPTIMAL (WINDOW_S, SHIFT_S)")
    print("=" * 92)

    stage1_cols = ["window_s", "temporal_shift", "input_dim", "feature_cosine_sim", "feature_mae", "feature_mse", "feature_r2"]
    print("\n1) Stage-1 Window Feature Prediction (at each state model's best params):")
    print(state_df.groupby("model", sort=False)[stage1_cols].mean().round(4).to_string())

    rows = []
    w_r, s_r = best_params["real"]
    if g == "continuous":
        rows.append(
            dict(
                model="real_fnirs",
                window_s=w_r,
                shift_s=s_r,
                n_kept=int(real_df["pid"].nunique()),
                spearman=float(real_df["real_fnirs_spearman"].mean()),
                r2=float(real_df["real_fnirs_r2"].mean()),
                score_pearson=1.0,
            )
        )
        for m_name, sub in state_df.groupby("model", sort=False):
            wm, sm = best_params[m_name]
            rows.append(
                dict(
                    model=m_name,
                    window_s=wm,
                    shift_s=sm,
                    n_kept=int(sub["pid"].nunique()),
                    spearman=float(sub["pred_feat_spearman"].mean()),
                    r2=float(sub["pred_feat_r2"].mean()),
                    score_pearson=float(sub["clf_score_pearson"].mean()),
                )
            )
    else:
        rows.append(
            dict(
                model="real_fnirs",
                window_s=w_r,
                shift_s=s_r,
                n_kept=int(real_df["pid"].nunique()),
                macro_f1=float(real_df["real_fnirs_f1"].mean()),
                auc=float(real_df["real_fnirs_auc"].mean()),
                clf_agreement=1.0,
                proba_pearson=1.0,
            )
        )
        for m_name, sub in state_df.groupby("model", sort=False):
            wm, sm = best_params[m_name]
            rows.append(
                dict(
                    model=m_name,
                    window_s=wm,
                    shift_s=sm,
                    n_kept=int(sub["pid"].nunique()),
                    macro_f1=float(sub["pred_feat_f1"].mean()),
                    auc=float(sub["pred_feat_auc"].mean()),
                    clf_agreement=float(sub["clf_agreement_acc"].mean()),
                    proba_pearson=float(sub["clf_score_pearson"].mean()),
                )
            )

    print("\n2) Stage-2 Classification Performance (each model at its own optimal window_s & shift_s):")
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())


def run_synthetic_comparison(
    models_to_run: list[str],
    out_dir: Path,
    granularity: str = "binary",
    single_ridge_alpha: float = 10.0,
    window_ridge_alpha: float = 10.0,
    seed: int = 42,
    show: bool = False,
) -> int:
    """Synthetic Fetch comparison when ParticipantData is unavailable."""
    print("=" * 84)
    print("SYNTHETIC COMPARISON: Single-Frame vs Window-Flatten vs Window-Summary")
    print("=" * 84)

    channels, pairs, _ = channel_setup({"drop_suspect_channels": False})
    n_channels = len(channels)
    group_slices = _feature_group_slices(len(channels), len(pairs))
    window_s, rate_hz = 6.0, 5.2
    win_steps = int(round(window_s * rate_hz))
    obs_dim, goal_dim = 25, 3
    g = normalize_granularity(granularity)

    all_rows: list[dict[str, Any]] = []
    subject_artifacts: list[dict[str, Any]] = []
    synth_pids = ["005", "010", "013", "015", "018", "019"]

    for s_idx, pid in enumerate(synth_pids):
        sub_rng = np.random.default_rng(seed + s_idx * 17)
        n_samples = 280
        split = int(0.65 * n_samples)
        noise_scale = 0.28 + 0.07 * s_idx

        t_norm = np.linspace(0.0, 1.0, win_steps, dtype=np.float32)[None, :, None]
        start_obs = sub_rng.normal(size=(n_samples, 1, obs_dim)).astype(np.float32)
        goals = sub_rng.normal(size=(n_samples, 1, goal_dim)).astype(np.float32)
        progress = sub_rng.uniform(-0.4, 1.0, size=(n_samples, 1, 1)).astype(np.float32)

        obs_windows = np.broadcast_to(start_obs, (n_samples, win_steps, obs_dim)).copy()
        obs_windows[:, :, :3] = (
            start_obs[:, :, :3]
            + progress * t_norm * (goals - start_obs[:, :, :3])
            + sub_rng.normal(scale=0.05, size=(n_samples, win_steps, 3)).astype(np.float32)
        )
        obs_windows[:, :, 3:] += sub_rng.normal(
            scale=0.08, size=(n_samples, win_steps, obs_dim - 3)
        ).astype(np.float32)
        goal_windows = np.broadcast_to(goals, (n_samples, win_steps, goal_dim)).copy()

        S_win = np.concatenate([obs_windows, goal_windows], axis=2)  # [N, 31, 28]
        S_sf = S_win[:, -1, :]                                       # [N, 28] end frame

        d_start = np.linalg.norm(obs_windows[:, 0, :3] - goal_windows[:, 0, :], axis=1)
        d_end = np.linalg.norm(obs_windows[:, -1, :3] - goal_windows[:, -1, :], axis=1)
        goal_progress = d_start - d_end
        latent_opt = (goal_progress - np.median(goal_progress)) / (np.std(goal_progress) + 1e-8)

        W_state = sub_rng.normal(scale=0.30, size=(obs_dim + goal_dim, n_channels))
        t_grid = np.arange(win_steps, dtype=float) / rate_hz
        raw_windows = []
        for i in range(n_samples):
            ch_traj = S_win[i] @ W_state
            slope = 0.25 * latent_opt[i] * np.array([1, 1, 0.5, 0.5, -1, -1, -0.5, -0.5])[:n_channels]
            win = (
                ch_traj
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
        Z_true = (F[split:] - f_mu) / f_sd

        art: dict[str, Any] = {
            "pid": pid,
            "F_te": F[split:],
            "y_te": y[split:],
            "T_te": np.arange(n_samples - split, dtype=float),
            "channels": list(channels),
            "pairs": list(pairs),
            "existing_clf": existing_clf,
            "real_trace": _continuous_signal(existing_clf, F[split:], g),
            "real_f1": real_metrics["f1"],
            "by_model": {},
        }

        for m_key in models_to_run:
            if m_key in ("single_frame", "single_frame_mlp"):
                sf_reg = "mlp" if m_key == "single_frame_mlp" else "ridge"
                dec = SingleFrameStateDecoder(
                    regressor_type=sf_reg,
                    ridge_alpha=single_ridge_alpha,
                    seed=seed,
                    granularity=g,
                    neural_clf=existing_clf,
                )
                dec.fit(S_sf[:split], F[:split])
                F_hat_te = dec.predict_features(S_sf[split:])
                input_dim = dec.state_dim
            else:
                w_mode = "flatten" if m_key.startswith("window_flatten") else "summary"
                reg_t = "mlp" if m_key.endswith("_mlp") else "ridge"
                dec = WindowStateDecoder(
                    regressor_type=reg_t,
                    window_s=window_s,
                    rate_hz=rate_hz,
                    window_mode=w_mode,
                    ridge_alpha=window_ridge_alpha,
                    seed=seed,
                    granularity=g,
                    neural_clf=existing_clf,
                )
                dec.fit(S_win[:split], F[:split])
                F_hat_te = dec.predict_features(S_win[split:])
                input_dim = dec.state_dim

            feat_m = dec.score_feature_predictions(F[split:], F_hat_te)
            surr_m = _score_clf_predictions(existing_clf, F_hat_te, y[split:], g)
            fid_m = _score_decoder_fidelity(existing_clf, F[split:], F_hat_te, g)
            Z_hat = (F_hat_te - f_mu) / f_sd
            col_mae = _feature_standardized_mae_vector(Z_true, Z_hat)
            group_maes = {
                f"mae_{g_name}": float(np.nanmean(col_mae[sl])) for g_name, sl in group_slices
            }

            art["by_model"][m_key] = {
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
                    window_s=window_s,
                    temporal_shift=0.0,
                    model=m_key,
                    input_dim=input_dim,
                    feature_dim=F.shape[1],
                    n_train=split,
                    n_holdout=n_samples - split,
                    feature_cosine_sim=feat_m["feature_cosine_sim"],
                    feature_mae=feat_m["feature_mae"],
                    feature_mse=feat_m["feature_mse"],
                    feature_r2=feat_m["feature_r2"],
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

    df = pd.DataFrame(all_rows)
    synth_params = {"real": (window_s, 0.0), **{m: (window_s, 0.0) for m in models_to_run}}
    real_synth_df = df.drop_duplicates(subset=["pid"]).copy()
    arts_map = {"real": subject_artifacts, **{m: subject_artifacts for m in models_to_run}}
    _print_comparison_summary(real_synth_df, df, synth_params, g)

    saved = generate_comparison_figures(
        real_df=real_synth_df,
        state_df=df,
        best_arts_by_key=arts_map,
        models_to_run=models_to_run,
        best_params=synth_params,
        granularity=g,
        out_dir=out_dir,
        stem=f"compare_synthetic_fetch_{g}",
        title_prefix=f"Synthetic Fetch ({g.capitalize()})",
        show=show,
    )
    print("\nSaved comparison visualizations (no CSV written):")
    for p in saved:
        print(f"  {p}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=os.environ.get("DATA_PATH", DEFAULT_DATA))
    p.add_argument("--conditions", default="RW", help="Comma-separated conditions, e.g. RW or RW,RP")
    p.add_argument("--pids", default="", help="Comma-separated participant IDs (default: all available)")
    p.add_argument("--granularity", default="binary", choices=["binary", "ternary", "discrete", "continuous"])
    p.add_argument(
        "--models",
        default="single_frame,single_frame_mlp,window_flatten,window_flatten_mlp,window_summary,window_summary_mlp",
        help=(
            "Comma-separated models to compare "
            "(single_frame,single_frame_mlp,window_flatten,window_flatten_mlp,window_summary,window_summary_mlp)"
        ),
    )
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
    p.add_argument("--drop-suspect-channels", action="store_true")
    p.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge alpha for single_frame (and default for windowed)")
    p.add_argument(
        "--window-ridge-alpha",
        type=float,
        default=None,
        help="Optional separate Ridge alpha for window_flatten / window_summary (defaults to --ridge-alpha)",
    )
    p.add_argument(
        "--hidden-sizes",
        default="128,64",
        help="Comma-separated MLP hidden layer sizes for window_summary_mlp (default: 128,64)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "compare_state_decoders",
        help="Directory where comparison PNG figures are saved",
    )
    p.add_argument("--show", action="store_true", help="Also open interactive matplotlib windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true", help="Run synthetic Fetch comparison")
    args = p.parse_args(argv)

    valid_models = {
        "single_frame",
        "single_frame_mlp",
        "window_flatten",
        "window_flatten_mlp",
        "window_summary",
        "window_summary_mlp",
    }
    models_to_run = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models_to_run if m not in valid_models]
    if unknown or not models_to_run:
        p.error(f"Invalid --models {unknown}. Choose from: {sorted(valid_models)}")

    single_alpha = float(args.ridge_alpha)
    window_alpha = float(args.window_ridge_alpha) if args.window_ridge_alpha is not None else single_alpha
    hidden_sizes = tuple(int(h.strip()) for h in args.hidden_sizes.split(",") if h.strip())

    if args.synthetic:
        return run_synthetic_comparison(
            models_to_run=models_to_run,
            out_dir=args.out_dir,
            granularity=args.granularity,
            single_ridge_alpha=single_alpha,
            window_ridge_alpha=window_alpha,
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
        f"Comparing {models_to_run} | conditions={conditions} granularity={args.granularity} "
        f"channels={ch_tag} windows={windows} shifts={shifts} "
        f"min_holdout_f1={args.min_holdout_f1} single_alpha={single_alpha:g} window_alpha={window_alpha:g}"
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

                subj_rows, art, err = evaluate_subject_comparison(
                    pid,
                    conditions,
                    processed,
                    labeled,
                    task_df,
                    decoder_keys=_keys(grp),
                    agent_keys=_keys(agrp),
                    models_to_run=models_to_run,
                    granularity=args.granularity,
                    window_s=win_s,
                    step_s=args.step_s,
                    embargo_s=args.embargo_s,
                    rate_hz=args.rate_hz,
                    temporal_shift=shift_s,
                    channels=list(channels),
                    pairs=list(pairs),
                    single_ridge_alpha=single_alpha,
                    window_ridge_alpha=window_alpha,
                    hidden_sizes=hidden_sizes,
                    min_holdout_f1=args.min_holdout_f1,
                    seed=args.seed,
                )
                if err is not None:
                    if not is_sweep:
                        print(f"  drop pid={pid}: {err}")
                    continue

                setting_arts.append(art)
                if normalize_granularity(args.granularity) != "continuous":
                    parts = [f"{r['model']}={r['pred_feat_f1']:.3f} (agree={r['clf_agreement_acc']:.2f})" for r in subj_rows]
                    rf1 = subj_rows[0]["real_fnirs_f1"]
                    print(f"  pid={normalize_pid(pid)} real_F={rf1:.3f} | " + " | ".join(parts))
                else:
                    parts = [f"{r['model']}={r['pred_feat_spearman']}" for r in subj_rows]
                    rs = subj_rows[0]["real_fnirs_spearman"]
                    print(f"  pid={normalize_pid(pid)} real_F={rs} | " + " | ".join(parts))

                sweep_rows.extend(subj_rows)

            if not setting_arts:
                print("  (0 participants passed min_holdout_f1 at this setting)")
            else:
                artifacts_by_setting[(win_s, shift_s)] = setting_arts

    if not sweep_rows:
        print("No subjects passed screening across any (window_s, shift_s) setting.")
        return 1

    sweep_df = pd.DataFrame(sweep_rows)
    cond_tag = "_".join(conditions)
    a_tag = f"a{single_alpha:g}_wa{window_alpha:g}".replace(".", "p")
    saved_figs: list[Path] = []

    if is_sweep:
        best_params = _print_sweep_leaderboard(sweep_df, models_to_run, args.granularity)
        sweep_fig = _plot_window_shift_sweep(
            sweep_df=sweep_df,
            models_to_run=models_to_run,
            granularity=args.granularity,
            windows=windows,
            shifts=shifts,
            out_path=args.out_dir / f"compare_{cond_tag}_{args.granularity}_{a_tag}_window_shift_sweep.png",
            title_prefix=f"{'+'.join(conditions)} ({args.granularity.capitalize()}, min_F1≥{args.min_holdout_f1:g})",
        )
        saved_figs.append(sweep_fig)
    else:
        best_params = {"real": (windows[0], shifts[0]), **{m: (windows[0], shifts[0]) for m in models_to_run}}

    w_r, s_r = best_params["real"]
    real_best_df = (
        sweep_df[(sweep_df["window_s"] == w_r) & (sweep_df["temporal_shift"] == s_r)]
        .drop_duplicates(subset=["pid"])
        .copy()
    )
    state_parts = []
    for m_key in models_to_run:
        wm, sm = best_params[m_key]
        state_parts.append(
            sweep_df[
                (sweep_df["model"] == m_key)
                & (sweep_df["window_s"] == wm)
                & (sweep_df["temporal_shift"] == sm)
            ].copy()
        )
    state_best_df = pd.concat(state_parts, ignore_index=True)

    best_arts_by_key = {
        "real": artifacts_by_setting.get(best_params["real"], []),
        **{m: artifacts_by_setting.get(best_params[m], []) for m in models_to_run},
    }

    _print_comparison_summary(real_best_df, state_best_df, best_params, args.granularity)

    param_tag = "per_model_best" if is_sweep else f"w{windows[0]:g}s_sh{shifts[0]:g}s".replace(".", "p")
    best_figs = generate_comparison_figures(
        real_df=real_best_df,
        state_df=state_best_df,
        best_arts_by_key=best_arts_by_key,
        models_to_run=models_to_run,
        best_params=best_params,
        granularity=args.granularity,
        out_dir=args.out_dir,
        stem=f"compare_{cond_tag}_{args.granularity}_{param_tag}_{a_tag}",
        title_prefix=(
            f"{'+'.join(conditions)} ({args.granularity.capitalize()}, min_F1≥{args.min_holdout_f1:g})"
        ),
        show=args.show,
    )
    saved_figs.extend(best_figs)

    print("\nSaved comparison visualizations (no CSV written):")
    for fig_path in saved_figs:
        print(f"  {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
