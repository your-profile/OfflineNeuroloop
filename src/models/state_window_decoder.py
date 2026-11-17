"""Two-stage state-window -> fNIRS window feature -> neural classification decoder.

Stage 1 takes a state trajectory spanning the **same window duration**
(``window_s`` seconds, ``window_steps = round(window_s * rate_hz)`` timesteps)
as the fNIRS window — where each timestep for Robot / Fetch is
``np.concatenate([observation[t], desired_goal[t]])`` — and regresses it via
``Ridge`` onto the eval-compatible fNIRS window feature vector ``F``
(32 features for 8 channels: ``[mean, std, slope, pair_sum, pair_diff]``).

Stage 2 passes those predicted window feature vectors ``F_hat`` into a neural
decoder (e.g. ``RobustWindowDecoder`` or ``SubjectDecoderBank``) to predict
neural optimality classes and class probabilities.

Note: This module is standalone and does not modify any RL training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor

from src.eval.channels import CHANNELS_8, PAIRS_8
from src.eval.cv import fit_score, make_model
from src.eval.data import _resolve_data_file, load_aligned, pid_key
from src.eval.features import build_windows
from src.eval.settings import normalize_granularity
from src.models.decoder_bank import SubjectDecoderBank, normalize_pid
from src.models.eval_compatible import (
    RobustWindowDecoder,
    _assign_windows_to_episodes,
    _embargo_train_mask,
    _episode_spans,
    _robust_stats,
)


def _has_valid_goal(goal: Any) -> bool:
    """Return True when ``goal`` is a non-empty, all-finite array (e.g. Fetch 3D goal)."""
    if goal is None:
        return False
    try:
        arr = np.asarray(goal, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return False
    return arr.size > 0 and bool(np.all(np.isfinite(arr)))


def extract_row_state(
    state_val: Any,
    desired_goal_val: Any = None,
    *,
    require_goal: bool = False,
) -> np.ndarray | None:
    """Extract a 1D state vector from a single ``task_df`` row or Gymnasium step.

    For Robot / Fetch rows (or dict observations), returns
    ``np.concatenate([observation, desired_goal])``. Returns ``None`` if the
    observation (or required goal) is missing or non-finite.
    """
    if state_val is None:
        return None

    if isinstance(state_val, dict):
        obs_raw = state_val.get("observation")
        if desired_goal_val is None:
            desired_goal_val = state_val.get("desired_goal")
    else:
        obs_raw = state_val

    try:
        observation = np.asarray(obs_raw, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None

    if observation.size == 0 or not np.all(np.isfinite(observation)):
        return None

    if _has_valid_goal(desired_goal_val):
        desired_goal = np.asarray(desired_goal_val, dtype=np.float64).ravel()
        return np.concatenate([observation, desired_goal]).astype(np.float64, copy=False)

    if require_goal:
        return None

    return observation.astype(np.float64, copy=False)


def _resample_time_axis(S_3d: np.ndarray, target_steps: int) -> np.ndarray:
    """Resample ``[N, T, D]`` along the time axis ``T`` to ``target_steps``."""
    n_batch, cur_steps, state_dim = S_3d.shape
    if cur_steps == target_steps:
        return S_3d
    if cur_steps == 1:
        return np.broadcast_to(S_3d, (n_batch, target_steps, state_dim)).copy()

    src_t = np.linspace(0.0, 1.0, cur_steps)
    dst_t = np.linspace(0.0, 1.0, target_steps)
    out = np.empty((n_batch, target_steps, state_dim), dtype=np.float64)
    for b in range(n_batch):
        for d in range(state_dim):
            out[b, :, d] = np.interp(dst_t, src_t, S_3d[b, :, d])
    return out


def format_state_window(
    observation: Any,
    desired_goal: Any = None,
    *,
    window_steps: int | None = None,
    per_step_dim: int | None = None,
    n_expected_windows: int | None = None,
) -> np.ndarray:
    """Format state window(s) into a 3D array ``[N, window_steps, state_dim]``."""
    if isinstance(observation, dict):
        vec = extract_row_state(observation, desired_goal)
        if vec is None:
            raise ValueError("Invalid observation/desired_goal in dict input.")
        S_3d = vec[None, None, :]
        return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

    if isinstance(observation, (list, tuple)) and observation:
        if isinstance(observation[0], dict):
            rows = []
            for t_i, item in enumerate(observation):
                g_t = None
                if desired_goal is not None:
                    g_arr = np.asarray(desired_goal)
                    g_t = desired_goal[t_i] if g_arr.ndim >= 2 else desired_goal
                vec = extract_row_state(item, g_t)
                if vec is None:
                    raise ValueError(f"Invalid dict observation at step {t_i}.")
                rows.append(vec)
            S_3d = np.vstack(rows)[None, :, :]
            return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

        if isinstance(observation[0], (list, tuple)) and observation[0] and isinstance(observation[0][0], dict):
            wins = [
                format_state_window(win_dicts, window_steps=window_steps)[0]
                for win_dicts in observation
            ]
            return np.stack(wins, axis=0)

    obs_arr = np.asarray(observation, dtype=np.float64)

    if obs_arr.ndim == 3:
        if desired_goal is None or not _has_valid_goal(desired_goal):
            S_3d = obs_arr
        else:
            goal_arr = np.asarray(desired_goal, dtype=np.float64)
            n_b, n_t, _ = obs_arr.shape
            if goal_arr.ndim == 1:
                goal_3d = np.broadcast_to(goal_arr[None, None, :], (n_b, n_t, goal_arr.size))
            elif goal_arr.ndim == 2:
                if len(goal_arr) == n_b:
                    goal_3d = np.broadcast_to(goal_arr[:, None, :], (n_b, n_t, goal_arr.shape[1]))
                elif len(goal_arr) == n_t and n_b == 1:
                    goal_3d = goal_arr[None, :, :]
                else:
                    raise ValueError(
                        f"2D desired_goal shape {goal_arr.shape} incompatible with 3D observation {obs_arr.shape}."
                    )
            elif goal_arr.ndim == 3:
                if goal_arr.shape[:2] != (n_b, n_t):
                    raise ValueError(
                        f"3D desired_goal shape {goal_arr.shape} incompatible with observation {obs_arr.shape}."
                    )
                goal_3d = goal_arr
            else:
                raise ValueError(f"Unsupported desired_goal ndim={goal_arr.ndim}.")
            S_3d = np.concatenate([obs_arr, goal_3d], axis=2)

        return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

    if obs_arr.ndim == 2:
        if desired_goal is not None and _has_valid_goal(desired_goal):
            goal_arr = np.asarray(desired_goal, dtype=np.float64)
            if goal_arr.ndim == 1:
                goal_2d = np.broadcast_to(goal_arr[None, :], (len(obs_arr), goal_arr.size))
                comb_2d = np.concatenate([obs_arr, goal_2d], axis=1)
                if n_expected_windows is not None and n_expected_windows == len(comb_2d) and window_steps == 1:
                    S_3d = comb_2d[:, None, :]
                else:
                    S_3d = comb_2d[None, :, :]
            elif goal_arr.ndim == 2:
                if len(goal_arr) != len(obs_arr):
                    raise ValueError(
                        f"Shape mismatch: observation {obs_arr.shape} vs desired_goal {goal_arr.shape}."
                    )
                comb_2d = np.concatenate([obs_arr, goal_arr], axis=1)
                if n_expected_windows is not None and n_expected_windows == len(comb_2d) and window_steps == 1:
                    S_3d = comb_2d[:, None, :]
                else:
                    S_3d = comb_2d[None, :, :]
            else:
                raise ValueError(f"Cannot pair 2D observation with {goal_arr.ndim}D desired_goal.")
            return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

        if (
            window_steps is not None
            and per_step_dim is not None
            and obs_arr.shape[1] == window_steps * per_step_dim
            and (per_step_dim != obs_arr.shape[1] or window_steps == 1)
        ):
            return obs_arr.reshape(len(obs_arr), window_steps, per_step_dim)

        if n_expected_windows is not None and len(obs_arr) == n_expected_windows and (
            window_steps is not None and obs_arr.shape[1] % window_steps == 0 and per_step_dim is None
        ):
            d_inferred = obs_arr.shape[1] // window_steps
            return obs_arr.reshape(len(obs_arr), window_steps, d_inferred)

        S_3d = obs_arr[None, :, :]
        return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

    if obs_arr.ndim == 1:
        if desired_goal is not None and _has_valid_goal(desired_goal):
            goal_1d = np.asarray(desired_goal, dtype=np.float64).ravel()
            vec = np.concatenate([obs_arr, goal_1d])
            S_3d = vec[None, None, :]
            return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

        if (
            window_steps is not None
            and per_step_dim is not None
            and obs_arr.size == window_steps * per_step_dim
        ):
            return obs_arr.reshape(1, window_steps, per_step_dim)

        S_3d = obs_arr[None, None, :]
        return _resample_time_axis(S_3d, window_steps) if window_steps is not None else S_3d

    raise ValueError(f"Unsupported observation shape {obs_arr.shape}.")


def format_state_representation(
    observation: Any,
    desired_goal: Any = None,
    *,
    window_steps: int | None = None,
) -> np.ndarray:
    """Format state window(s) into flattened 2D window vectors ``[N, window_steps * state_dim]``."""
    S_3d = format_state_window(observation, desired_goal, window_steps=window_steps)
    return S_3d.reshape(len(S_3d), -1)


def _summarize_state_windows(S_3d: np.ndarray, rate_hz: float = 5.2) -> np.ndarray:
    """Compute ``[mean, std, slope, start, end, delta]`` across time for ``[N, T, D]`` windows."""
    _, n_t, _ = S_3d.shape
    mean = S_3d.mean(axis=1)
    std = S_3d.std(axis=1)
    start = S_3d[:, 0, :]
    end = S_3d[:, -1, :]
    delta = end - start

    if n_t >= 2:
        tt = np.arange(n_t, dtype=np.float64) / max(float(rate_hz), 1e-6)
        tt_c = tt - tt.mean()
        denom = float(np.sum(tt_c ** 2))
        if denom > 1e-12:
            slope = np.sum(tt_c[None, :, None] * (S_3d - mean[:, None, :]), axis=1) / denom
        else:
            slope = np.zeros_like(mean)
    else:
        slope = np.zeros_like(mean)

    return np.concatenate([mean, std, slope, start, end, delta], axis=1)


@dataclass
class StateToWindowDecoder:
    """Predicts 32-dim eval fNIRS window features from a ``window_s``-long state window via Ridge, then classifies.

    Parameters
    ----------
    window_s:
        Window duration in seconds (matches the fNIRS window duration, e.g. ``6.0``s).
    rate_hz:
        Temporal grid rate in Hz (default ``5.2`` Hz, giving ``window_steps = round(window_s * rate_hz)``
        timesteps per window — the exact same temporal length as the fNIRS window).
    window_mode:
        - ``"flatten"`` (default): Concatenates all ``window_steps`` state vectors in the
          window into a single ``(window_steps * per_step_state_dim,)`` vector.
        - ``"summary"``: Extracts ``[mean, std, slope, start, end, delta]`` across the
          ``window_steps`` state trajectory.
    regressor_type:
        ``"ridge"`` (default) or ``"mlp"`` (multi-output ``MLPRegressor``, e.g. for
        ``window_mode="summary"``).
    ridge_alpha:
        L2 regularization strength for multi-output ``Ridge``.
    neural_clf:
        Fitted Stage-2 neural decoder accepting window feature matrices ``F``
        via ``.predict(F)`` / ``.predict_proba(F)`` (e.g. ``RobustWindowDecoder``).
    """

    regressor_type: str = "ridge"
    window_s: float = 6.0
    rate_hz: float = 5.2
    window_mode: str = "flatten"
    ridge_alpha: float = 10.0
    hidden_layer_sizes: tuple[int, ...] = (128, 64)
    activation: str = "relu"
    mlp_alpha: float = 1e-3
    max_iter: int = 400
    early_stopping: bool = True
    seed: int = 42
    granularity: str = "binary"
    neural_clf: Any | None = None

    # Populated during .fit(...)
    feature_regressor: Any = field(default=None, init=False)
    state_mean: np.ndarray | None = field(default=None, init=False)
    state_std: np.ndarray | None = field(default=None, init=False)
    feat_mean: np.ndarray | None = field(default=None, init=False)
    feat_std: np.ndarray | None = field(default=None, init=False)
    window_steps: int = field(default=0, init=False)
    per_step_state_dim: int = field(default=0, init=False)
    state_dim: int = field(default=0, init=False)
    feature_dim: int = field(default=0, init=False)
    n_train: int = field(default=0, init=False)
    n_holdout: int = field(default=0, init=False)
    holdout_metric: float | None = field(default=None, init=False)

    def _make_regressor(self, n_samples: int):
        kind = str(self.regressor_type).lower().strip()
        if kind == "ridge":
            return Ridge(alpha=float(self.ridge_alpha))
        if kind == "mlp":
            use_es = bool(self.early_stopping) and n_samples >= 30
            return MLPRegressor(
                hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
                activation=self.activation,
                alpha=float(self.mlp_alpha),
                solver="adam",
                max_iter=int(self.max_iter),
                early_stopping=use_es,
                random_state=int(self.seed),
            )
        raise ValueError(f"Unknown regressor_type {self.regressor_type!r}. Use 'ridge' or 'mlp'.")

    def _encode_windows(self, S_3d: np.ndarray) -> np.ndarray:
        mode = str(self.window_mode).lower().strip()
        if mode == "flatten":
            return S_3d.reshape(len(S_3d), -1)
        if mode == "summary":
            return _summarize_state_windows(S_3d, rate_hz=self.rate_hz)
        raise ValueError(f"Unknown window_mode {self.window_mode!r}. Use 'flatten' or 'summary'.")

    def _prepare_encoded_input(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        is_fit: bool = False,
        n_expected_windows: int | None = None,
    ) -> np.ndarray:
        target_steps = (
            self.window_steps
            if not is_fit and self.window_steps > 0
            else max(2, int(round(float(self.window_s) * float(self.rate_hz))))
        )
        step_dim = None if is_fit else (self.per_step_state_dim or None)
        S_3d = format_state_window(
            observation,
            desired_goal,
            window_steps=target_steps,
            per_step_dim=step_dim,
            n_expected_windows=n_expected_windows,
        )
        if is_fit:
            self.window_steps = int(S_3d.shape[1])
            self.per_step_state_dim = int(S_3d.shape[2])
        elif S_3d.shape[2] != self.per_step_state_dim:
            raise ValueError(
                f"Expected per-step state_dim={self.per_step_state_dim}, got {S_3d.shape[2]}. "
                "For Robot/Fetch pass [observation, desired_goal] over the window."
            )
        return self._encode_windows(S_3d)

    def _scale_state(self, S: np.ndarray) -> np.ndarray:
        if self.state_mean is None or self.state_std is None:
            raise RuntimeError("StateToWindowDecoder has not been fitted yet.")
        return np.clip((S - self.state_mean) / self.state_std, -10.0, 10.0)

    def _unscale_features(self, Z_feat: np.ndarray) -> np.ndarray:
        if self.feat_mean is None or self.feat_std is None:
            raise RuntimeError("StateToWindowDecoder has not been fitted yet.")
        if Z_feat.ndim == 1:
            Z_feat = Z_feat[:, None] if self.feature_dim == 1 else Z_feat[None, :]
        return Z_feat * self.feat_std + self.feat_mean

    def fit(
        self,
        observation: Any,
        F_targets: np.ndarray,
        *,
        desired_goal: Any = None,
        neural_clf: Any | None = None,
    ) -> "StateToWindowDecoder":
        """Fit Stage-1 ``state_window -> window_features`` Ridge regressor."""
        F = np.asarray(F_targets, dtype=np.float64)
        if F.ndim == 1:
            F = F[:, None]

        S = self._prepare_encoded_input(
            observation,
            desired_goal,
            is_fit=True,
            n_expected_windows=len(F),
        )
        if len(S) != len(F):
            raise ValueError(
                f"State window count ({len(S)}) does not match target window count ({len(F)})."
            )
        if len(S) < 2:
            raise ValueError("Need at least 2 windows to fit StateToWindowDecoder.")

        self.state_dim = int(S.shape[1])
        self.feature_dim = int(F.shape[1])
        self.n_train = int(len(S))

        s_mean = S.mean(axis=0)
        s_std = S.std(axis=0)
        s_std[s_std < 1e-8] = 1.0
        self.state_mean = s_mean
        self.state_std = s_std

        f_mean = F.mean(axis=0)
        f_std = F.std(axis=0)
        f_std[f_std < 1e-8] = 1.0
        self.feat_mean = f_mean
        self.feat_std = f_std

        S_scaled = self._scale_state(S)
        F_scaled = (F - self.feat_mean) / self.feat_std

        reg = self._make_regressor(len(S))
        reg.fit(S_scaled, F_scaled if F_scaled.shape[1] > 1 else F_scaled.ravel())
        self.feature_regressor = reg

        if neural_clf is not None:
            self.neural_clf = neural_clf
        return self

    def predict_features(
        self,
        observation: Any,
        desired_goal: Any = None,
    ) -> np.ndarray:
        """Predict window feature matrix ``F_hat`` of shape ``[N, feature_dim]``."""
        if self.feature_regressor is None:
            raise RuntimeError("StateToWindowDecoder has not been fitted yet.")
        S = self._prepare_encoded_input(observation, desired_goal, is_fit=False)
        S_scaled = self._scale_state(S)
        pred_scaled = np.asarray(self.feature_regressor.predict(S_scaled), dtype=np.float64)
        return self._unscale_features(pred_scaled)

    def predict(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        participant: Any = None,
    ) -> np.ndarray:
        """Two-stage prediction: ``state_window -> F_hat -> neural_clf.predict(F_hat)``."""
        if self.neural_clf is None:
            raise RuntimeError(
                "No neural_clf attached to StateToWindowDecoder; "
                "use predict_features(...) or attach neural_clf."
            )
        F_hat = self.predict_features(observation, desired_goal)
        if participant is not None and hasattr(self.neural_clf, "predict_for"):
            return np.asarray(self.neural_clf.predict_for(F_hat, participant))
        return np.asarray(self.neural_clf.predict(F_hat))

    def predict_proba(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        participant: Any = None,
    ) -> np.ndarray:
        """Two-stage class probabilities: ``state_window -> F_hat -> neural_clf.predict_proba(F_hat)``."""
        if self.neural_clf is None:
            raise RuntimeError("No neural_clf attached to StateToWindowDecoder.")
        F_hat = self.predict_features(observation, desired_goal)
        if participant is not None and hasattr(self.neural_clf, "predict_proba_for"):
            return np.asarray(self.neural_clf.predict_proba_for(F_hat, participant))
        if not hasattr(self.neural_clf, "predict_proba"):
            raise AttributeError("Attached neural_clf does not support predict_proba.")
        return np.asarray(self.neural_clf.predict_proba(F_hat))

    def predict_with_features(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        participant: Any = None,
    ) -> tuple[np.ndarray, np.ndarray | str, np.ndarray]:
        """Return ``(y_pred, proba_or_tag, F_hat)`` in one forward pass."""
        F_hat = self.predict_features(observation, desired_goal)
        if self.neural_clf is None:
            raise RuntimeError("No neural_clf attached to StateToWindowDecoder.")

        if participant is not None and hasattr(self.neural_clf, "predict_for"):
            y_pred = np.asarray(self.neural_clf.predict_for(F_hat, participant))
        else:
            y_pred = np.asarray(self.neural_clf.predict(F_hat))

        try:
            if participant is not None and hasattr(self.neural_clf, "predict_proba_for"):
                proba: np.ndarray | str = np.asarray(
                    self.neural_clf.predict_proba_for(F_hat, participant)
                )
            else:
                proba = np.asarray(self.neural_clf.predict_proba(F_hat))
        except Exception:
            proba = "regression"

        return y_pred, proba, F_hat

    @property
    def classes_(self):
        if self.neural_clf is None:
            return None
        return getattr(self.neural_clf, "classes_", None)

    @staticmethod
    def score_feature_predictions(F_true: np.ndarray, F_pred: np.ndarray) -> dict[str, float]:
        """Evaluate Stage-1 window feature reconstruction quality."""
        Ft = np.asarray(F_true, dtype=np.float64)
        Fp = np.asarray(F_pred, dtype=np.float64)
        mse = float(mean_squared_error(Ft, Fp))
        mae = float(np.mean(np.abs(Ft - Fp)))
        r2 = float(r2_score(Ft, Fp, multioutput="variance_weighted")) if len(Ft) >= 2 else float("nan")

        num = np.sum(Ft * Fp, axis=1)
        denom = np.linalg.norm(Ft, axis=1) * np.linalg.norm(Fp, axis=1)
        valid = denom > 1e-12
        cos = float(np.mean(num[valid] / denom[valid])) if np.any(valid) else float("nan")
        return {
            "feature_r2": r2,
            "feature_mse": mse,
            "feature_mae": mae,
            "feature_cosine_sim": cos,
        }

    def score_classification(
        self,
        observation: Any,
        y_true: np.ndarray,
        *,
        desired_goal: Any = None,
        F_true: np.ndarray | None = None,
        participant: Any = None,
    ) -> dict[str, Any]:
        """Evaluate Stage-2 neural classification (and Stage-1 features if ``F_true`` given)."""
        yt = np.asarray(y_true).ravel()
        F_hat = self.predict_features(observation, desired_goal)
        out: dict[str, Any] = {"n": int(yt.size)}

        if F_true is not None:
            out.update(self.score_feature_predictions(F_true, F_hat))

        if self.neural_clf is None or yt.size == 0:
            return out

        g = normalize_granularity(self.granularity)
        if participant is not None and hasattr(self.neural_clf, "predict_for"):
            yp = np.asarray(self.neural_clf.predict_for(F_hat, participant)).ravel()
        else:
            yp = np.asarray(self.neural_clf.predict(F_hat)).ravel()

        if g == "continuous":
            yt_f = yt.astype(float)
            yp_f = yp.astype(float)
            out["r2"] = float(r2_score(yt_f, yp_f)) if yt_f.size >= 2 else float("nan")
            out["mse"] = float(mean_squared_error(yt_f, yp_f))
            out["mae"] = float(np.mean(np.abs(yt_f - yp_f)))
            if yt_f.size >= 2 and np.std(yt_f) > 0 and np.std(yp_f) > 0:
                rho, _ = spearmanr(yt_f, yp_f)
                out["spearman"] = float(rho) if np.isfinite(rho) else None
            else:
                out["spearman"] = None
            out["holdout_metric"] = out["spearman"]
            out["metric_name"] = "Spearman"
            return out

        yt_i = yt.astype(int)
        yp_i = yp.astype(int)
        out["accuracy"] = float(accuracy_score(yt_i, yp_i))
        out["holdout_f1"] = float(f1_score(yt_i, yp_i, average="macro", zero_division=0))
        out["holdout_auc"] = _compute_clf_auc(self.neural_clf, F_hat, yt_i, g, participant=participant)
        out["holdout_metric"] = out["holdout_f1"]
        out["metric_name"] = "macroF1"
        return out


class SubjectStateDecoderBank(SubjectDecoderBank):
    """Per-subject / ensemble bank of ``StateToWindowDecoder`` models."""

    def _window_steps_hint(self) -> int | None:
        first = next(iter(self.models.values()), None)
        ws = getattr(first, "window_steps", 0)
        return int(ws) if ws and ws > 0 else None

    def _step_dim_hint(self) -> int | None:
        first = next(iter(self.models.values()), None)
        sd = getattr(first, "per_step_state_dim", 0)
        return int(sd) if sd and sd > 0 else None

    def predict_features_for(
        self,
        observation: Any,
        participant: Any,
        desired_goal: Any = None,
    ) -> np.ndarray:
        S_3d = format_state_window(
            observation,
            desired_goal,
            window_steps=self._window_steps_hint(),
            per_step_dim=self._step_dim_hint(),
        )
        model, _ = self._resolve(participant)
        if model is not None:
            return model.predict_features(S_3d)
        return self.predict_features(S_3d)

    def predict_features(
        self,
        observation: Any,
        desired_goal: Any = None,
    ) -> np.ndarray:
        S_3d = format_state_window(
            observation,
            desired_goal,
            window_steps=self._window_steps_hint(),
            per_step_dim=self._step_dim_hint(),
        )
        if self.mode in ("single_subject", "matched", "per_subject"):
            model, _ = self._resolve(None)
            return model.predict_features(S_3d)
        preds = [m.predict_features(S_3d) for m in self.models.values()]
        return np.mean(preds, axis=0)

    def predict_state(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        participant: Any = None,
    ) -> np.ndarray:
        S_3d = format_state_window(
            observation,
            desired_goal,
            window_steps=self._window_steps_hint(),
            per_step_dim=self._step_dim_hint(),
        )
        if participant is not None:
            return self.predict_for(S_3d, participant)
        return self.predict(S_3d)

    def predict_proba_state(
        self,
        observation: Any,
        desired_goal: Any = None,
        *,
        participant: Any = None,
    ) -> np.ndarray:
        S_3d = format_state_window(
            observation,
            desired_goal,
            window_steps=self._window_steps_hint(),
            per_step_dim=self._step_dim_hint(),
        )
        if participant is not None:
            return self.predict_proba_for(S_3d, participant)
        return self.predict_proba(S_3d)


def _compute_clf_auc(
    clf: Any,
    F: np.ndarray,
    y_true: np.ndarray,
    granularity: str,
    *,
    participant: Any = None,
) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    inner = getattr(clf, "model", clf)
    try:
        if granularity == "binary":
            if participant is None and hasattr(inner, "decision_function"):
                scores = inner.decision_function(F)
                return float(roc_auc_score(y_true, scores))
            if participant is not None and hasattr(clf, "predict_proba_for"):
                proba = clf.predict_proba_for(F, participant)[:, 1]
            else:
                proba = clf.predict_proba(F)[:, 1]
            return float(roc_auc_score(y_true, proba))
        if granularity == "discrete":
            if participant is not None and hasattr(clf, "predict_proba_for"):
                proba = clf.predict_proba_for(F, participant)
            else:
                proba = clf.predict_proba(F)
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return None
    return None


def _prepare_subject_task_rows(
    task_df: pd.DataFrame,
    pid: str,
    cond: str | None = None,
    *,
    require_goal: bool | None = None,
) -> pd.DataFrame:
    """Filter ``task_df`` to ``pid`` (+ optional ``cond``) and precompute valid state vectors."""
    df = task_df.copy()
    df["participantKey"] = df["participantKey"].astype(str)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype("Int64")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["participantKey"].map(normalize_pid) == normalize_pid(pid)]
    if cond is not None and "condition" in df.columns:
        df = df[df["condition"].astype(str).str.strip().str.upper() == str(cond).strip().upper()]
    df = df.dropna(subset=["episode", "time"]).sort_values("time").reset_index(drop=True)
    if df.empty:
        return df

    if require_goal is None:
        if cond is not None:
            require_goal = str(cond).upper().startswith("R")
        elif "condition" in df.columns and len(df):
            require_goal = str(df["condition"].iloc[0]).upper().startswith("R")
        else:
            require_goal = False

    has_goal_col = "desired_goal" in df.columns
    vecs: list[np.ndarray | None] = []
    for idx in range(len(df)):
        s_val = df["states"].iloc[idx]
        g_val = df["desired_goal"].iloc[idx] if has_goal_col else None
        vecs.append(extract_row_state(s_val, g_val, require_goal=bool(require_goal)))

    df = df.assign(_state_vec=vecs)
    df = df[df["_state_vec"].notna()].reset_index(drop=True)
    return df


def _match_windows_to_episode_state_windows(
    Tw: np.ndarray,
    t0: pd.Timestamp,
    window_s: float,
    rate_hz: float,
    ep_indices: np.ndarray,
    spans: list[tuple[tuple, pd.Timestamp, pd.Timestamp]],
    subject_task_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a ``[window_steps, state_dim]`` state trajectory for each fNIRS window."""
    window_steps = max(2, int(round(float(window_s) * float(rate_hz))))
    if subject_task_df.empty or len(Tw) == 0:
        return np.empty((0, window_steps, 0), dtype=np.float64), np.zeros(len(Tw), dtype=bool)

    expected_dim: int | None = None
    grouped_rows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (pk, ep), g in subject_task_df.groupby(["participantKey", "episode"], sort=False):
        g_sorted = g.sort_values("time")
        raw_vecs = list(g_sorted["_state_vec"])
        raw_t = pd.DatetimeIndex(g_sorted["time"]).asi8.astype(np.int64, copy=False)
        valid_vecs = []
        valid_t = []
        for t_val, v in zip(raw_t, raw_vecs):
            if expected_dim is None:
                expected_dim = int(v.size)
            if v.size == expected_dim:
                valid_vecs.append(v)
                valid_t.append(t_val)
        if valid_vecs:
            grouped_rows[(str(pk), int(ep))] = (
                np.asarray(valid_t, dtype=np.int64),
                np.vstack(valid_vecs),
            )

    if not grouped_rows or expected_dim is None:
        return np.empty((0, window_steps, 0), dtype=np.float64), np.zeros(len(Tw), dtype=bool)

    step_offsets_ns = np.round(
        (np.arange(window_steps, dtype=np.float64) / float(rate_hz)) * 1e9
    ).astype(np.int64)

    abs_start = pd.to_datetime(t0, utc=True) + pd.to_timedelta(Tw, unit="s")
    start_ns = pd.DatetimeIndex(abs_start).asi8.astype(np.int64, copy=False)

    matched_windows: list[np.ndarray] = []
    keep_mask = np.zeros(len(Tw), dtype=bool)

    for i, si in enumerate(ep_indices):
        if si < 0 or si >= len(spans):
            continue
        span_key = spans[si][0]
        norm_key = (str(span_key[0]), int(span_key[1]))
        entry = grouped_rows.get(norm_key)
        if entry is None:
            continue
        t_ns, ep_mat = entry
        if len(t_ns) == 0:
            continue

        grid_ns = start_ns[i] + step_offsets_ns
        pos = np.searchsorted(t_ns, grid_ns, side="right") - 1
        pos = np.clip(pos, 0, len(ep_mat) - 1)
        matched_windows.append(ep_mat[pos])
        keep_mask[i] = True

    if not matched_windows:
        return np.empty((0, window_steps, expected_dim), dtype=np.float64), np.zeros(len(Tw), dtype=bool)

    return np.stack(matched_windows, axis=0), keep_mask


def build_subject_state_window_dataset(
    pid: Any,
    conditions: Sequence[str],
    processed_dir: str | Path,
    labeled_dir: str | Path,
    task_df: pd.DataFrame,
    *,
    granularity: str = "binary",
    window_s: float = 6.0,
    step_s: float = 1.0,
    rate_hz: float = 5.2,
    temporal_shift: float = 0.0,
    channels: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    require_goal: bool | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Build aligned ``(S, F, y, T, ep_i, spans, med, mad)`` arrays for one participant.

    - ``S`` has shape ``[N_windows, window_steps, state_dim]``.
    - ``F`` has shape ``[N_windows, 3 * len(channels) + 2 * len(pairs)]`` (32 features for 8ch).
    """
    channels = list(channels or CHANNELS_8)
    pairs = list(pairs or PAIRS_8)
    g = normalize_granularity(granularity)
    pid_str = pid_key(pid)

    win = dict(
        granularity=g,
        window_s=float(window_s),
        step_s=float(step_s),
        rate=float(rate_hz),
        ambig_lo=0.25,
        ambig_hi=0.75,
        min_majority=0.5,
        min_windows=20,
        min_per_class=8,
        min_std=1e-6,
        temporal_shift=float(temporal_shift),
    )

    spans = _episode_spans(task_df, pid_str)
    if not spans:
        return None, {"error": "no episode spans", "pid": pid_str}

    S_all, F_all, y_all, T_all, ep_idx_all = [], [], [], [], []
    Zu_pool = []
    skip_reasons: list[str] = []

    for cond in conditions:
        raw, reason = load_aligned(
            pid_str,
            cond,
            processed_dir,
            labeled_dir,
            channels=channels,
            granularity=g,
            robust_scale=True,
            return_reason=True,
        )
        if raw is None:
            skip_reasons.append(f"{cond}: {reason}")
            continue
        ts, Z, tl, y = raw

        sp = _resolve_data_file(Path(processed_dir), pid_str, cond, "processed")
        if sp is None:
            skip_reasons.append(f"{cond}: processed file vanished after load")
            continue
        s_csv = pd.read_csv(sp)
        s_csv["time"] = pd.to_datetime(s_csv["time"], utc=True)
        t0 = s_csv["time"].iloc[0]

        built = build_windows(ts, Z, tl, y, channels=channels, pairs=pairs, **win)
        if built is None:
            skip_reasons.append(f"{cond}: build_windows returned None")
            continue
        F_cond, y_cond, T_cond = built

        assign = _assign_windows_to_episodes(T_cond, t0, spans)
        subj_task = _prepare_subject_task_rows(
            task_df, pid_str, cond=cond, require_goal=require_goal
        )
        S_cond, matched_mask = _match_windows_to_episode_state_windows(
            T_cond, t0, float(window_s), float(rate_hz), assign, spans, subj_task
        )
        if not matched_mask.any():
            skip_reasons.append(f"{cond}: no windows matched valid task state trajectories")
            continue

        raw_u, _ = load_aligned(
            pid_str,
            cond,
            processed_dir,
            labeled_dir,
            channels=channels,
            granularity=g,
            robust_scale=False,
            return_reason=True,
        )
        if raw_u is not None:
            Zu_pool.append(raw_u[1])

        S_all.append(S_cond)
        F_all.append(F_cond[matched_mask])
        y_all.append(y_cond[matched_mask])
        T_all.append(T_cond[matched_mask])
        ep_idx_all.append(assign[matched_mask])

    if not S_all or not Zu_pool:
        detail = "; ".join(skip_reasons) if skip_reasons else "unknown"
        return None, {
            "error": "no aligned state-window samples",
            "detail": detail,
            "pid": pid_str,
            "conditions": list(conditions),
        }

    med_ref, mad_ref = _robust_stats(np.vstack(Zu_pool))
    bundle = {
        "S": np.concatenate(S_all, axis=0),
        "F": np.vstack(F_all),
        "y": np.concatenate(y_all),
        "T": np.concatenate(T_all),
        "ep_i": np.concatenate(ep_idx_all),
        "spans": spans,
        "med": med_ref,
        "mad": mad_ref,
        "channels": channels,
        "pairs": pairs,
        "granularity": g,
        "window_s": float(window_s),
        "rate_hz": float(rate_hz),
    }
    return bundle, {"status": "ok"}


def train_subject_state_window_decoder(
    pid: Any,
    conditions: Sequence[str],
    processed_dir: str | Path,
    labeled_dir: str | Path,
    task_df: pd.DataFrame,
    decoder_keys: set[tuple],
    agent_keys: set[tuple] | None = None,
    *,
    granularity: str = "binary",
    window_s: float = 6.0,
    step_s: float = 1.0,
    embargo_s: float = 2.0,
    rate_hz: float = 5.2,
    temporal_shift: float = 5.0,
    channels: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    window_mode: str = "flatten",
    regressor_type: str = "ridge",
    ridge_alpha: float = 10.0,
    hidden_layer_sizes: Sequence[int] = (128, 64),
    mlp_alpha: float = 1e-3,
    max_iter: int = 400,
    early_stopping: bool = True,
    neural_clf: RobustWindowDecoder | None = None,
    require_goal: bool | None = None,
    seed: int = 42,
) -> tuple[StateToWindowDecoder | None, dict[str, Any]]:
    """Fit a per-subject two-stage ``StateToWindowDecoder`` on full ``window_s`` state trajectories."""
    agent_keys = agent_keys or set()
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
        require_goal=require_goal,
    )
    if bundle is None:
        return None, status

    S = bundle["S"]
    F = bundle["F"]
    y = bundle["y"]
    T = bundle["T"]
    ep_i = bundle["ep_i"]
    spans = bundle["spans"]
    g = bundle["granularity"]
    gap = float(window_s) + float(embargo_s)

    def _as_key(k):
        return (str(k[0]), int(k[1]))

    span_keys = [sp[0] for sp in spans]
    decoder_keys_n = {_as_key(k) for k in decoder_keys}
    agent_keys_n = {_as_key(k) for k in agent_keys}

    is_dec = np.zeros(len(y), dtype=bool)
    is_te = np.zeros(len(y), dtype=bool)
    for i, si in enumerate(ep_i):
        if si < 0:
            continue
        key = _as_key(span_keys[si])
        if key in decoder_keys_n:
            is_dec[i] = True
        if key in agent_keys_n:
            is_te[i] = True

    tr = is_dec & _embargo_train_mask(T, is_te, gap) if is_te.any() else is_dec
    te = is_te

    if tr.sum() < 15:
        return None, {"error": f"too few train state-windows after embargo ({int(tr.sum())})"}
    if g != "continuous" and len(np.unique(y[tr])) < 2:
        return None, {"error": "train windows lack both classes"}

    if neural_clf is None:
        stage2_est = make_model(g)
        stage2_est.fit(F[tr], y[tr])
        neural_clf = RobustWindowDecoder(
            model=stage2_est,
            med=bundle["med"],
            mad=bundle["mad"],
            channels=bundle["channels"],
            pairs=bundle["pairs"],
            granularity=g,
            n_train=int(tr.sum()),
            n_holdout=int(te.sum()),
        )

    decoder = StateToWindowDecoder(
        regressor_type=regressor_type,
        window_s=float(window_s),
        rate_hz=float(rate_hz),
        window_mode=window_mode,
        ridge_alpha=float(ridge_alpha),
        hidden_layer_sizes=tuple(int(h) for h in hidden_layer_sizes),
        mlp_alpha=float(mlp_alpha),
        max_iter=int(max_iter),
        early_stopping=bool(early_stopping),
        seed=int(seed),
        granularity=g,
        neural_clf=neural_clf,
    )
    decoder.fit(S[tr], F[tr])
    decoder.n_holdout = int(te.sum())

    report: dict[str, Any] = {
        "pid": pid_key(pid),
        "window_steps": decoder.window_steps,
        "per_step_state_dim": decoder.per_step_state_dim,
        "state_dim": decoder.state_dim,
        "window_mode": window_mode,
        "feature_dim": decoder.feature_dim,
        "regressor_type": regressor_type,
        "ridge_alpha": float(ridge_alpha),
        "n_train": int(tr.sum()),
        "n_holdout": int(te.sum()),
        "n_embargo_dropped": int(is_dec.sum() - tr.sum()),
        "gap_s": gap,
        "temporal_shift": float(temporal_shift),
        "window_s": float(window_s),
    }

    if te.sum() >= 10 and (g == "continuous" or len(np.unique(y[te])) >= 2):
        te_scores = decoder.score_classification(S[te], y[te], F_true=F[te])
        report.update(te_scores)
        decoder.holdout_metric = te_scores.get("holdout_metric")

        oracle_primary, oracle_extra = fit_score(F[tr], y[tr], F[te], y[te], granularity=g)
        if g == "continuous":
            report["oracle_neural_spearman"] = (
                float(oracle_primary) if oracle_primary == oracle_primary else None
            )
        else:
            report["oracle_neural_auc"] = (
                float(oracle_primary) if oracle_primary == oracle_primary else None
            )
            report["oracle_neural_f1"] = (
                float(oracle_extra) if oracle_extra == oracle_extra else None
            )
    else:
        if te.sum() >= 2:
            F_hat_te = decoder.predict_features(S[te])
            report.update(decoder.score_feature_predictions(F[te], F_hat_te))
        report["holdout_metric"] = None
        report["holdout_f1"] = None
        report["holdout_auc"] = None
        report["note"] = "holdout too small or single-class"

    return decoder, report
