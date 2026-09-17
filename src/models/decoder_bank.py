"""Per-participant decoder bank for single-subject and ensemble RL use."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_pid(participant) -> str:
    """Map participantKey ('010RW') or int/str id to zero-padded pid ('010')."""
    if participant is None:
        return ""
    s = str(participant)
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return s
    # keys are NNN + condition (e.g. 010RW) — take leading participant id
    if len(digits) >= 3 and not s.isdigit():
        return digits[:3].zfill(3)
    return digits.zfill(3)


class SubjectDecoderBank:
    """Sklearn-like wrapper over {pid: fitted_estimator}.

    Modes
    -----
    single_subject / matched
        ``predict_for(X, participant)`` uses that subject's model.
        Plain ``predict`` uses the last ``set_participant`` / falls back to majority
        model if unset (prefer passing participant through get_neural_signal).
    ensemble
        Soft-vote (proba mean) or hard-vote / mean for continuous.
    """

    def __init__(
        self,
        models: dict[str, Any],
        *,
        mode: str = "ensemble",
        granularity: str = "binary",
        reports: dict[str, Any] | None = None,
    ):
        if not models:
            raise ValueError("SubjectDecoderBank requires at least one fitted model")
        self.models = {normalize_pid(k): v for k, v in models.items()}
        self.mode = str(mode).lower().strip()
        self.granularity = str(granularity).lower().strip()
        self.reports = reports or {}
        self._active_pid: str | None = None
        self.classes_ = self._infer_classes()

    def _infer_classes(self):
        for m in self.models.values():
            if hasattr(m, "classes_"):
                return np.asarray(m.classes_)
        if self.granularity.startswith("t"):
            return np.asarray([0, 1, 2])
        if self.granularity.startswith("c"):
            return None
        return np.asarray([0, 1])

    def set_participant(self, participant) -> None:
        self._active_pid = normalize_pid(participant)

    def _resolve(self, participant=None):
        pid = normalize_pid(participant) if participant is not None else self._active_pid
        if pid and pid in self.models:
            return self.models[pid], pid
        if self.mode in ("single_subject", "matched", "per_subject"):
            raise KeyError(
                f"No decoder for participant {pid!r}. "
                f"Available: {sorted(self.models)}"
            )
        # ensemble fallback for unmatched pid: use all models
        return None, pid

    def predict_for(self, X, participant):
        X = np.asarray(X)
        model, _ = self._resolve(participant)
        if model is not None:
            return model.predict(X)
        return self.predict(X)

    def predict_proba_for(self, X, participant):
        X = np.asarray(X)
        model, _ = self._resolve(participant)
        if model is not None:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            raise AttributeError("model has no predict_proba")
        return self.predict_proba(X)

    def predict_raw_for(self, X_raw, participant):
        """Predict from raw [T,C] window when subject models are RobustWindowDecoder."""
        model, _ = self._resolve(participant)
        if model is None:
            # ensemble raw: majority / mean over subjects
            if self.granularity.startswith("c"):
                preds = [m.predict_raw(X_raw)[0] for m in self.models.values() if hasattr(m, "predict_raw")]
                return np.asarray([float(np.mean(preds))])
            votes = [int(m.predict_raw(X_raw)[0]) for m in self.models.values() if hasattr(m, "predict_raw")]
            vals, counts = np.unique(votes, return_counts=True)
            return np.asarray([vals[np.argmax(counts)]])
        if hasattr(model, "predict_raw"):
            return model.predict_raw(X_raw)
        raise AttributeError("model has no predict_raw")

    def predict_proba_raw_for(self, X_raw, participant):
        model, _ = self._resolve(participant)
        if model is None:
            probas = [m.predict_proba_raw(X_raw)[0] for m in self.models.values() if hasattr(m, "predict_proba_raw")]
            return np.asarray([np.mean(probas, axis=0)])
        if hasattr(model, "predict_proba_raw"):
            return model.predict_proba_raw(X_raw)
        raise AttributeError("model has no predict_proba_raw")

    def predict(self, X):
        X = np.asarray(X)
        if self.mode in ("single_subject", "matched", "per_subject"):
            model, _ = self._resolve(None)
            return model.predict(X)

        # ensemble
        if self.granularity.startswith("c"):
            preds = np.column_stack([m.predict(X) for m in self.models.values()])
            return preds.mean(axis=1)

        if all(hasattr(m, "predict_proba") for m in self.models.values()):
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # hard vote
        votes = np.column_stack([m.predict(X).astype(int) for m in self.models.values()])
        out = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.asarray(out)

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.mode in ("single_subject", "matched", "per_subject"):
            model, _ = self._resolve(None)
            return model.predict_proba(X)

        probas = []
        for m in self.models.values():
            if not hasattr(m, "predict_proba"):
                raise AttributeError("ensemble predict_proba requires classifiers with predict_proba")
            p = m.predict_proba(X)
            # align columns to self.classes_
            if self.classes_ is not None and hasattr(m, "classes_"):
                aligned = np.zeros((len(X), len(self.classes_)), dtype=float)
                for j, c in enumerate(m.classes_):
                    if c in self.classes_:
                        aligned[:, list(self.classes_).index(c)] = p[:, j]
                probas.append(aligned)
            else:
                probas.append(p)
        return np.mean(probas, axis=0)

    def __repr__(self) -> str:
        return (
            f"SubjectDecoderBank(mode={self.mode!r}, "
            f"n={len(self.models)}, pids={sorted(self.models)})"
        )
