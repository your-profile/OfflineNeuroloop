import random

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

"""
ModelTrainer():
Trains the neural decoder.
Supports shrinkage LDA / Ridge (eval default) or MLP.
Handles model noise injection for the RL loop.
"""


class ModelTrainer:
    def __init__(self, cfg, seed, verbose=False):
        self.model_type = str(cfg.get("type", cfg.get("model_type", "lda"))).lower().strip()
        self.model_noise = float(cfg.get("model_noise", 0.0))
        self.ridge_alpha = float(cfg.get("ridge_alpha", 10.0))
        self.n_folds = int(cfg.get("n_folds", 8))
        self.window_s = float(cfg.get("window_s", 8.0))
        self.embargo_s = float(cfg.get("embargo_s", 4.0))
        # MLP hyperparameters (only required when type == mlp)
        self.binary_hidden_layer_sizes = tuple(cfg.get("binary_hidden_layer_sizes", [50, 20, 5]))
        self.ternary_hidden_layer_sizes = tuple(cfg.get("ternary_hidden_layer_sizes", [75, 50, 5]))
        self.regressor_hidden_layer_sizes = tuple(cfg.get("regressor_hidden_layer_sizes", [100, 40, 2]))
        self.reg_activation = cfg.get("reg_activation", "tanh")
        self.binary_activation = cfg.get("binary_activation", "relu")
        self.ternary_activation = cfg.get("ternary_activation", "relu")
        self.early_stop = cfg.get("early_stopping", True)
        self.binary_alpha = cfg.get("binary_alpha", 5e-4)
        self.ternary_alpha = cfg.get("ternary_alpha", 1e-4)
        self.reg_alpha = cfg.get("reg_alpha", 5e-4)
        self.max_iter = 200
        self.seed = seed
        self.verbose = verbose
        self._np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._py_rng = random.Random(seed) if seed is not None else random.Random()

    def get_report(self, y_test, y_pred, classifier=False):
        y_test = np.asarray(y_test).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_test.size == 0 or y_pred.size == 0:
            return "no samples"

        if classifier:
            y_test = [int(x) for x in y_test]
            y_pred = [int(x) for x in y_pred]
            return classification_report(y_test, y_pred, output_dict=False)

        y_test = np.array([float(x) for x in y_test])
        y_pred = np.array([float(x) for x in y_pred])
        return {
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": float(np.mean(np.abs(y_test - y_pred))),
        }

    def noisy_output(self, model, X, granularity, flip_rate):
        if granularity[0] == "c":
            return self.noisy_regressor(model, X, flip_rate)
        if granularity[0] == "b":
            return self.noisy_binary(model, X, flip_rate)
        if granularity[0] == "t":
            return self.noisy_ternary(model, X, flip_rate)

    def flip_labels(self, prediction, flip_rate, classes):
        """Randomly reassign a fraction of predictions to a wrong class with noise."""
        if self._py_rng.random() < flip_rate:
            wrong_classes = [c for c in classes if c != prediction]
            noisy = self._np_rng.choice(wrong_classes)
        else:
            noisy = prediction

        if self.verbose:
            print(f"Flipped label from {prediction} to {noisy}")
            print(f"Flip rate: {flip_rate}")

        return noisy

    def noisy_binary(self, model, preds, flip_rate=0.1):
        return self.flip_labels(preds, flip_rate, classes=[0, 1])

    def noisy_ternary(self, model, preds, flip_rate=0.1):
        return self.flip_labels(preds, flip_rate, classes=[0, 1, 2])

    def noisy_regressor(self, model, preds, noise_level=0.1):
        if self._py_rng.random() < noise_level:
            return self._np_rng.random()
        return preds

    def _make_lda_or_ridge(self, granularity: str):
        g = str(granularity).lower().strip()
        if g.startswith("c"):
            return make_pipeline(StandardScaler(), Ridge(alpha=self.ridge_alpha))
        return LDA(solver="lsqr", shrinkage="auto")

    def _make_mlp_classifier(self, granularity: str, random_state: int):
        if granularity in ("discrete", "ternary"):
            return MLPClassifier(
                hidden_layer_sizes=self.ternary_hidden_layer_sizes,
                activation=self.ternary_activation,
                alpha=self.ternary_alpha,
                solver="adam",
                max_iter=self.max_iter,
                early_stopping=self.early_stop,
                random_state=random_state,
            )
        return MLPClassifier(
            hidden_layer_sizes=self.binary_hidden_layer_sizes,
            activation=self.binary_activation,
            alpha=self.binary_alpha,
            solver="adam",
            max_iter=self.max_iter,
            early_stopping=self.early_stop,
            random_state=random_state,
        )

    def _purged_holdout_mask(self, n: int, test_size: float):
        """Contiguous end-block holdout + embargo gap (same idea as eval purged CV)."""
        n_te = max(int(round(n * test_size)), 5)
        n_te = min(n_te, n - 30) if n > 40 else max(n // 5, 1)
        te = np.zeros(n, dtype=bool)
        te[n - n_te :] = True
        # embargo: drop train samples within window+embargo of the test block start
        gap = int(round((self.window_s + self.embargo_s)))  # in window steps ≈ seconds at 1 Hz proxy
        cut = max(0, n - n_te - gap)
        tr = np.zeros(n, dtype=bool)
        tr[:cut] = True
        if tr.sum() < 30:
            tr = ~te
        return tr, te

    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.1,
        granularity="binary",
        random_state: int = 42,
        shuffle_data: bool = True,
        T: np.ndarray | None = None,
    ):
        if str(granularity).lower().startswith("c"):
            return self.train_regressor(
                X, y, test_size, random_state, shuffle_data, T=T
            )

        use_lda = self.model_type in ("lda", "shrinkage_lda", "ridge_lda")

        if use_lda:
            # Purged contiguous holdout for the report; fit returned model on all rows.
            if T is not None and len(T) == len(y):
                from src.eval.cv import purged_oof

                metric, pred, _blk, ok = purged_oof(
                    X,
                    y,
                    T,
                    n_folds=self.n_folds,
                    window_s=self.window_s,
                    embargo_s=self.embargo_s,
                    granularity=granularity,
                )
                clf = self._make_lda_or_ridge(granularity)
                clf.fit(X, y)
                g0 = str(granularity).lower()[0]
                name = "AUC" if g0 == "b" else "macroF1"
                report = {f"purged_{name}": metric, "n_oof": int(ok.sum())}
                if g0 == "t":
                    y_hat = pred[ok].astype(int)
                    report["classification_report"] = classification_report(
                        y[ok].astype(int), y_hat, output_dict=False
                    )
                if self.verbose:
                    print(f"LDA purged-CV {name}={metric}")
                return clf, report

            tr, te = self._purged_holdout_mask(len(y), test_size)
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
            clf_eval = self._make_lda_or_ridge(granularity)
            clf_eval.fit(X_tr, y_tr)
            y_pred = clf_eval.predict(X_te)
            report = classification_report(y_te.astype(int), y_pred.astype(int), output_dict=False)
            clf = self._make_lda_or_ridge(granularity)
            clf.fit(X, y)
            if self.verbose:
                print(
                    f"Training shrinkage LDA (solver=lsqr, shrinkage=auto); "
                    f"purged holdout report on {te.sum()} windows"
                )
            return clf, report

        # MLP path (legacy)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle_data,
            stratify=y,
        )
        clf = self._make_mlp_classifier(granularity, random_state)
        if self.verbose:
            print(f"Training MLP classifier ({granularity})")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = classification_report(y_te, y_pred, output_dict=False)
        return clf, report

    def train_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.1,
        random_state: int = 42,
        shuffle_data: bool = True,
        T: np.ndarray | None = None,
    ):
        use_ridge = self.model_type in ("lda", "shrinkage_lda", "ridge_lda", "ridge")

        if use_ridge:
            if T is not None and len(T) == len(y):
                from src.eval.cv import purged_oof

                metric, _pred, _blk, _ok = purged_oof(
                    X,
                    y,
                    T,
                    n_folds=self.n_folds,
                    window_s=self.window_s,
                    embargo_s=self.embargo_s,
                    granularity="continuous",
                )
                clf = self._make_lda_or_ridge("continuous")
                clf.fit(X, y)
                if self.verbose:
                    print(f"Ridge purged-CV Spearman={metric}")
                return clf, {"purged_spearman": metric}

            tr, te = self._purged_holdout_mask(len(y), test_size)
            X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
            clf_eval = self._make_lda_or_ridge("continuous")
            clf_eval.fit(X_tr, y_tr)
            y_pred = clf_eval.predict(X_te)
            report = {
                "R2": r2_score(y_te, y_pred),
                "MSE": mean_squared_error(y_te, y_pred),
                "MAE": float(np.mean(np.abs(y_te - y_pred))),
            }
            clf = self._make_lda_or_ridge("continuous")
            clf.fit(X, y)
            if self.verbose:
                print(f"Training Ridge(alpha={self.ridge_alpha}) with purged holdout report")
            return clf, report

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle_data
        )
        clf = MLPRegressor(
            hidden_layer_sizes=self.regressor_hidden_layer_sizes,
            activation=self.reg_activation,
            alpha=self.reg_alpha,
            solver="adam",
            max_iter=self.max_iter,
            early_stopping=self.early_stop,
            random_state=random_state,
        )
        if self.verbose:
            print("Training MLP regressor")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = {
            "R2": r2_score(y_te, y_pred),
            "MSE": mean_squared_error(y_te, y_pred),
            "MAE": float(np.mean(np.abs(y_te - y_pred))),
        }
        return clf, report
