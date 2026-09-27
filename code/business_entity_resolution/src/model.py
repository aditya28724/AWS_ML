"""LightGBM pairwise match classifier (multithreaded, early stopping, sample weights)."""
import inspect
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from logger import fmt_int, fmt_pct, get_logger

log = get_logger(__name__)


def pair_scores(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "logloss": float("nan")}
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))}


def _log_every(period: int):
    def _callback(env):
        if period and (env.iteration + 1) % period == 0:
            res = ", ".join(f"{name} {val:.5f}" for _, name, val, _ in env.evaluation_result_list)
            log.info("    round %4d | %s", env.iteration + 1, res)
    _callback.order = 10
    return _callback


class MatchModel:
    def __init__(self, params: dict, seed: int = 42, threads: int = -1):
        self.params = params
        self.seed = seed
        self.threads = threads
        self.model: lgb.LGBMClassifier | None = None
        self.features: list[str] = []
        self.best_iteration: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, features: list[str],
            X_val: np.ndarray, y_val: np.ndarray, early_stopping: int) -> "MatchModel":
        t0 = time.perf_counter()
        self.features = list(features)
        log.info("LightGBM: %s rows x %d features | positives %s (%s) | weighted rows %s | "
                 "threads %d", fmt_int(len(y)), X.shape[1], fmt_int(y.sum()), fmt_pct(y.mean()),
                 fmt_int((w != 1).sum()), self.threads)
        log.debug("LightGBM params: %s", self.params)
        self.model = lgb.LGBMClassifier(random_state=self.seed, n_jobs=self.threads, **self.params)
        # LightGBM >= 4.7 prefers eval_X/eval_y; older versions only know eval_set
        if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
            eval_kw = {"eval_X": (X_val,), "eval_y": (y_val,)}
        else:
            eval_kw = {"eval_set": [(X_val, y_val)]}
        self.model.fit(
            X, y, sample_weight=w, feature_name=self.features, eval_names=["tune"],
            eval_metric="binary_logloss", **eval_kw,
            callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False),
                       _log_every(100)],
        )
        self.best_iteration = int(self.model.best_iteration_ or self.params.get("n_estimators", 0))
        log.info("Fit done in %.1fs | best round %d / %d", time.perf_counter() - t0,
                 self.best_iteration, self.params.get("n_estimators", 0))
        if self.best_iteration >= self.params.get("n_estimators", 0):
            log.warning("Early stopping never triggered; consider raising n_estimators")
        return self

    def predict_proba(self, X: np.ndarray, cascade: tuple[int, float] | None = None) -> np.ndarray:
        """Match probabilities. With cascade=(trees, cutoff), every row is first scored
        with the first `trees` trees; rows below `cutoff` keep that (tiny) score and
        only the rest get the full model. Validated to leave decisions unchanged."""
        if X.shape[1] != len(self.features):
            log.error("Feature count mismatch: got %d, model expects %d", X.shape[1], len(self.features))
            raise ValueError("feature mismatch")
        t0 = time.perf_counter()
        nt = self.threads if self.threads > 0 else 0
        if cascade is None or not self.best_iteration or cascade[0] >= self.best_iteration:
            p = self.model.booster_.predict(X, num_iteration=self.best_iteration, num_threads=nt)
            log.debug("  predicted %s rows in %.1fs", fmt_int(len(X)), time.perf_counter() - t0)
            return p.astype(np.float32)
        trees, cutoff = cascade
        p = self.model.booster_.predict(X, num_iteration=trees, num_threads=nt)
        hard = np.flatnonzero(p >= cutoff)
        if len(hard):
            p[hard] = self.model.booster_.predict(X[hard], num_iteration=self.best_iteration, num_threads=nt)
        log.debug("  predicted %s rows in %.1fs (cascade: %s rows got the full %d trees)",
                  fmt_int(len(X)), time.perf_counter() - t0, fmt_int(len(hard)), self.best_iteration)
        return p.astype(np.float32)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"),
                         index=self.features).sort_values(ascending=False)

    def log_importance(self, top: int = 20) -> None:
        imp = self.feature_importance()
        share = imp / max(imp.sum(), 1e-12)
        lines = [f"  {c:<22} {share[c]:6.1%}" for c in imp.index[:top]]
        log.info("Top %d features by gain:\n%s", top, "\n".join(lines))
        zero = imp[imp == 0].index.tolist()
        if zero:
            log.warning("Features never used by the model: %s", zero)

    def save(self, path: Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump({"model": self.model, "features": self.features,
                     "best_iteration": self.best_iteration, **(extra or {})}, tmp)
        tmp.replace(path)
        log.info("Saved model to %s (%.1f MB)", path.resolve(), path.stat().st_size / 2**20)

    @classmethod
    def load(cls, path: Path, threads: int = -1) -> tuple["MatchModel", dict]:
        path = Path(path)
        if not path.exists():
            log.error("Model not found at %s; run `train` first", path.resolve())
            raise FileNotFoundError(path)
        blob = joblib.load(path)
        mm = cls(params={}, threads=threads)
        mm.model, mm.features = blob.pop("model"), blob.pop("features")
        mm.best_iteration = blob.pop("best_iteration", None)
        log.info("Loaded model from %s (%d features, best round %s)", path.resolve(),
                 len(mm.features), mm.best_iteration)
        return mm, blob
