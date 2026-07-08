"""
Evaluation utilities: regression metrics, split-conformal prediction,
applicability domain analysis, classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix,
    hamming_loss, jaccard_score,
)


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        name: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {"name": name, "R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "N": int(len(y_true))}


# ── Split-conformal prediction intervals ──────────────────────────────────────

def conformal_quantile(cal_residuals: np.ndarray, alpha: float) -> float:
    n = len(cal_residuals)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_residuals, q_level))


def conformal_intervals(cal_y: np.ndarray, cal_pred: np.ndarray,
                        test_pred: np.ndarray,
                        alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.abs(np.asarray(cal_y) - np.asarray(cal_pred))
    q = conformal_quantile(residuals, alpha)
    lo = np.asarray(test_pred) - q
    hi = np.asarray(test_pred) + q
    return lo, hi


def empirical_coverage(y_test: np.ndarray, lo: np.ndarray,
                        hi: np.ndarray) -> float:
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


def coverage_table(cal_y: np.ndarray, cal_pred: np.ndarray,
                   test_y: np.ndarray, test_pred: np.ndarray) -> pd.DataFrame:
    alphas = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    rows = []
    for alpha in alphas:
        lo, hi = conformal_intervals(cal_y, cal_pred, test_pred, alpha)
        emp = empirical_coverage(test_y, lo, hi)
        width = float(np.mean(hi - lo))
        rows.append({
            "Nominal (%)":  round((1 - alpha) * 100, 0),
            "Empirical (%)": round(emp * 100, 1),
            "Δ (pp)":        round((emp - (1 - alpha)) * 100, 1),
            "Mean Width":    round(width, 3),
        })
    return pd.DataFrame(rows)


# ── Applicability Domain ──────────────────────────────────────────────────────

class ApplicabilityDomain:
    """
    Tanimoto nearest-neighbour applicability domain.
    AD threshold = 5th percentile of max-Tanimoto within the training set.
    """

    def __init__(self, percentile: float = 5.0):
        self.percentile = percentile
        self.threshold_: float | None = None
        self._train_fps: np.ndarray | None = None

    def fit(self, train_fps: np.ndarray) -> "ApplicabilityDomain":
        self._train_fps = train_fps.astype(np.float32)
        n = len(train_fps)
        sample_idx = (np.random.default_rng(42).choice(n, min(n, 2000), replace=False)
                      if n > 2000 else np.arange(n))
        sims = self._max_tanimoto(self._train_fps[sample_idx], self._train_fps,
                                  exclude_self=True)
        self.threshold_ = float(np.percentile(sims, self.percentile))
        return self

    def predict(self, query_fps: np.ndarray) -> np.ndarray:
        sims = self.similarity_scores(query_fps)
        return sims >= self.threshold_

    def similarity_scores(self, query_fps: np.ndarray) -> np.ndarray:
        return self._max_tanimoto(query_fps.astype(np.float32), self._train_fps)

    @staticmethod
    def _max_tanimoto(query: np.ndarray, ref: np.ndarray,
                      exclude_self: bool = False,
                      batch_size: int = 512) -> np.ndarray:
        # Compute in batches to avoid OOM for large datasets
        nq = query.shape[0]
        max_sims = np.zeros(nq, dtype=np.float32)
        sum_ref = np.sum(ref, axis=1)

        for start in range(0, nq, batch_size):
            end = min(start + batch_size, nq)
            qb = query[start:end]
            dot = qb @ ref.T
            sq  = np.sum(qb, axis=1, keepdims=True)
            union = sq + sum_ref[np.newaxis, :] - dot
            union = np.where(union == 0, 1e-9, union)
            tan = dot / union
            if exclude_self:
                for k in range(end - start):
                    global_k = start + k
                    if global_k < tan.shape[1]:
                        tan[k, global_k] = -1.0
            max_sims[start:end] = np.max(tan, axis=1)
        return max_sims


# ── Classification metrics ────────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            name: str = "") -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"name": name, "Accuracy": round(acc, 4), "F1_macro": round(f1, 4)}


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        name: str = "") -> dict:
    hl  = hamming_loss(y_true, y_pred)
    jac = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    exact = float(np.mean(np.all(y_true == y_pred, axis=1)))
    return {
        "name": name,
        "Exact_Match": round(exact, 4),
        "Hamming_Loss": round(float(hl), 4),
        "Jaccard": round(float(jac), 4),
    }
