from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def _to_1d_proba(proba) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 1]
    return np.clip(arr, 1e-6, 1 - 1e-6)


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    proba = _to_1d_proba(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        if i == 0:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba > lower) & (proba <= upper)
        if not np.any(mask):
            continue
        bin_conf = proba[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def adaptive_expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    proba = _to_1d_proba(proba)
    n = len(y_true)
    if n == 0:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(proba, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    ece = 0.0
    for i in range(len(edges) - 1):
        lower = edges[i]
        upper = edges[i + 1]
        if i == 0:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba > lower) & (proba <= upper)
        if not np.any(mask):
            continue
        bin_conf = proba[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_gap_summary(y_true, proba, n_bins: int = 15) -> Tuple[float, float, int]:
    y_true = np.asarray(y_true, dtype=float)
    proba = _to_1d_proba(proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []

    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        if i == 0:
            mask = (proba >= lower) & (proba <= upper)
        else:
            mask = (proba > lower) & (proba <= upper)
        if not np.any(mask):
            continue
        bin_conf = proba[mask].mean()
        bin_acc = y_true[mask].mean()
        gaps.append(abs(float(bin_conf - bin_acc)))

    if not gaps:
        return (0.0, 0.0, 0)
    return (float(np.mean(gaps)), float(np.max(gaps)), int(len(gaps)))


def calibration_slope_intercept(y_true, proba) -> Tuple[float, float]:
    y = np.asarray(y_true, dtype=float)
    p = _to_1d_proba(proba)
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    logits = np.log(p / (1 - p))

    def nll(params):
        intercept, slope = params
        z = intercept + slope * logits
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    res = minimize(
        nll,
        x0=np.array([0.0, 1.0], dtype=float),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-10.0, 10.0)],
    )
    if not res.success:
        return (float("nan"), float("nan"))
    intercept, slope = res.x
    return (float(slope), float(intercept))


def compute_calibration_diagnostics(y_true, proba) -> Dict[str, float]:
    slope, intercept = calibration_slope_intercept(y_true, proba)
    rel_gap_mean, rel_gap_max, rel_nonempty_bins = reliability_gap_summary(y_true, proba)
    return {
        "calib_slope": slope,
        "calib_intercept": intercept,
        "ece_adaptive": adaptive_expected_calibration_error(y_true, proba, n_bins=15),
        "rel_gap_mean": rel_gap_mean,
        "rel_gap_max": rel_gap_max,
        "rel_nonempty_bins": float(rel_nonempty_bins),
    }


def compute_metrics(y_true, proba) -> Dict[str, float]:
    proba_1d = _to_1d_proba(proba)
    try:
        auc = roc_auc_score(y_true, proba_1d)
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, proba_1d)
    except ValueError:
        pr_auc = float("nan")
    preds = (proba_1d >= 0.5).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, preds)
    brier = brier_score_loss(y_true, proba_1d, pos_label=1)
    ll = log_loss(y_true, np.clip(proba_1d, 1e-15, 1 - 1e-15), labels=[0, 1])
    ece = expected_calibration_error(y_true, proba_1d, n_bins=15)
    return {
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "brier": float(brier),
        "logloss": float(ll),
        "ece": float(ece),
    }
