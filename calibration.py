import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_1d_proba(proba):
    proba = np.asarray(proba)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


def _logits_from_proba(proba):
    p = np.clip(_to_1d_proba(proba), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _clip_proba(proba):
    return np.clip(_to_1d_proba(proba).astype(float), 1e-6, 1 - 1e-6)


def temperature_scale_fit(proba_val, y_val) -> float:
    y_val = np.asarray(y_val).astype(float)
    logits = _logits_from_proba(proba_val)

    def nll(log_t):
        t = np.exp(log_t[0])
        scaled = logits / t
        p = 1 / (1 + np.exp(-scaled))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    res = minimize(nll, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(-5, 5)])
    t = float(np.exp(res.x[0]))
    return t


def temperature_scale_predict(proba, T: float):
    logits = _logits_from_proba(proba)
    scaled = logits / max(T, 1e-6)
    p = 1 / (1 + np.exp(-scaled))
    return p


def platt_fit(proba_val, y_val):
    y_val = np.asarray(y_val).astype(int)
    logits = _logits_from_proba(proba_val).reshape(-1, 1)
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(logits, y_val)
    return model


def platt_predict(proba, model):
    logits = _logits_from_proba(proba).reshape(-1, 1)
    p = model.predict_proba(logits)[:, 1]
    return _clip_proba(p)


def isotonic_fit(proba_val, y_val):
    y_val = np.asarray(y_val).astype(int)
    p_val = _clip_proba(proba_val)
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(p_val, y_val)
    return model


def isotonic_predict(proba, model):
    p = _clip_proba(proba)
    pred = model.predict(p)
    return _clip_proba(pred)


def beta_calibration_fit(proba_val, y_val):
    y_val = np.asarray(y_val).astype(int)
    p = _clip_proba(proba_val)
    # Beta calibration features (Kull et al.): log(p), log(1-p).
    X = np.column_stack([np.log(p), np.log(1 - p)])
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(X, y_val)
    return model


def beta_calibration_predict(proba, model):
    p = _clip_proba(proba)
    X = np.column_stack([np.log(p), np.log(1 - p)])
    pred = model.predict_proba(X)[:, 1]
    return _clip_proba(pred)
