import json
import os
import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


DEFAULT_MIN_ROWS = 5000
DEFAULT_MIN_FEATURES = 8
DEFAULT_MIN_POS_RATE = 0.05
DEFAULT_MAX_POS_RATE = 0.95
DEFAULT_MAX_MISSING_RATE = 0.98
DEFAULT_DATASET_SOURCE_POLICY = "openml_only"

TARGET_LIKE_NAME_HINTS = {
    "target",
    "label",
    "class",
    "response",
    "outcome",
    "subscribed",
    "deposit",
    "default",
}


DATASET_REGISTRY = {
    "adult_income": {
        "openml_data_id": 1590,
        "openml_candidates": [
            {"data_id": 1590},
            {"name": "adult", "version": 2},
        ],
    },
    "credit_default": {
        "openml_data_id": 42477,
        "openml_candidates": [
            {"data_id": 42477},
            {"name": "default-of-credit-card-clients", "version": 1},
        ],
        "csv_fallback": "credit_default.csv",
    },
    "bank_marketing": {
        "openml_data_id": 1461,
        "openml_candidates": [
            {"data_id": 1461},
            {"name": "bank-marketing", "version": 1},
            {"name": "bank-marketing", "version": 2},
        ],
        "csv_fallback": "bank_marketing.csv",
    },
    "electricity": {
        "openml_data_id": 151,
        "openml_candidates": [
            {"data_id": 151},
            {"name": "electricity", "version": 1},
        ],
    },
    "kick": {
        "openml_data_id": 41162,
        "openml_candidates": [
            {"data_id": 41162},
            {"name": "kick", "version": 1},
        ],
    },
    "diabetes130us": {
        "openml_data_id": 45022,
        "openml_candidates": [
            {"data_id": 45022},
            {"name": "Diabetes130US", "version": 2},
        ],
        "features_waiver": True,
        "min_features": 7,
        "cat_cols_waiver": True,
    },
    "aps_failure": {
        "openml_data_id": 41138,
        "openml_candidates": [
            {"data_id": 41138},
            {"name": "APSFailure", "version": 1},
        ],
        "cat_cols_waiver": True,
        "positive_rate_waiver": True,
    },
    "diabetes_hospitals_fairlearn": {
        "openml_data_id": 43903,
        "openml_candidates": [
            {"data_id": 43903},
            {"name": "Diabetes-130-Hospitals_(Fairlearn)", "version": 1},
        ],
    },
    "telco_churn": {
        "openml_data_id": 45568,
        "openml_candidates": [
            {"data_id": 45568},
            {"name": "telco-customer-churn", "version": 2},
        ],
    },
    "airlines": {
        "openml_data_id": 42493,
        "openml_candidates": [
            {"data_id": 42493},
            {"name": "airlines", "version": 3},
        ],
        "features_waiver": True,
        "min_features": 7,
    },
    "law_school_admission": {
        "openml_data_id": 43904,
        "openml_candidates": [
            {"data_id": 43904},
            {"name": "law-school-admission-binary", "version": 3},
            {"name": "law-school-admission-bianry", "version": 3},
        ],
    },
    "compass": {
        "openml_data_id": 44162,
        "openml_candidates": [
            {"data_id": 44162},
            {"name": "compass", "version": 3},
        ],
    },
}


def _normalize_col_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def _infer_schema(X: pd.DataFrame, dataset_name: str) -> Dict[str, list]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # OpenML variants of credit-default frequently encode categorical fields as integers.
    if dataset_name == "credit_default":
        cat_hints = {
            "sex",
            "education",
            "marriage",
            "pay0",
            "pay2",
            "pay3",
            "pay4",
            "pay5",
            "pay6",
            "x2",
            "x3",
            "x4",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
        }
        hinted_cat = [c for c in X.columns if _normalize_col_name(c) in cat_hints]
        for col in hinted_cat:
            if col in num_cols:
                num_cols.remove(col)
            if col not in cat_cols:
                cat_cols.append(col)

    return {"num_cols": num_cols, "cat_cols": cat_cols}


def _binarize_labels(y: pd.Series) -> pd.Series:
    y = pd.Series(y).copy()
    if y.isna().any():
        raise ValueError("Target contains missing labels.")
    if pd.api.types.is_numeric_dtype(y):
        unique = pd.unique(y.dropna())
        if len(unique) != 2:
            raise ValueError(f"Expected binary labels, got {len(unique)} classes")
        unique_sorted = np.sort(unique)
        return (y == unique_sorted[-1]).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    unique = pd.unique(y_str.dropna())
    unique_set = set(unique.tolist())

    # Common binary coding.
    if unique_set.issubset({"0", "1"}):
        return (y_str == "1").astype(int)

    pos_values = {"yes", "true", ">50k", ">50k.", "positive", "default", "subscribed"}
    y_bin = y_str.isin(pos_values).astype(int)
    if y_bin.nunique() == 2:
        return y_bin

    if len(unique) != 2:
        raise ValueError(f"Expected binary labels, got {len(unique)} classes")
    pos = sorted(unique)[-1]
    return (y_str == pos).astype(int)


def _try_fetch_openml(candidates):
    last_err = None
    for entry in candidates:
        try:
            if isinstance(entry, dict) and "data_id" in entry:
                bunch = fetch_openml(data_id=entry["data_id"], as_frame=True)
                return bunch, f"openml_data_id:{entry['data_id']}"
            if isinstance(entry, dict) and "name" in entry:
                version = entry.get("version")
                bunch = fetch_openml(name=entry["name"], version=version, as_frame=True)
                return bunch, f"openml_name:{entry['name']}@{version}"
            bunch = fetch_openml(name=entry, as_frame=True)
            return bunch, f"openml_name:{entry}"
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        raise last_err
    raise ValueError("No OpenML candidates provided")


def _load_from_csv(path: str, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    target_candidates = {
        "credit_default": ["default", "target", "y", "label", "class"],
        "bank_marketing": ["y", "target", "label", "class", "deposit", "subscribed"],
    }
    candidates = target_candidates.get(dataset_name, ["target", "y", "label", "class"])
    target_col = None
    for col in candidates:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError(
            f"Could not find target column in {path}. Expected one of: {candidates}"
        )
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _apply_dataset_label_mapping(name: str, y: pd.Series) -> pd.Series:
    y_series = pd.Series(y)
    y_norm = y_series.astype(str).str.strip().str.lower()
    unique = set(pd.unique(y_norm.dropna()).tolist())

    if name == "bank_marketing" and unique == {"1", "2"}:
        # OpenML bank-marketing commonly uses labels {"1","2"} where "2" is the positive class.
        return (y_norm == "2").astype(int)
    if name == "electricity" and unique == {"up", "down"}:
        return (y_norm == "up").astype(int)
    if name == "compass" and unique == {"-1", "0"}:
        # OpenML compass version 3 encodes labels {-1, 0}; map 0 as the positive class.
        return (y_norm == "0").astype(int)
    return _binarize_labels(y_series)


def _detect_leakage_proxy_columns(X: pd.DataFrame, y: pd.Series) -> List[str]:
    suspicious = []
    y_int = pd.Series(y).astype(int)

    for col in X.columns:
        series = X[col]
        if series.isna().all():
            continue
        valid_mask = series.notna()
        if valid_mask.sum() < 50:
            continue

        y_sub = y_int.loc[valid_mask].to_numpy()

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.sum() >= 50:
            vals = numeric.loc[numeric_mask].to_numpy()
            uniq = np.unique(vals)
            if len(uniq) == 2 and set(np.round(uniq, 8).tolist()).issubset({0.0, 1.0}):
                pred = (vals == np.max(uniq)).astype(int)
                y_num = y_int.loc[numeric_mask].to_numpy()
                match = max(
                    float(np.mean(pred == y_num)),
                    float(np.mean((1 - pred) == y_num)),
                )
                if match >= 0.999:
                    suspicious.append(col)
                    continue

        text = series.loc[valid_mask].astype(str).str.strip().str.lower()
        uniq_text = sorted(pd.unique(text).tolist())
        if len(uniq_text) == 2:
            pred = (text == uniq_text[-1]).astype(int).to_numpy()
            match = max(
                float(np.mean(pred == y_sub)),
                float(np.mean((1 - pred) == y_sub)),
            )
            if match >= 0.999:
                suspicious.append(col)
                continue

    return sorted(set(suspicious))


def _validate_dataset(
    name: str, X: pd.DataFrame, y: pd.Series, schema: Dict[str, list], conf: Dict[str, Any]
) -> None:
    errors = []

    if len(X) == 0:
        errors.append("dataset has zero rows.")
    if X.shape[1] == 0:
        errors.append("dataset has zero features.")
    if len(X) != len(y):
        errors.append(f"row mismatch between X ({len(X)}) and y ({len(y)}).")

    y_bin = pd.Series(y).astype(int)
    y_unique = set(pd.unique(y_bin).tolist())
    if y_unique != {0, 1}:
        errors.append(f"target is not strict binary 0/1 after mapping; got {sorted(y_unique)}.")

    min_rows = int(conf.get("min_rows", DEFAULT_MIN_ROWS))
    if len(X) < min_rows and not conf.get("rows_waiver", False):
        errors.append(f"rows={len(X)} below required min_rows={min_rows}.")

    min_features = int(conf.get("min_features", DEFAULT_MIN_FEATURES))
    if X.shape[1] < min_features and not conf.get("features_waiver", False):
        errors.append(f"features={X.shape[1]} below required min_features={min_features}.")

    n_num = len(schema.get("num_cols", []))
    n_cat = len(schema.get("cat_cols", []))
    if n_num == 0 and not conf.get("num_cols_waiver", False):
        errors.append("no numerical columns after schema inference.")
    if n_cat == 0 and not conf.get("cat_cols_waiver", False):
        errors.append("no categorical columns after schema inference.")

    pos_rate = float(np.mean(y_bin))
    pos_low = float(conf.get("min_positive_rate", DEFAULT_MIN_POS_RATE))
    pos_high = float(conf.get("max_positive_rate", DEFAULT_MAX_POS_RATE))
    if (pos_rate < pos_low or pos_rate > pos_high) and not conf.get("positive_rate_waiver", False):
        errors.append(
            f"positive_rate={pos_rate:.6f} outside [{pos_low:.3f}, {pos_high:.3f}] without waiver."
        )

    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        errors.append(f"duplicate feature names found: {duplicate_cols[:5]}.")

    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        errors.append(f"all-missing feature columns found: {all_missing_cols[:5]}.")

    max_missing_rate = float(X.isna().mean().max()) if X.shape[1] > 0 else 0.0
    global_missing_rate = float(X.isna().mean().mean()) if X.shape[1] > 0 else 0.0
    missing_cap = float(conf.get("max_missing_rate", DEFAULT_MAX_MISSING_RATE))
    if global_missing_rate > missing_cap and not conf.get("missingness_waiver", False):
        errors.append(
            f"overall_missing_rate={global_missing_rate:.6f} exceeds cap={missing_cap:.3f}."
        )

    allowed_target_like = {
        _normalize_col_name(c) for c in conf.get("allowed_target_like_feature_names", [])
    }
    target_like = [
        c
        for c in X.columns
        if _normalize_col_name(c) in TARGET_LIKE_NAME_HINTS
        and _normalize_col_name(c) not in allowed_target_like
    ]
    if target_like:
        errors.append(f"target-like feature names detected: {target_like[:5]}.")

    leakage_proxy_cols = _detect_leakage_proxy_columns(X, y_bin)
    if leakage_proxy_cols:
        errors.append(f"possible direct target proxies detected: {leakage_proxy_cols[:5]}.")

    if errors:
        msg = "\n- ".join(errors)
        raise ValueError(f"Dataset validation failed for '{name}':\n- {msg}")


def _dataset_profile(
    X: pd.DataFrame,
    y: pd.Series,
    schema: Dict[str, list],
    source: str,
    conf: Dict[str, Any],
    source_policy: str,
    source_sha256: str | None = None,
) -> Dict[str, Any]:
    n_constant = int((X.nunique(dropna=False) <= 1).sum())
    per_col_missing = X.isna().mean()
    source_sha_value = str(source_sha256 or "").strip()
    if not source_sha_value:
        if str(source).startswith("openml"):
            source_sha_value = "N/A (OpenML)"
        else:
            source_sha_value = "N/A"
    return {
        "source": source,
        "source_policy": source_policy,
        "source_sha256": source_sha_value,
        "openml_data_id": conf.get("openml_data_id"),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_num_cols": int(len(schema["num_cols"])),
        "n_cat_cols": int(len(schema["cat_cols"])),
        "positive_rate": float(np.mean(pd.Series(y).astype(int))),
        "overall_missing_rate": float(per_col_missing.mean()) if len(per_col_missing) else 0.0,
        "max_col_missing_rate": float(per_col_missing.max()) if len(per_col_missing) else 0.0,
        "n_constant_cols": n_constant,
    }


def load_dataset(
    name: str, data_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, list], Dict[str, Any]]:
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    conf = DATASET_REGISTRY[name]
    candidates = conf.get("openml_candidates", [])
    source_policy = os.getenv("DATASET_SOURCE_POLICY", DEFAULT_DATASET_SOURCE_POLICY).strip().lower()
    if source_policy not in {"openml_only", "openml_or_csv"}:
        raise ValueError(
            "Unknown DATASET_SOURCE_POLICY={!r}. Expected one of: 'openml_only', 'openml_or_csv'.".format(
                source_policy
            )
        )
    source_sha256 = None
    try:
        bunch, source = _try_fetch_openml(candidates)
        X = bunch.data
        y = bunch.target
    except Exception as exc:
        if source_policy != "openml_or_csv":
            raise RuntimeError(
                "Failed to fetch dataset '{}' from pinned OpenML sources while DATASET_SOURCE_POLICY='openml_only'. "
                "Set DATASET_SOURCE_POLICY='openml_or_csv' to allow documented CSV fallback."
                .format(name)
            ) from exc
        csv_fallback = conf.get("csv_fallback")
        if not csv_fallback:
            raise
        csv_path = f"{data_dir}/{csv_fallback}"
        X, y = _load_from_csv(csv_path, name)
        source = f"csv:{csv_path}"
        source_sha256 = _file_sha256(csv_path)

    y = _apply_dataset_label_mapping(name, pd.Series(y))
    X = pd.DataFrame(X)
    schema = _infer_schema(X, dataset_name=name)
    _validate_dataset(name, X, y, schema, conf)
    profile = _dataset_profile(
        X,
        y,
        schema,
        source=source,
        conf=conf,
        source_policy=source_policy,
        source_sha256=source_sha256,
    )
    return X, y, schema, profile


def dataset_description() -> Dict[str, str]:
    return {
        "credit_default": json.dumps(
            {
                "target": "default (0/1)",
                "notes": "Use one of target/default/y/label/class as target column if providing CSV.",
            }
        ),
        "bank_marketing": json.dumps(
            {
                "target": "y (yes/no)",
                "notes": "Use one of y/target/label/class/deposit/subscribed as target column if providing CSV.",
            }
        ),
        "adult_income": json.dumps(
            {
                "target": "class (>50K/<=50K)",
                "notes": "Fetched via OpenML.",
            }
        ),
        "electricity": json.dumps(
            {
                "target": "class (down/up)",
                "notes": "OpenML data_id 151; mapped as positive=up.",
            }
        ),
        "kick": json.dumps(
            {
                "target": "binary 0/1",
                "notes": "OpenML data_id 41162.",
            }
        ),
        "diabetes130us": json.dumps(
            {
                "target": "binary 0/1",
                "notes": "OpenML data_id 45022.",
            }
        ),
        "aps_failure": json.dumps(
            {
                "target": "neg/pos",
                "notes": "OpenML data_id 41138.",
            }
        ),
        "diabetes_hospitals_fairlearn": json.dumps(
            {
                "target": "binary 0/1",
                "notes": "OpenML data_id 43903.",
            }
        ),
        "telco_churn": json.dumps(
            {
                "target": "churn (yes/no)",
                "notes": "OpenML data_id 45568.",
            }
        ),
        "airlines": json.dumps(
            {
                "target": "delay (0/1)",
                "notes": "OpenML data_id 42493.",
            }
        ),
        "law_school_admission": json.dumps(
            {
                "target": "admission (0/1)",
                "notes": "OpenML data_id 43904.",
            }
        ),
        "compass": json.dumps(
            {
                "target": "binary (-1/0)",
                "notes": "OpenML data_id 44162; mapped as positive=0.",
            }
        ),
    }
