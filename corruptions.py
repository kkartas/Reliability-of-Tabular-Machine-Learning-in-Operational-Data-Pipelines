from typing import Dict, List, Tuple

import hashlib
import numpy as np
import pandas as pd


def _rng(seed: int):
    return np.random.default_rng(seed)


def _common_diag(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    n_rows_affected: int,
    n_cols_affected: int,
    n_cells_targeted: int,
) -> Dict[str, float]:
    n_rows_before = int(len(X_before))
    n_rows_after = int(len(X_after))
    n_cols_total = int(X_before.shape[1])
    denom_cells = max(n_rows_after * n_cols_total, 1)
    return {
        "n_rows_before": n_rows_before,
        "n_rows_after": n_rows_after,
        "n_rows_affected": int(n_rows_affected),
        "row_impact_rate": float(n_rows_affected / max(n_rows_before, 1)),
        "n_cols_total": n_cols_total,
        "n_cols_affected": int(n_cols_affected),
        "col_impact_rate": float(n_cols_affected / max(n_cols_total, 1)),
        "n_cells_targeted": int(n_cells_targeted),
        "cell_impact_rate": float(n_cells_targeted / denom_cells),
    }


def _anchored_rows(
    X: pd.DataFrame, k: int, rng: np.random.Generator, cat_cols: List[str]
) -> Tuple[np.ndarray, str, str, str]:
    if k <= 0:
        return np.array([], dtype=int), "none", "", ""

    if cat_cols:
        anchor_col = str(rng.choice(cat_cols))
        series = X[anchor_col]
        counts = series.astype(str).value_counts(dropna=True)
        if len(counts) > 0:
            anchor_val = str(counts.index[0])
            anchor_rows = X.index[series.astype(str) == anchor_val].to_numpy()
            if len(anchor_rows) >= k:
                row_idx = rng.permutation(anchor_rows)[:k]
                return row_idx, "categorical_anchor", anchor_col, anchor_val

            selected = anchor_rows.tolist()
            remaining = X.index.difference(pd.Index(selected)).to_numpy()
            need = min(k - len(selected), len(remaining))
            if need > 0:
                extra = rng.permutation(remaining)[:need].tolist()
                selected.extend(extra)
            row_idx = np.asarray(selected[:k])
            return row_idx, "categorical_anchor_plus_fill", anchor_col, anchor_val

    row_idx = rng.permutation(X.index.to_numpy())[: min(k, len(X))]
    return row_idx, "uniform_fallback", "", ""


def apply_c1_duplication_with_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    seed: int,
    anchor_mode: str = "label_informed",
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    X_out = X.copy()
    y_out = y.copy()
    n = len(X_out)
    k = int(round(alpha * n))
    if k <= 0:
        diag = _common_diag(X_out, X_out, 0, 0, 0)
        diag.update(
            {
                "c1_anchor_mode": anchor_mode,
                "c1_strategy": "none",
                "c1_anchor_col": "",
                "c1_anchor_value": "",
                "c1_cohort_size": 0,
                "c1_cohort_positive_rate": float("nan"),
                "c1_overall_positive_rate": float(np.mean(y_out)),
            }
        )
        return X_out, y_out, diag

    rng = _rng(seed)
    cat_cols = X_out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    overall_pos = float(np.mean(y_out))

    if anchor_mode not in {"label_informed", "label_agnostic"}:
        raise ValueError("anchor_mode must be one of: 'label_informed', 'label_agnostic'.")

    if cat_cols:
        anchor_col = cat_cols[seed % len(cat_cols)]
        series = X_out[anchor_col].astype(str)
        if anchor_mode == "label_informed":
            grp = pd.DataFrame({"key": series, "y": y_out.astype(float).to_numpy()})
            stats = grp.groupby("key", observed=False)["y"].agg(["mean", "count"]).reset_index()
            min_support = max(20, int(round(0.005 * n)))
            stats = stats[stats["count"] >= min_support].copy()
            if stats.empty:
                counts = series.value_counts(dropna=True)
                if len(counts) > 0:
                    anchor_val = str(counts.index[0])
                else:
                    anchor_val = str(series.iloc[0])
                strategy = "categorical_largest_group"
            else:
                stats["deviation"] = (stats["mean"] - overall_pos).abs()
                stats = stats.sort_values(["deviation", "count"], ascending=[False, False])
                anchor_val = str(stats.iloc[0]["key"])
                strategy = "categorical_target_skew"
        else:
            counts = series.value_counts(dropna=True)
            if len(counts) > 0:
                anchor_val = str(counts.index[0])
            else:
                anchor_val = str(series.iloc[0])
            strategy = "categorical_largest_group_label_agnostic"

        cohort_idx = X_out.index[series == anchor_val].to_numpy()
        if len(cohort_idx) == 0:
            cohort_idx = X_out.index.to_numpy()
            strategy = "uniform_fallback"
        anchor_col_name = anchor_col
        anchor_value = anchor_val
    else:
        num_cols = X_out.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            anchor_col_name = num_cols[0]
            vals = pd.to_numeric(X_out[anchor_col_name], errors="coerce")
            q = float(vals.quantile(0.75))
            cohort_idx = X_out.index[vals >= q].to_numpy()
            if len(cohort_idx) == 0:
                cohort_idx = X_out.index.to_numpy()
            anchor_value = f"q75_plus({q:.6f})"
            strategy = "numeric_q75"
        else:
            cohort_idx = X_out.index.to_numpy()
            anchor_col_name = ""
            anchor_value = ""
            strategy = "uniform_fallback"

    replace = len(cohort_idx) < k
    dup_idx = rng.choice(cohort_idx, size=k, replace=replace)
    X_dup = pd.concat([X_out, X_out.loc[dup_idx].copy()], ignore_index=True)
    y_dup = pd.concat([y_out, y_out.loc[dup_idx].copy()], ignore_index=True)

    cohort_y = y_out.loc[cohort_idx] if len(cohort_idx) > 0 else y_out
    diag = _common_diag(X_out, X_dup, k, 0, 0)
    diag.update(
        {
            "c1_anchor_mode": anchor_mode,
            "c1_strategy": strategy,
            "c1_anchor_col": anchor_col_name,
            "c1_anchor_value": str(anchor_value),
            "c1_cohort_size": int(len(cohort_idx)),
            "c1_cohort_positive_rate": float(np.mean(cohort_y)),
            "c1_overall_positive_rate": overall_pos,
        }
    )
    return X_dup, y_dup, diag


def apply_c2_missingness_with_diagnostics(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> Tuple[pd.DataFrame, Dict]:
    X_out = X.copy()
    n = len(X_out)
    k = int(round(alpha * n))
    if k <= 0:
        diag = _common_diag(X_out, X_out, 0, 0, 0)
        diag.update(
            {
                "c2_anchor_strategy": "none",
                "c2_anchor_col": "",
                "c2_anchor_value": "",
                "c2_new_missing_cells": 0,
                "c2_new_missing_rate_targeted": 0.0,
                "c2_masked_column_fraction_target": 0.0,
            }
        )
        return X_out, diag

    rng = _rng(seed)
    cols = [c for c in (schema["num_cols"] + schema["cat_cols"]) if c in X_out.columns]
    if not cols:
        diag = _common_diag(X_out, X_out, 0, 0, 0)
        diag.update(
            {
                "c2_anchor_strategy": "none",
                "c2_anchor_col": "",
                "c2_anchor_value": "",
                "c2_new_missing_cells": 0,
                "c2_new_missing_rate_targeted": 0.0,
                "c2_masked_column_fraction_target": 0.0,
            }
        )
        return X_out, diag

    col_frac = float(min(1.0, max(alpha, 1.0 / len(cols))))
    m = max(1, int(round(col_frac * len(cols))))
    cols_to_mask = [str(c) for c in rng.choice(cols, size=m, replace=False).tolist()]
    row_idx, anchor_strategy, anchor_col, anchor_val = _anchored_rows(
        X_out, k, rng, [c for c in schema["cat_cols"] if c in X_out.columns]
    )
    row_idx = np.asarray(row_idx)

    if len(row_idx) == 0:
        diag = _common_diag(X_out, X_out, 0, len(cols_to_mask), 0)
        diag.update(
            {
                "c2_anchor_strategy": anchor_strategy,
                "c2_anchor_col": anchor_col,
                "c2_anchor_value": anchor_val,
                "c2_new_missing_cells": 0,
                "c2_new_missing_rate_targeted": 0.0,
                "c2_masked_column_fraction_target": float(m / len(cols)),
            }
        )
        return X_out, diag

    before_na = X_out.loc[row_idx, cols_to_mask].isna().to_numpy()
    X_out.loc[row_idx, cols_to_mask] = np.nan
    after_na = X_out.loc[row_idx, cols_to_mask].isna().to_numpy()
    new_missing = int(np.logical_and(~before_na, after_na).sum())
    targeted_cells = int(len(row_idx) * len(cols_to_mask))

    diag = _common_diag(X, X_out, len(row_idx), len(cols_to_mask), targeted_cells)
    diag.update(
        {
            "c2_anchor_strategy": anchor_strategy,
            "c2_anchor_col": anchor_col,
            "c2_anchor_value": anchor_val,
            "c2_new_missing_cells": new_missing,
            "c2_new_missing_rate_targeted": float(new_missing / max(targeted_cells, 1)),
            "c2_masked_column_fraction_target": float(m / len(cols)),
        }
    )
    return X_out, diag


def _build_c3_token_pool(
    col: str, alpha: float, seed: int, rng: np.random.Generator
) -> Tuple[List[str], List[str]]:
    families = ["new", "legacy", "vendor", "hash"]
    n_families = int(min(len(families), max(2, int(round(2 + 2 * alpha)))))
    chosen = rng.choice(families, size=n_families, replace=False).tolist()
    token_count = max(2, int(round(2 + 6 * alpha)))

    tokens = []
    for i in range(token_count):
        fam = chosen[i % n_families]
        if fam == "new":
            token = f"NEW_{col}_{seed % 1000}_{i}"
        elif fam == "legacy":
            token = f"LEGACY_{rng.integers(100, 999)}_{i}"
        elif fam == "vendor":
            token = f"SRC{rng.integers(1, 9)}::{str(col).upper()}::{i}"
        else:
            key = f"{col}|{seed}|{i}".encode("utf-8")
            digest = hashlib.sha256(key).hexdigest()
            token = f"UNK_{int(digest[:10], 16) % 100000}"
        tokens.append(token)
    return tokens, chosen


def apply_c3_categorical_drift_with_diagnostics(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> Tuple[pd.DataFrame, Dict]:
    X_out = X.copy()
    n = len(X_out)
    k = int(round(alpha * n))
    cat_cols = [c for c in schema["cat_cols"] if c in X_out.columns]
    if k <= 0 or not cat_cols:
        diag = _common_diag(X_out, X_out, 0, 0, 0)
        diag.update(
            {
                "c3_pattern_families_used": "",
                "c3_total_unique_tokens": 0,
                "c3_mean_tokens_per_column": 0.0,
            }
        )
        return X_out, diag

    rng = _rng(seed)
    row_idx = X_out.index[rng.choice(n, size=k, replace=False)]
    col_frac = float(min(1.0, max(alpha, 1.0 / len(cat_cols))))
    m = max(1, int(round(col_frac * len(cat_cols))))
    cols_to_drift = [str(c) for c in rng.choice(cat_cols, size=m, replace=False).tolist()]

    used_families: List[str] = []
    total_tokens = 0
    inserted_values = set()

    for col in cols_to_drift:
        if not pd.api.types.is_object_dtype(X_out[col]):
            X_out[col] = X_out[col].astype("object")
        token_pool, fams = _build_c3_token_pool(col, alpha, seed, rng)
        used_families.extend(fams)
        total_tokens += len(token_pool)
        assigned = rng.choice(np.asarray(token_pool, dtype=object), size=len(row_idx), replace=True)
        X_out.loc[row_idx, col] = assigned
        inserted_values.update(assigned.tolist())

    targeted_cells = int(len(row_idx) * len(cols_to_drift))
    diag = _common_diag(X, X_out, len(row_idx), len(cols_to_drift), targeted_cells)
    diag.update(
        {
            "c3_pattern_families_used": "|".join(sorted(set(used_families))),
            "c3_total_unique_tokens": int(len(inserted_values)),
            "c3_mean_tokens_per_column": float(total_tokens / max(len(cols_to_drift), 1)),
        }
    )
    return X_out, diag


def apply_c4_measurement_with_diagnostics(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> Tuple[pd.DataFrame, Dict]:
    X_out = X.copy()
    n = len(X_out)
    k = int(round(alpha * n))
    num_cols = [c for c in schema["num_cols"] if c in X_out.columns]
    if k <= 0 or not num_cols:
        diag = _common_diag(X_out, X_out, 0, 0, 0)
        diag.update(
            {
                "c4_cols_scale_shift": 0,
                "c4_cols_shift_only": 0,
                "c4_cols_rounding": 0,
                "c4_cols_clipping": 0,
                "c4_cols_scale_shift_rate": 0.0,
                "c4_cols_shift_only_rate": 0.0,
                "c4_cols_rounding_rate": 0.0,
                "c4_cols_clipping_rate": 0.0,
                "c4_mean_abs_delta": 0.0,
                "c4_median_abs_delta": 0.0,
                "c4_max_abs_delta": 0.0,
            }
        )
        return X_out, diag

    rng = _rng(seed)
    row_idx = X_out.index[rng.choice(n, size=k, replace=False)]
    col_frac = float(min(1.0, max(0.25 + 0.75 * alpha, 1.0 / len(num_cols))))
    m = max(1, int(round(col_frac * len(num_cols))))
    cols_to_corrupt = [str(c) for c in rng.choice(num_cols, size=m, replace=False).tolist()]

    variants = ["scale_shift", "shift_only", "rounding", "clipping"]
    variant_counts = {v: 0 for v in variants}
    abs_deltas: List[float] = []
    abs_deltas_std_units: List[float] = []

    for col in cols_to_corrupt:
        col_vals = pd.to_numeric(X_out[col], errors="coerce").astype(float)
        target_vals = col_vals.loc[row_idx].copy()
        valid = target_vals.notna()
        if not valid.any():
            continue

        old = target_vals.loc[valid].to_numpy(copy=True)
        std = float(np.nanstd(col_vals.to_numpy()))
        if not np.isfinite(std) or std <= 0:
            std = 1.0

        variant = str(rng.choice(variants, p=np.array([0.35, 0.25, 0.20, 0.20])))
        variant_counts[variant] += 1

        if variant == "scale_shift":
            scale = rng.normal(loc=1.0, scale=0.05 + 0.25 * alpha, size=len(old))
            shift = rng.normal(loc=0.0, scale=(0.02 + 0.20 * alpha) * std, size=len(old))
            new = scale * old + shift
        elif variant == "shift_only":
            shift = rng.normal(loc=0.0, scale=(0.05 + 0.30 * alpha) * std, size=len(old))
            new = old + shift
        elif variant == "rounding":
            step = max(std * (0.01 + 0.20 * alpha), 1e-6)
            new = np.round(old / step) * step
        else:
            q = float(min(0.49, 0.01 + 0.30 * alpha))
            low = float(np.nanquantile(col_vals.to_numpy(), q))
            high = float(np.nanquantile(col_vals.to_numpy(), 1.0 - q))
            new = np.clip(old, low, high)

        # Guardrail against unrealistic numeric explosions on heavy-tailed features.
        cap = (0.25 + 2.00 * alpha) * std
        new = np.clip(new, old - cap, old + cap)

        target_vals.loc[valid] = new
        col_vals.loc[row_idx] = target_vals
        X_out[col] = col_vals
        delta_abs = np.abs(new - old)
        abs_deltas.extend(delta_abs.tolist())
        abs_deltas_std_units.extend((delta_abs / max(std, 1e-9)).tolist())

    targeted_cells = int(len(row_idx) * len(cols_to_corrupt))
    denom_cols = max(len(cols_to_corrupt), 1)
    diag = _common_diag(X, X_out, len(row_idx), len(cols_to_corrupt), targeted_cells)
    diag.update(
        {
            "c4_cols_scale_shift": int(variant_counts["scale_shift"]),
            "c4_cols_shift_only": int(variant_counts["shift_only"]),
            "c4_cols_rounding": int(variant_counts["rounding"]),
            "c4_cols_clipping": int(variant_counts["clipping"]),
            "c4_cols_scale_shift_rate": float(variant_counts["scale_shift"] / denom_cols),
            "c4_cols_shift_only_rate": float(variant_counts["shift_only"] / denom_cols),
            "c4_cols_rounding_rate": float(variant_counts["rounding"] / denom_cols),
            "c4_cols_clipping_rate": float(variant_counts["clipping"] / denom_cols),
            "c4_mean_abs_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
            "c4_median_abs_delta": float(np.median(abs_deltas)) if abs_deltas else 0.0,
            "c4_max_abs_delta": float(np.max(abs_deltas)) if abs_deltas else 0.0,
            "c4_mean_abs_delta_std_units": (
                float(np.mean(abs_deltas_std_units)) if abs_deltas_std_units else 0.0
            ),
            "c4_median_abs_delta_std_units": (
                float(np.median(abs_deltas_std_units)) if abs_deltas_std_units else 0.0
            ),
        }
    )
    return X_out, diag


def apply_c2_c4_compound_with_diagnostics(
    X: pd.DataFrame, schema: dict, alpha_c2: float, alpha_c4: float, seed: int
) -> Tuple[pd.DataFrame, Dict]:
    X_mid, d2 = apply_c2_missingness_with_diagnostics(X, schema, alpha_c2, seed)
    X_out, d4 = apply_c4_measurement_with_diagnostics(X_mid, schema, alpha_c4, seed + 10000)

    # Use union-based row/column impact over the final composed transformation.
    X_before_obj = X.astype("object")
    X_after_obj = X_out.astype("object")
    equal_mask = X_before_obj.eq(X_after_obj) | (X_before_obj.isna() & X_after_obj.isna())
    changed_mask = ~equal_mask
    n_rows_union = int(changed_mask.any(axis=1).sum())
    n_cols_union = int(changed_mask.any(axis=0).sum())

    diag = _common_diag(
        X_before=X,
        X_after=X_out,
        n_rows_affected=n_rows_union,
        n_cols_affected=n_cols_union,
        n_cells_targeted=int(d2.get("n_cells_targeted", 0)) + int(d4.get("n_cells_targeted", 0)),
    )
    for key, value in d2.items():
        if key.startswith("c2_"):
            diag[key] = value
    for key, value in d4.items():
        if key.startswith("c4_"):
            diag[key] = value
    diag["compound_alpha_c2"] = float(alpha_c2)
    diag["compound_alpha_c4"] = float(alpha_c4)
    return X_out, diag


def apply_c1_duplication(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    seed: int,
    anchor_mode: str = "label_informed",
) -> Tuple[pd.DataFrame, pd.Series]:
    X_out, y_out, _ = apply_c1_duplication_with_diagnostics(
        X,
        y,
        alpha,
        seed,
        anchor_mode=anchor_mode,
    )
    return X_out, y_out


def apply_c2_missingness(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> pd.DataFrame:
    X_out, _ = apply_c2_missingness_with_diagnostics(X, schema, alpha, seed)
    return X_out


def apply_c3_categorical_drift(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> pd.DataFrame:
    X_out, _ = apply_c3_categorical_drift_with_diagnostics(X, schema, alpha, seed)
    return X_out


def apply_c4_measurement(
    X: pd.DataFrame, schema: dict, alpha: float, seed: int
) -> pd.DataFrame:
    X_out, _ = apply_c4_measurement_with_diagnostics(X, schema, alpha, seed)
    return X_out


def apply_c2_c4_compound(
    X: pd.DataFrame, schema: dict, alpha_c2: float, alpha_c4: float, seed: int
) -> pd.DataFrame:
    X_out, _ = apply_c2_c4_compound_with_diagnostics(X, schema, alpha_c2, alpha_c4, seed)
    return X_out
