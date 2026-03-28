import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as t_dist
from scipy.stats import ttest_1samp
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from calibration import (
    beta_calibration_fit,
    beta_calibration_predict,
    isotonic_fit,
    isotonic_predict,
    platt_fit,
    platt_predict,
    temperature_scale_fit,
    temperature_scale_predict,
)
from corruptions import (
    apply_c1_duplication_with_diagnostics,
    apply_c2_c4_compound_with_diagnostics,
    apply_c2_missingness_with_diagnostics,
    apply_c3_categorical_drift_with_diagnostics,
    apply_c4_measurement_with_diagnostics,
)
from datasets import load_dataset
from metrics import compute_calibration_diagnostics, compute_metrics
from models import get_models
from plotting import generate_all_plots
from preprocess import build_preprocessor


SEEDS = list(range(20))
SINGLE_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.40]
C2_LEVELS = [0.0, 0.10, 0.20, 0.40]
C4_LEVELS = [0.0, 0.20, 0.40]
ENCODINGS = ["ignore_unknown", "unknown_bucket"]
CORE_DATASETS = ["credit_default", "bank_marketing", "adult_income"]
EXTENDED_DATASETS = CORE_DATASETS + [
    "electricity",
    "kick",
    "diabetes130us",
    "aps_failure",
    "diabetes_hospitals_fairlearn",
    "telco_churn",
    "airlines",
    "law_school_admission",
    "compass",
]
SINGLE_CORRUPTIONS = ["C1", "C2", "C3", "C4"]
PRIMARY_METRICS = ["auc", "f1", "ece"]
ALL_METRICS = ["auc", "pr_auc", "f1", "bal_acc", "brier", "logloss", "ece"]
CALIBRATION_DIAGNOSTIC_METRICS = [
    "calib_slope",
    "calib_intercept",
    "ece_adaptive",
    "rel_gap_mean",
    "rel_gap_max",
    "rel_nonempty_bins",
]
CALIBRATION_METRICS = ALL_METRICS + CALIBRATION_DIAGNOSTIC_METRICS

DEFAULT_PUBLIC_REPO_URL = "https://github.com/PLACEHOLDER-ORG/mlpaper-reliability"
DEFAULT_PUBLIC_RELEASE_TAG = "v0.0.0-submission"
DEFAULT_PUBLIC_RELEASE_COMMIT = "PLACEHOLDER-COMMIT-SHA"
DEFAULT_PUBLIC_ARCHIVE_DOI = "10.5281/zenodo.00000000"


def _file_sha256(path: str) -> str:
    import hashlib

    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _code_snapshot_candidates() -> List[Path]:
    root = Path(".").resolve()
    # Restrict the immutable code snapshot to the files that define the
    # canonical rerun and manuscript build. This avoids pinning local planning
    # notes or stale template files that are not part of the submission artifact.
    rel_paths = [
        "README.md",
        "requirements.txt",
        "requirements-lock.txt",
        "run_experiments.py",
        "datasets.py",
        "preprocess.py",
        "corruptions.py",
        "models.py",
        "metrics.py",
        "calibration.py",
        "plotting.py",
        "paper/paper.tex",
        "paper/references.bib",
        "paper/AuthorGuide/einformatica.cls",
        "paper/AuthorGuide/IEEEtran_for_EI.bst",
        "paper/AuthorGuide/EISEJ_logo.png",
        "paper/AuthorGuide/ORCID.pdf",
        "docs/dataset_protocol.md",
        "docs/statistical_analysis_plan.md",
        "scripts/build_latex_tables.py",
        "scripts/build_submission_package.ps1",
        "scripts/c1_anchor_sensitivity.py",
        "scripts/compile_paper.ps1",
        "scripts/extract_paper_numbers.py",
        "scripts/generate_dataset_hash_manifest.py",
        "scripts/regenerate_derived_tables.py",
        "scripts/rerun_pipeline.ps1",
    ]

    paths: set[Path] = set()
    for rel in rel_paths:
        p = root / rel
        if p.exists() and p.is_file():
            paths.add(p)
    return sorted(paths, key=lambda x: str(x).lower())


def _write_code_snapshot_manifest(path: str = "results/code_snapshot_manifest.csv") -> Dict[str, str]:
    import hashlib

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(".").resolve()
    rows: List[Tuple[str, str, int]] = []
    for p in _code_snapshot_candidates():
        rel = p.resolve().relative_to(root).as_posix()
        sha = _file_sha256(str(p))
        size = int(p.stat().st_size)
        rows.append((rel, sha, size))

    digest = hashlib.sha256()
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        f.write("relative_path,sha256,size_bytes\n")
        for rel, sha, size in rows:
            line = f"{rel},{sha},{size}\n"
            f.write(line)
            digest.update(line.encode("utf-8"))

    manifest_sha = _file_sha256(str(manifest_path))
    return {
        "snapshot_manifest_path": manifest_path.as_posix(),
        "snapshot_manifest_sha256": manifest_sha,
        "snapshot_manifest_payload_sha256": digest.hexdigest(),
        "snapshot_file_count": str(len(rows)),
    }


def _git_text(*args: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        return out.strip()
    except Exception:
        return ""


def _code_fingerprint(snapshot_info: Dict[str, str] | None = None) -> Dict[str, str]:
    snapshot_info = snapshot_info or {}
    snapshot_id = os.getenv("CODE_SNAPSHOT_ID", "").strip() or "local-submission-snapshot"
    release_repo_url = (
        os.getenv("PUBLIC_REPO_URL", "").strip() or DEFAULT_PUBLIC_REPO_URL
    )
    release_tag = (
        os.getenv("PUBLIC_RELEASE_TAG", "").strip() or DEFAULT_PUBLIC_RELEASE_TAG
    )
    release_commit = (
        os.getenv("PUBLIC_RELEASE_COMMIT", "").strip() or DEFAULT_PUBLIC_RELEASE_COMMIT
    )
    release_archive_doi = (
        os.getenv("PUBLIC_ARCHIVE_DOI", "").strip() or DEFAULT_PUBLIC_ARCHIVE_DOI
    )

    full_commit = _git_text("rev-parse", "HEAD")
    short_commit = _git_text("rev-parse", "--short", "HEAD")
    status_short = _git_text("status", "--short")
    branch = _git_text("rev-parse", "--abbrev-ref", "HEAD")
    manifest_path = snapshot_info.get("snapshot_manifest_path", "")
    manifest_sha = snapshot_info.get("snapshot_manifest_sha256", "")
    manifest_payload_sha = snapshot_info.get("snapshot_manifest_payload_sha256", "")
    source_commit = release_commit or full_commit
    base = {
        "code_snapshot_id": snapshot_id,
        "code_snapshot_manifest_path": manifest_path,
        "code_snapshot_manifest_sha256": manifest_sha,
        "code_snapshot_manifest_payload_sha256": manifest_payload_sha,
        "code_snapshot_file_count": snapshot_info.get("snapshot_file_count", ""),
        "source_repository_url": release_repo_url,
        "source_release_tag": release_tag,
        "source_release_commit": source_commit,
        "source_archive_doi": release_archive_doi,
        "immutable_snapshot_sha256": manifest_payload_sha or manifest_sha,
    }

    if not full_commit:
        return base | {
            "git_available": "false",
            "git_commit": "",
            "git_commit_short": "",
            "git_branch": "",
            "git_status_short": "",
        }

    return base | {
        "git_available": "true",
        "git_commit": full_commit,
        "git_commit_short": short_commit or full_commit[:8],
        "git_branch": branch or "unknown",
        "git_status_short": status_short if status_short else "clean",
    }


def _split_data(X: pd.DataFrame, y: pd.Series, seed: int):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _to_1d_proba(proba) -> np.ndarray:
    arr = np.asarray(proba)
    if arr.ndim == 2:
        return arr[:, 1]
    return arr.astype(float)


def _optimal_f1_threshold(y_true, proba_1d: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(proba_1d, dtype=float)
    grid = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        preds = (p >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2.0 * tp) / denom
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _policy_metrics(y_true, proba_1d: np.ndarray, threshold: float, cost_fp: float, cost_fn: float):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(proba_1d, dtype=float)
    preds = (p >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    tpr = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    tnr = 0.0 if (tn + fp) == 0 else tn / (tn + fp)
    bal_acc = 0.5 * (tpr + tnr)

    expected_cost = (cost_fp * fp + cost_fn * fn) / max(len(y_true), 1)

    return {
        "f1_policy": float(f1),
        "bal_acc_policy": float(bal_acc),
        "expected_cost_policy": float(expected_cost),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _record(
    results: List[Dict],
    dataset: str,
    encoding: str,
    seed: int,
    model: str,
    corruption: str,
    severity,
    y_true,
    proba_raw,
    T: float,
    policy_results: List[Dict] | None = None,
    policy_thresholds: Dict[str, Dict[str, float]] | None = None,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
):
    metrics_uncal = compute_metrics(y_true, proba_raw)
    proba_cal = temperature_scale_predict(proba_raw, T)
    metrics_cal = compute_metrics(y_true, proba_cal)
    results.append(
        {
            "dataset": dataset,
            "encoding": encoding,
            "seed": seed,
            "model": model,
            "corruption": corruption,
            "severity": severity,
            "variant": "uncal",
            **metrics_uncal,
        }
    )
    results.append(
        {
            "dataset": dataset,
            "encoding": encoding,
            "seed": seed,
            "model": model,
            "corruption": corruption,
            "severity": severity,
            "variant": "temp_scaled",
            **metrics_cal,
        }
    )

    if policy_results is not None and policy_thresholds is not None:
        variant_proba = {
            "uncal": _to_1d_proba(proba_raw),
            "temp_scaled": _to_1d_proba(proba_cal),
        }
        for variant, proba_1d in variant_proba.items():
            thresholds = policy_thresholds.get(variant, {})
            for policy_name, threshold in thresholds.items():
                pm = _policy_metrics(y_true, proba_1d, float(threshold), cost_fp, cost_fn)
                policy_results.append(
                    {
                        "dataset": dataset,
                        "encoding": encoding,
                        "seed": seed,
                        "model": model,
                        "corruption": corruption,
                        "severity": severity,
                        "variant": variant,
                        "policy": policy_name,
                        "threshold": float(threshold),
                        "cost_fp": float(cost_fp),
                        "cost_fn": float(cost_fn),
                        **pm,
                    }
                )


def _record_corruption_diagnostics(
    rows: List[Dict],
    dataset: str,
    encoding: str,
    seed: int,
    model: str,
    corruption: str,
    severity,
    diagnostics: Dict,
):
    payload = {
        "dataset": dataset,
        "encoding": encoding,
        "seed": seed,
        "model": model,
        "corruption": corruption,
        "severity": severity,
    }
    payload.update(diagnostics)
    rows.append(payload)


def _ensure_dirs():
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def _resolve_datasets() -> Tuple[List[str], str]:
    dataset_set = os.getenv("DATASET_SET", "extended").strip().lower()
    if dataset_set == "extended":
        return EXTENDED_DATASETS, dataset_set
    if dataset_set == "core":
        return CORE_DATASETS, dataset_set
    raise ValueError(
        f"Unknown DATASET_SET={dataset_set!r}. Expected one of: 'core', 'extended'."
    )


def _expected_row_counts(n_models: int, n_datasets: int) -> Tuple[int, int]:
    rows_per_setting = 2 * (
        1 + len(SINGLE_CORRUPTIONS) * len(SINGLE_ALPHAS) + len(C2_LEVELS) * len(C4_LEVELS)
    )
    n_with_seed = n_datasets * len(ENCODINGS) * len(SEEDS) * n_models
    n_without_seed = n_datasets * len(ENCODINGS) * n_models
    expected_raw = n_with_seed * rows_per_setting
    expected_summary = n_without_seed * rows_per_setting
    return expected_raw, expected_summary


def _parse_compound_severity(val) -> Tuple[float, float]:
    if isinstance(val, dict):
        return float(val.get("C2", np.nan)), float(val.get("C4", np.nan))
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return float(parsed.get("C2", np.nan)), float(parsed.get("C4", np.nan))
        except Exception:
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def _add_severity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["severity_single"] = pd.to_numeric(out["severity"], errors="coerce")
    parsed = out["severity"].apply(_parse_compound_severity)
    out["severity_c2"] = [x[0] for x in parsed]
    out["severity_c4"] = [x[1] for x in parsed]
    return out


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "encoding", "model", "corruption", "severity", "variant"]
    agg = (
        df.groupby(group_cols)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            bal_acc_mean=("bal_acc", "mean"),
            bal_acc_std=("bal_acc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            logloss_mean=("logloss", "mean"),
            logloss_std=("logloss", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            n=("auc", "count"),
        )
        .reset_index()
    )
    for metric in ALL_METRICS:
        std_col = f"{metric}_std"
        sem_col = f"{metric}_sem"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        agg[std_col] = agg[std_col].fillna(0.0)
        agg[sem_col] = agg[std_col] / np.sqrt(agg["n"].clip(lower=1))
        t_crit = pd.Series(0.0, index=agg.index, dtype=float)
        mask = agg["n"] > 1
        t_crit.loc[mask] = t_dist.ppf(0.975, agg.loc[mask, "n"] - 1)
        half_width = t_crit * agg[sem_col]
        agg[low_col] = agg[f"{metric}_mean"] - half_width
        agg[high_col] = agg[f"{metric}_mean"] + half_width
    return agg


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return p_values
    order = np.argsort(p_values)
    ranks = np.arange(1, m + 1)
    p_sorted = p_values[order]
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_values = np.empty(m, dtype=float)
    q_values[order] = q_sorted
    return q_values


def _hypothesis_family(corruption: str, variant: str) -> str:
    if variant == "uncal" and corruption in {"C2", "C2+C4"}:
        return "primary_confirmatory"
    if variant == "uncal":
        return "secondary_confirmatory"
    return "exploratory"


def _effect_size_label(dz: float) -> str:
    if not np.isfinite(dz):
        return "undefined"
    a = abs(float(dz))
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def _paired_seed_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS
    rng = np.random.default_rng(2026)
    group_cols = ["dataset", "encoding", "model", "corruption", "variant"]
    for (dataset, encoding, model, corruption, variant), sub in df.groupby(group_cols):
        sub = sub.copy()
        if corruption == "C2+C4":
            if {"severity_c2", "severity_c4"}.issubset(sub.columns):
                sub["sev_c2"] = sub["severity_c2"]
                sub["sev_c4"] = sub["severity_c4"]
            else:
                sub[["sev_c2", "sev_c4"]] = sub["severity"].apply(
                    lambda x: pd.Series(_parse_compound_severity(x))
                )
            base = sub[(sub["sev_c2"] == 0.0) & (sub["sev_c4"] == 0.0)]
            high = sub[(sub["sev_c2"] == 0.4) & (sub["sev_c4"] == 0.4)]
            baseline_label = "(0.0,0.0)"
            severe_label = "(0.4,0.4)"
        else:
            if "severity_single" in sub.columns:
                sub["sev_num"] = sub["severity_single"]
            else:
                sub["sev_num"] = pd.to_numeric(sub["severity"], errors="coerce")
            base = sub[sub["sev_num"] == 0.0]
            high = sub[sub["sev_num"] == 0.4]
            baseline_label = "0.0"
            severe_label = "0.4"

        if base.empty or high.empty:
            continue

        merged = base[["seed"] + metrics].merge(
            high[["seed"] + metrics], on="seed", suffixes=("_base", "_high")
        )
        if merged.empty:
            continue

        for metric in metrics:
            base_vals = merged[f"{metric}_base"].astype(float).to_numpy()
            high_vals = merged[f"{metric}_high"].astype(float).to_numpy()
            delta = high_vals - base_vals
            if len(delta) == 0:
                continue
            n_pairs_total = int(len(delta))
            non_zero_delta = delta[np.abs(delta) > 1e-12]
            n_pairs_nonzero = int(len(non_zero_delta))

            delta_mean = float(np.mean(delta))
            delta_std = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
            ci_low, ci_high = _bootstrap_mean_ci(delta, rng=rng)

            t_stat = np.nan
            p_value = np.nan
            if len(delta) > 1:
                test = ttest_rel(high_vals, base_vals, nan_policy="omit")
                if np.isfinite(test.statistic):
                    t_stat = float(test.statistic)
                if np.isfinite(test.pvalue):
                    p_value = float(test.pvalue)

            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
            if n_pairs_nonzero > 0:
                try:
                    w = wilcoxon(non_zero_delta, alternative="two-sided", zero_method="wilcox")
                    if np.isfinite(w.statistic):
                        wilcoxon_stat = float(w.statistic)
                    if np.isfinite(w.pvalue):
                        wilcoxon_p = float(w.pvalue)
                except Exception:
                    pass

            effect_size = np.nan
            if delta_std > 0:
                effect_size = float(delta_mean / delta_std)
            effect_label = _effect_size_label(effect_size)
            family = _hypothesis_family(corruption, variant)

            rows.append(
                {
                    "dataset": dataset,
                    "encoding": encoding,
                    "model": model,
                    "corruption": corruption,
                    "variant": variant,
                    "metric": metric,
                    "baseline_severity": baseline_label,
                    "severe_severity": severe_label,
                    "hypothesis_family": family,
                    "n_pairs": n_pairs_total,
                    "n_pairs_total": n_pairs_total,
                    "n_pairs_nonzero": n_pairs_nonzero,
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "paired_t_stat": t_stat,
                    "paired_t_p_value": p_value,
                    "wilcoxon_stat": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                    "paired_effect_size_dz": effect_size,
                    "paired_effect_size_label": effect_label,
                }
            )

    effects = pd.DataFrame(rows)
    if effects.empty:
        return effects

    effects["paired_t_p_value_bh_global_metric"] = np.nan
    for metric, metric_sub in effects.groupby("metric"):
        valid = metric_sub["paired_t_p_value"].notna()
        if not valid.any():
            continue
        idx = metric_sub.index[valid]
        q_values = _benjamini_hochberg(metric_sub.loc[idx, "paired_t_p_value"].to_numpy())
        effects.loc[idx, "paired_t_p_value_bh_global_metric"] = q_values

    effects["paired_t_p_value_bh_family_metric"] = np.nan
    effects["wilcoxon_p_value_bh_family_metric"] = np.nan
    for (family, metric), fam_sub in effects.groupby(["hypothesis_family", "metric"]):
        valid_t = fam_sub["paired_t_p_value"].notna()
        if valid_t.any():
            idx_t = fam_sub.index[valid_t]
            q_t = _benjamini_hochberg(fam_sub.loc[idx_t, "paired_t_p_value"].to_numpy())
            effects.loc[idx_t, "paired_t_p_value_bh_family_metric"] = q_t

        valid_w = fam_sub["wilcoxon_p_value"].notna()
        if valid_w.any():
            idx_w = fam_sub.index[valid_w]
            q_w = _benjamini_hochberg(fam_sub.loc[idx_w, "wilcoxon_p_value"].to_numpy())
            effects.loc[idx_w, "wilcoxon_p_value_bh_family_metric"] = q_w
    return effects


def _single_corruption_delta_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS
    uncal = summary_df[summary_df["variant"] == "uncal"].copy()
    rng = np.random.default_rng(2027)
    for dataset in sorted(uncal["dataset"].unique()):
        ds = uncal[uncal["dataset"] == dataset]
        for corruption in SINGLE_CORRUPTIONS:
            sub = ds[ds["corruption"] == corruption].copy()
            if "severity_single" in sub.columns:
                sub["sev_num"] = sub["severity_single"]
            else:
                sub["sev_num"] = pd.to_numeric(sub["severity"], errors="coerce")
            base = sub[sub["sev_num"] == 0.0]
            high = sub[sub["sev_num"] == 0.4]
            key_cols = ["encoding", "model"]
            merged = base[key_cols + [f"{m}_mean" for m in metrics]].merge(
                high[key_cols + [f"{m}_mean" for m in metrics]],
                on=key_cols,
                suffixes=("_base", "_high"),
            )
            if merged.empty:
                continue

            row = {"dataset": dataset, "corruption": corruption}
            for metric in metrics:
                delta = (
                    merged[f"{metric}_mean_high"].to_numpy()
                    - merged[f"{metric}_mean_base"].to_numpy()
                )
                ci_low, ci_high = _bootstrap_mean_ci(delta, rng=rng)
                row[f"delta_{metric}_mean"] = float(np.mean(delta))
                row[f"delta_{metric}_ci95_low_boot"] = ci_low
                row[f"delta_{metric}_ci95_high_boot"] = ci_high
            rows.append(row)
    return pd.DataFrame(rows)


def _primary_inference_table(effects_df: pd.DataFrame) -> pd.DataFrame:
    if effects_df.empty:
        return pd.DataFrame()

    sub = effects_df[effects_df["hypothesis_family"] == "primary_confirmatory"].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    rng = np.random.default_rng(2030)
    for (corruption, metric), g in sub.groupby(["corruption", "metric"]):
        deltas = g["delta_mean"].dropna().to_numpy()
        ci_low, ci_high = _bootstrap_mean_ci(deltas, rng=rng)

        rows.append(
            {
                "corruption": corruption,
                "metric": metric,
                "n_settings": int(len(g)),
                "n_pairs_total_min": int(g["n_pairs_total"].min()),
                "n_pairs_total_max": int(g["n_pairs_total"].max()),
                "n_pairs_nonzero_median": float(g["n_pairs_nonzero"].median()),
                "delta_mean_across_settings": float(np.mean(deltas)) if len(deltas) else np.nan,
                "delta_ci95_low_boot": ci_low,
                "delta_ci95_high_boot": ci_high,
                "effect_size_dz_mean": float(g["paired_effect_size_dz"].dropna().mean()),
                "effect_size_dz_median": float(g["paired_effect_size_dz"].dropna().median()),
                "n_sig_t_bh_family_metric": int(
                    (
                        g["paired_t_p_value_bh_family_metric"].notna()
                        & (g["paired_t_p_value_bh_family_metric"] < 0.05)
                    ).sum()
                ),
                "n_sig_wilcoxon_bh_family_metric": int(
                    (
                        g["wilcoxon_p_value_bh_family_metric"].notna()
                        & (g["wilcoxon_p_value_bh_family_metric"] < 0.05)
                    ).sum()
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    corruption_order = {"C2": 0, "C2+C4": 1}
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _primary_dataset_seed_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2031)
    sub = df[
        (df["variant"] == "uncal") & (df["corruption"].isin(["C2", "C2+C4"]))
    ].copy()
    if sub.empty:
        return pd.DataFrame(rows)

    group_cols = ["dataset", "corruption", "encoding", "model", "seed"]
    for dataset in sorted(sub["dataset"].unique()):
        ds = sub[sub["dataset"] == dataset].copy()
        for corruption in ["C2", "C2+C4"]:
            corr = ds[ds["corruption"] == corruption].copy()
            if corr.empty:
                continue

            if corruption == "C2+C4":
                if {"severity_c2", "severity_c4"}.issubset(corr.columns):
                    corr["sev_c2"] = corr["severity_c2"]
                    corr["sev_c4"] = corr["severity_c4"]
                else:
                    corr[["sev_c2", "sev_c4"]] = corr["severity"].apply(
                        lambda x: pd.Series(_parse_compound_severity(x))
                    )
                base = corr[(corr["sev_c2"] == 0.0) & (corr["sev_c4"] == 0.0)]
                high = corr[(corr["sev_c2"] == 0.4) & (corr["sev_c4"] == 0.4)]
                baseline_label = "(0.0,0.0)"
                severe_label = "(0.4,0.4)"
            else:
                if "severity_single" in corr.columns:
                    corr["sev_num"] = corr["severity_single"]
                else:
                    corr["sev_num"] = pd.to_numeric(corr["severity"], errors="coerce")
                base = corr[corr["sev_num"] == 0.0]
                high = corr[corr["sev_num"] == 0.4]
                baseline_label = "0.0"
                severe_label = "0.4"

            if base.empty or high.empty:
                continue

            for metric in PRIMARY_METRICS:
                merged = base[group_cols + [metric]].merge(
                    high[group_cols + [metric]],
                    on=group_cols,
                    suffixes=("_base", "_high"),
                )
                if merged.empty:
                    continue

                seed_stats = (
                    merged.groupby("seed")
                    .agg(
                        metric_base=(f"{metric}_base", "mean"),
                        metric_high=(f"{metric}_high", "mean"),
                    )
                    .reset_index()
                )
                if seed_stats.empty:
                    continue

                base_vals = seed_stats["metric_base"].to_numpy(dtype=float)
                high_vals = seed_stats["metric_high"].to_numpy(dtype=float)
                delta = high_vals - base_vals
                n_pairs_total = int(len(delta))
                non_zero_delta = delta[np.abs(delta) > 1e-12]
                n_pairs_nonzero = int(len(non_zero_delta))
                delta_mean = float(np.mean(delta))
                delta_std = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
                ci_low, ci_high = _bootstrap_mean_ci(delta, rng=rng)

                t_stat = np.nan
                p_value = np.nan
                if len(delta) > 1:
                    test = ttest_rel(high_vals, base_vals, nan_policy="omit")
                    if np.isfinite(test.statistic):
                        t_stat = float(test.statistic)
                    if np.isfinite(test.pvalue):
                        p_value = float(test.pvalue)

                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan
                if n_pairs_nonzero > 0:
                    try:
                        w = wilcoxon(non_zero_delta, alternative="two-sided", zero_method="wilcox")
                        if np.isfinite(w.statistic):
                            wilcoxon_stat = float(w.statistic)
                        if np.isfinite(w.pvalue):
                            wilcoxon_p = float(w.pvalue)
                    except Exception:
                        pass

                effect_size = np.nan
                if delta_std > 0:
                    effect_size = float(delta_mean / delta_std)

                rows.append(
                    {
                        "dataset": dataset,
                        "corruption": corruption,
                        "metric": metric,
                        "baseline_severity": baseline_label,
                        "severe_severity": severe_label,
                        "n_pairs_total": n_pairs_total,
                        "n_pairs_nonzero": n_pairs_nonzero,
                        "delta_mean": delta_mean,
                        "delta_std": delta_std,
                        "delta_ci95_low_boot": ci_low,
                        "delta_ci95_high_boot": ci_high,
                        "paired_t_stat": t_stat,
                        "paired_t_p_value": p_value,
                        "wilcoxon_stat": wilcoxon_stat,
                        "wilcoxon_p_value": wilcoxon_p,
                        "paired_effect_size_dz": effect_size,
                        "paired_effect_size_label": _effect_size_label(effect_size),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["paired_t_p_value_bh_metric"] = np.nan
    out["wilcoxon_p_value_bh_metric"] = np.nan
    for metric, sub_m in out.groupby("metric"):
        valid_t = sub_m["paired_t_p_value"].notna()
        if valid_t.any():
            idx_t = sub_m.index[valid_t]
            q_t = _benjamini_hochberg(sub_m.loc[idx_t, "paired_t_p_value"].to_numpy())
            out.loc[idx_t, "paired_t_p_value_bh_metric"] = q_t

        valid_w = sub_m["wilcoxon_p_value"].notna()
        if valid_w.any():
            idx_w = sub_m.index[valid_w]
            q_w = _benjamini_hochberg(sub_m.loc[idx_w, "wilcoxon_p_value"].to_numpy())
            out.loc[idx_w, "wilcoxon_p_value_bh_metric"] = q_w

    return out


def _primary_dataset_inference_summary(dataset_effects: pd.DataFrame) -> pd.DataFrame:
    if dataset_effects.empty:
        return pd.DataFrame()

    rows = []
    rng = np.random.default_rng(2032)
    for (corruption, metric), g in dataset_effects.groupby(["corruption", "metric"]):
        deltas = g["delta_mean"].dropna().to_numpy(dtype=float)
        if len(deltas) == 0:
            continue
        ci_low, ci_high = _bootstrap_mean_ci(deltas, rng=rng)

        if metric in {"auc", "f1"}:
            harmful = int((deltas < 0).sum())
        else:
            harmful = int((deltas > 0).sum())

        rows.append(
            {
                "corruption": corruption,
                "metric": metric,
                "n_datasets": int(len(g)),
                "n_pairs_total_min": int(g["n_pairs_total"].min()),
                "n_pairs_total_max": int(g["n_pairs_total"].max()),
                "n_pairs_nonzero_median": float(g["n_pairs_nonzero"].median()),
                "delta_mean_across_datasets": float(np.mean(deltas)),
                "delta_median_across_datasets": float(np.median(deltas)),
                "delta_ci95_low_boot": ci_low,
                "delta_ci95_high_boot": ci_high,
                "harmful_direction_count": harmful,
                "harmful_direction_rate": float(harmful / max(len(g), 1)),
                "effect_size_dz_median": float(g["paired_effect_size_dz"].dropna().median()),
                "n_sig_t_bh_metric": int(
                    (
                        g["paired_t_p_value_bh_metric"].notna()
                        & (g["paired_t_p_value_bh_metric"] < 0.05)
                    ).sum()
                ),
                "n_sig_wilcoxon_bh_metric": int(
                    (
                        g["wilcoxon_p_value_bh_metric"].notna()
                        & (g["wilcoxon_p_value_bh_metric"] < 0.05)
                    ).sum()
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    corruption_order = {"C2": 0, "C2+C4": 1}
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _primary_hypothesis_tests_table(dataset_effects: pd.DataFrame) -> pd.DataFrame:
    if dataset_effects.empty:
        return pd.DataFrame()

    rows = []
    rng = np.random.default_rng(2033)
    for corruption in ["C2", "C2+C4"]:
        for metric in PRIMARY_METRICS:
            sub = dataset_effects[
                (dataset_effects["corruption"] == corruption)
                & (dataset_effects["metric"] == metric)
            ].copy()
            if sub.empty:
                continue

            deltas = sub["delta_mean"].to_numpy(dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            if len(deltas) == 0:
                continue

            ci_low, ci_high = _bootstrap_mean_ci(deltas, rng=rng)
            n_total = int(len(deltas))
            n_nonzero = int((np.abs(deltas) > 1e-12).sum())

            t_stat = np.nan
            t_p = np.nan
            if n_total > 1:
                t_res = ttest_1samp(deltas, popmean=0.0, nan_policy="omit")
                if np.isfinite(t_res.statistic):
                    t_stat = float(t_res.statistic)
                if np.isfinite(t_res.pvalue):
                    t_p = float(t_res.pvalue)

            w_stat = np.nan
            w_p = np.nan
            if n_nonzero > 0:
                try:
                    w_res = wilcoxon(
                        deltas[np.abs(deltas) > 1e-12],
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    if np.isfinite(w_res.statistic):
                        w_stat = float(w_res.statistic)
                    if np.isfinite(w_res.pvalue):
                        w_p = float(w_res.pvalue)
                except Exception:
                    pass

            if metric in {"auc", "f1"}:
                harmful_count = int((deltas < 0).sum())
            else:
                harmful_count = int((deltas > 0).sum())

            rows.append(
                {
                    "corruption": corruption,
                    "metric": metric,
                    "n_datasets": n_total,
                    "n_datasets_nonzero": n_nonzero,
                    "delta_mean": float(np.mean(deltas)),
                    "delta_median": float(np.median(deltas)),
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "paired_effect_size_dz_median": float(
                        sub["paired_effect_size_dz"].dropna().median()
                    ),
                    "t_statistic": t_stat,
                    "t_p_value": t_p,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_p_value": w_p,
                    "harmful_direction_count": harmful_count,
                    "harmful_direction_rate": float(harmful_count / max(n_total, 1)),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["t_p_value_bh_primary6"] = np.nan
    table["wilcoxon_p_value_bh_primary6"] = np.nan
    valid_t = table["t_p_value"].notna()
    if valid_t.any():
        idx = table.index[valid_t]
        table.loc[idx, "t_p_value_bh_primary6"] = _benjamini_hochberg(
            table.loc[idx, "t_p_value"].to_numpy()
        )
    valid_w = table["wilcoxon_p_value"].notna()
    if valid_w.any():
        idx = table.index[valid_w]
        table.loc[idx, "wilcoxon_p_value_bh_primary6"] = _benjamini_hochberg(
            table.loc[idx, "wilcoxon_p_value"].to_numpy()
        )

    corr_order = {"C2": 0, "C2+C4": 1}
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corr_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _primary_hierarchical_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster-aware sensitivity for primary severe-vs-baseline effects.

    The bootstrap preserves the nested structure:
    dataset -> (encoding, model) -> seed.
    """
    sub = df[
        (df["variant"] == "uncal") & (df["corruption"].isin(["C2", "C2+C4"]))
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    base_rng = np.random.default_rng(2040)
    key_cols = ["dataset", "encoding", "model", "seed"]

    for corruption in ["C2", "C2+C4"]:
        corr = sub[sub["corruption"] == corruption].copy()
        if corr.empty:
            continue
        if corruption == "C2+C4":
            if {"severity_c2", "severity_c4"}.issubset(corr.columns):
                corr["sev_c2"] = corr["severity_c2"]
                corr["sev_c4"] = corr["severity_c4"]
            else:
                corr[["sev_c2", "sev_c4"]] = corr["severity"].apply(
                    lambda x: pd.Series(_parse_compound_severity(x))
                )
            base = corr[(corr["sev_c2"] == 0.0) & (corr["sev_c4"] == 0.0)]
            high = corr[(corr["sev_c2"] == 0.4) & (corr["sev_c4"] == 0.4)]
        else:
            if "severity_single" in corr.columns:
                corr["sev_num"] = corr["severity_single"]
            else:
                corr["sev_num"] = pd.to_numeric(corr["severity"], errors="coerce")
            base = corr[corr["sev_num"] == 0.0]
            high = corr[corr["sev_num"] == 0.4]

        if base.empty or high.empty:
            continue

        for metric in PRIMARY_METRICS:
            merged = base[key_cols + [metric]].merge(
                high[key_cols + [metric]],
                on=key_cols,
                suffixes=("_base", "_high"),
            )
            if merged.empty:
                continue

            merged = merged.copy()
            merged["delta"] = merged[f"{metric}_high"] - merged[f"{metric}_base"]
            merged = merged[np.isfinite(merged["delta"])]
            if merged.empty:
                continue

            ds_effects = (
                merged.groupby("dataset")["delta"].mean().dropna().to_numpy(dtype=float)
            )
            if len(ds_effects) == 0:
                continue
            naive_mean = float(np.mean(ds_effects))
            naive_low, naive_high = _bootstrap_mean_ci(ds_effects, rng=base_rng)

            by_dataset: List[List[np.ndarray]] = []
            for _, ds in merged.groupby("dataset"):
                settings = []
                for _, s in ds.groupby(["encoding", "model"]):
                    vals = s["delta"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        settings.append(vals)
                if settings:
                    by_dataset.append(settings)

            if not by_dataset:
                continue

            n_datasets = len(by_dataset)
            boot = np.empty(5000, dtype=float)
            for i in range(5000):
                sampled_ds = base_rng.integers(0, n_datasets, size=n_datasets)
                ds_means = np.empty(n_datasets, dtype=float)
                for j, ds_idx in enumerate(sampled_ds):
                    settings = by_dataset[int(ds_idx)]
                    n_settings = len(settings)
                    sampled_settings = base_rng.integers(0, n_settings, size=n_settings)
                    setting_means = np.empty(n_settings, dtype=float)
                    for k, s_idx in enumerate(sampled_settings):
                        vals = settings[int(s_idx)]
                        n_vals = len(vals)
                        sampled_vals = vals[base_rng.integers(0, n_vals, size=n_vals)]
                        setting_means[k] = float(np.mean(sampled_vals))
                    ds_means[j] = float(np.mean(setting_means))
                boot[i] = float(np.mean(ds_means))

            h_low = float(np.percentile(boot, 2.5))
            h_high = float(np.percentile(boot, 97.5))

            naive_nonzero = bool(naive_low > 0.0 or naive_high < 0.0)
            hier_nonzero = bool(h_low > 0.0 or h_high < 0.0)
            if metric in {"auc", "f1"}:
                harmful_naive = bool(naive_high < 0.0)
                harmful_hier = bool(h_high < 0.0)
                harmful_direction = "delta < 0"
            else:
                harmful_naive = bool(naive_low > 0.0)
                harmful_hier = bool(h_low > 0.0)
                harmful_direction = "delta > 0"

            rows.append(
                {
                    "corruption": corruption,
                    "metric": metric,
                    "n_datasets": int(len(ds_effects)),
                    "n_setting_seed_pairs": int(len(merged)),
                    "delta_mean_across_datasets": naive_mean,
                    "naive_ci95_low_boot": naive_low,
                    "naive_ci95_high_boot": naive_high,
                    "hierarchical_ci95_low_boot": h_low,
                    "hierarchical_ci95_high_boot": h_high,
                    "naive_ci_excludes_zero": str(naive_nonzero).lower(),
                    "hierarchical_ci_excludes_zero": str(hier_nonzero).lower(),
                    "harmful_direction": harmful_direction,
                    "harmful_supported_naive": str(harmful_naive).lower(),
                    "harmful_supported_hierarchical": str(harmful_hier).lower(),
                    "headline_direction_unchanged": str(harmful_naive == harmful_hier).lower(),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    corr_order = {"C2": 0, "C2+C4": 1}
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corr_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _compound_delta_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = PRIMARY_METRICS
    uncal = summary_df[
        (summary_df["variant"] == "uncal") & (summary_df["corruption"] == "C2+C4")
    ].copy()
    if uncal.empty:
        return pd.DataFrame(rows)

    if {"severity_c2", "severity_c4"}.issubset(uncal.columns):
        uncal["sev_c2"] = uncal["severity_c2"]
        uncal["sev_c4"] = uncal["severity_c4"]
    else:
        uncal[["sev_c2", "sev_c4"]] = uncal["severity"].apply(
            lambda x: pd.Series(_parse_compound_severity(x))
        )

    rng = np.random.default_rng(2028)
    for dataset in sorted(uncal["dataset"].unique()):
        sub = uncal[uncal["dataset"] == dataset]
        base = sub[(sub["sev_c2"] == 0.0) & (sub["sev_c4"] == 0.0)]
        high = sub[(sub["sev_c2"] == 0.4) & (sub["sev_c4"] == 0.4)]
        key_cols = ["encoding", "model"]
        merged = base[key_cols + [f"{m}_mean" for m in metrics]].merge(
            high[key_cols + [f"{m}_mean" for m in metrics]],
            on=key_cols,
            suffixes=("_base", "_high"),
        )
        if merged.empty:
            continue

        row = {"dataset": dataset, "corruption": "C2+C4 (0.4,0.4)"}
        for metric in metrics:
            delta = (
                merged[f"{metric}_mean_high"].to_numpy()
                - merged[f"{metric}_mean_base"].to_numpy()
            )
            ci_low, ci_high = _bootstrap_mean_ci(delta, rng=rng)
            row[f"delta_{metric}_mean"] = float(np.mean(delta))
            row[f"delta_{metric}_ci95_low_boot"] = ci_low
            row[f"delta_{metric}_ci95_high_boot"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def _clean_performance_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    clean = summary_df[
        (summary_df["corruption"] == "C0_clean") & (summary_df["severity_single"] == 0.0)
    ].copy()
    if clean.empty:
        return clean
    cols = [
        "dataset",
        "encoding",
        "model",
        "variant",
        "auc_mean",
        "auc_ci95_low",
        "auc_ci95_high",
        "pr_auc_mean",
        "pr_auc_ci95_low",
        "pr_auc_ci95_high",
        "f1_mean",
        "f1_ci95_low",
        "f1_ci95_high",
        "bal_acc_mean",
        "bal_acc_ci95_low",
        "bal_acc_ci95_high",
        "brier_mean",
        "brier_ci95_low",
        "brier_ci95_high",
        "logloss_mean",
        "logloss_ci95_low",
        "logloss_ci95_high",
        "ece_mean",
        "ece_ci95_low",
        "ece_ci95_high",
    ]
    return clean[cols].sort_values(["dataset", "encoding", "model", "variant"]).reset_index(
        drop=True
    )


def _cross_dataset_effect_table(
    single_delta: pd.DataFrame,
    compound_delta: pd.DataFrame,
    dataset_profiles: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2029)
    metrics = PRIMARY_METRICS
    c3_eligible_datasets = None
    if dataset_profiles:
        c3_eligible_datasets = {
            d for d, prof in dataset_profiles.items() if int(prof.get("n_cat_cols", 0)) > 0
        }

    for corruption in SINGLE_CORRUPTIONS:
        sub = single_delta[single_delta["corruption"] == corruption]
        eligibility = "all"
        if corruption == "C3" and c3_eligible_datasets is not None:
            sub = sub[sub["dataset"].isin(c3_eligible_datasets)]
            eligibility = "categorical_only"
        for metric in metrics:
            vals = sub[f"delta_{metric}_mean"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(vals, rng=rng)
            rows.append(
                {
                    "corruption": corruption,
                    "metric": metric,
                    "n_datasets": int(len(vals)),
                    "delta_mean_across_datasets": float(np.mean(vals)),
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "aggregation": "dataset_unweighted_mean",
                    "eligibility": eligibility,
                }
            )

    if not compound_delta.empty:
        for metric in metrics:
            vals = compound_delta[f"delta_{metric}_mean"].dropna().to_numpy()
            if len(vals) == 0:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(vals, rng=rng)
            rows.append(
                {
                    "corruption": "C2+C4 (0.4,0.4)",
                    "metric": metric,
                    "n_datasets": int(len(vals)),
                    "delta_mean_across_datasets": float(np.mean(vals)),
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "aggregation": "dataset_unweighted_mean",
                    "eligibility": "all",
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    corruption_order = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C2+C4 (0.4,0.4)": 4,
    }
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _weighted_mean_and_ci(
    values: np.ndarray, weights: np.ndarray, rng: np.random.Generator, n_boot: int = 5000
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.average(values, weights=weights))
    if len(values) == 1:
        return (mean, mean, mean)

    boot = []
    idx = np.arange(len(values))
    for _ in range(n_boot):
        sampled = rng.choice(idx, size=len(idx), replace=True)
        boot_vals = values[sampled]
        boot_w = weights[sampled]
        boot.append(np.average(boot_vals, weights=boot_w))
    return (mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def _cross_dataset_effect_weighted_table(
    single_delta: pd.DataFrame,
    compound_delta: pd.DataFrame,
    dataset_profiles: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2038)
    metrics = PRIMARY_METRICS
    weights = {
        d: float(prof.get("n_rows", np.nan))
        for d, prof in dataset_profiles.items()
        if np.isfinite(float(prof.get("n_rows", np.nan)))
    }
    c3_eligible_datasets = {
        d for d, prof in dataset_profiles.items() if int(prof.get("n_cat_cols", 0)) > 0
    }

    for corruption in SINGLE_CORRUPTIONS:
        sub = single_delta[single_delta["corruption"] == corruption].copy()
        if corruption == "C3":
            sub = sub[sub["dataset"].isin(c3_eligible_datasets)]
        for metric in metrics:
            if sub.empty:
                continue
            vals = sub[f"delta_{metric}_mean"].to_numpy(dtype=float)
            w = sub["dataset"].map(weights).to_numpy(dtype=float)
            mean, ci_low, ci_high = _weighted_mean_and_ci(vals, w, rng=rng)
            rows.append(
                {
                    "corruption": corruption,
                    "metric": metric,
                    "n_datasets": int(np.isfinite(vals).sum()),
                    "delta_weighted_mean": mean,
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "weight_basis": "dataset_n_rows",
                }
            )

    if not compound_delta.empty:
        sub = compound_delta.copy()
        for metric in metrics:
            vals = sub[f"delta_{metric}_mean"].to_numpy(dtype=float)
            w = sub["dataset"].map(weights).to_numpy(dtype=float)
            mean, ci_low, ci_high = _weighted_mean_and_ci(vals, w, rng=rng)
            rows.append(
                {
                    "corruption": "C2+C4 (0.4,0.4)",
                    "metric": metric,
                    "n_datasets": int(np.isfinite(vals).sum()),
                    "delta_weighted_mean": mean,
                    "delta_ci95_low_boot": ci_low,
                    "delta_ci95_high_boot": ci_high,
                    "weight_basis": "dataset_n_rows",
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    corruption_order = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C2+C4 (0.4,0.4)": 4,
    }
    metric_order = {"auc": 0, "f1": 1, "ece": 2}
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["corruption_order", "metric_order"]).drop(
        columns=["corruption_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _corruption_diagnostics_table(diag_df: pd.DataFrame) -> pd.DataFrame:
    if diag_df.empty:
        return diag_df

    group_cols = ["dataset", "corruption", "severity"]
    numeric_cols = []
    ignore_numeric = {"seed", "severity_single", "severity_c2", "severity_c4"}
    for col in diag_df.columns:
        if col in group_cols or col in ignore_numeric:
            continue
        if pd.api.types.is_numeric_dtype(diag_df[col]):
            numeric_cols.append(col)

    agg_parts = []
    if numeric_cols:
        num_agg = diag_df.groupby(group_cols)[numeric_cols].mean().reset_index()
        agg_parts.append(num_agg)

    cat_candidates = [
        "c1_strategy",
        "c1_anchor_col",
        "c1_anchor_value",
        "c2_anchor_strategy",
        "c2_anchor_col",
        "c2_anchor_value",
        "c3_pattern_families_used",
    ]
    cat_cols = [c for c in cat_candidates if c in diag_df.columns]
    if cat_cols:
        def _mode_or_first(s):
            s = s.astype(str)
            s = s[s != "nan"]
            if s.empty:
                return ""
            mode = s.mode()
            if len(mode) > 0:
                return str(mode.iloc[0])
            return str(s.iloc[0])

        cat_agg = diag_df.groupby(group_cols)[cat_cols].agg(_mode_or_first).reset_index()
        agg_parts.append(cat_agg)

    counts = diag_df.groupby(group_cols).size().rename("n_settings").reset_index()
    out = counts
    for part in agg_parts:
        out = out.merge(part, on=group_cols, how="left")

    out = _add_severity_columns(out)
    corr_order = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C2+C4": 4}
    out["corruption_order"] = out["corruption"].map(corr_order).fillna(99)
    out = out.sort_values(["dataset", "corruption_order", "severity_single", "severity"])
    return out.drop(columns=["corruption_order"]).reset_index(drop=True)


def _dataset_profiles_table(dataset_profiles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset, profile in dataset_profiles.items():
        row = {"dataset": dataset}
        row.update(profile)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    cols = [
        "dataset",
        "source",
        "source_policy",
        "source_sha256",
        "openml_data_id",
        "n_rows",
        "n_features",
        "n_num_cols",
        "n_cat_cols",
        "positive_rate",
        "overall_missing_rate",
        "max_col_missing_rate",
        "n_constant_cols",
    ]
    table = pd.DataFrame(rows)
    for col in cols:
        if col not in table.columns:
            table[col] = ""
    return table[cols].sort_values("dataset").reset_index(drop=True)


def _aggregate_policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "encoding",
        "model",
        "corruption",
        "severity",
        "variant",
        "policy",
    ]
    agg = (
        df.groupby(group_cols)
        .agg(
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            f1_policy_mean=("f1_policy", "mean"),
            f1_policy_std=("f1_policy", "std"),
            bal_acc_policy_mean=("bal_acc_policy", "mean"),
            bal_acc_policy_std=("bal_acc_policy", "std"),
            expected_cost_policy_mean=("expected_cost_policy", "mean"),
            expected_cost_policy_std=("expected_cost_policy", "std"),
            n=("f1_policy", "count"),
        )
        .reset_index()
    )

    for metric in ["threshold", "f1_policy", "bal_acc_policy", "expected_cost_policy"]:
        std_col = f"{metric}_std"
        sem_col = f"{metric}_sem"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        mean_col = f"{metric}_mean"
        agg[std_col] = agg[std_col].fillna(0.0)
        agg[sem_col] = agg[std_col] / np.sqrt(agg["n"].clip(lower=1))
        t_crit = pd.Series(0.0, index=agg.index, dtype=float)
        mask = agg["n"] > 1
        t_crit.loc[mask] = t_dist.ppf(0.975, agg.loc[mask, "n"] - 1)
        half_width = t_crit * agg[sem_col]
        agg[low_col] = agg[mean_col] - half_width
        agg[high_col] = agg[mean_col] + half_width
    return agg


def _threshold_policy_comparison_table(
    policy_summary: pd.DataFrame,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2034)
    scenarios = [
        ("C0_clean", "clean_0.0"),
        ("C2", "single_0.4"),
        ("C3", "single_0.4"),
        ("C4", "single_0.4"),
        ("C2+C4", "compound_0.4_0.4"),
    ]
    policies = ["fixed_0_5", "val_tuned_f1", "cost_sensitive"]

    for corruption, scenario in scenarios:
        if corruption == "C2+C4":
            sub = policy_summary[
                (policy_summary["corruption"] == "C2+C4")
                & (policy_summary["severity_c2"] == 0.4)
                & (policy_summary["severity_c4"] == 0.4)
            ]
        elif corruption == "C0_clean":
            sub = policy_summary[
                (policy_summary["corruption"] == "C0_clean")
                & (policy_summary["severity_single"] == 0.0)
            ]
        else:
            sub = policy_summary[
                (policy_summary["corruption"] == corruption)
                & (policy_summary["severity_single"] == 0.4)
            ]
            if corruption == "C3" and c3_eligible_datasets is not None:
                sub = sub[sub["dataset"].isin(c3_eligible_datasets)]

        if sub.empty:
            continue

        for variant, var_sub in sub.groupby("variant"):
            if var_sub.empty:
                continue
            base = var_sub[var_sub["policy"] == "fixed_0_5"]
            if base.empty:
                continue

            key_cols = ["dataset", "encoding", "model"]
            base_k = base[key_cols + ["f1_policy_mean", "bal_acc_policy_mean", "expected_cost_policy_mean"]]
            base_k = base_k.rename(
                columns={
                    "f1_policy_mean": "f1_base",
                    "bal_acc_policy_mean": "bal_acc_base",
                    "expected_cost_policy_mean": "cost_base",
                }
            )

            for policy in policies:
                pol = var_sub[var_sub["policy"] == policy]
                if pol.empty:
                    continue
                merged = pol.merge(base_k, on=key_cols, how="inner")
                if merged.empty:
                    continue
                delta_f1 = (merged["f1_policy_mean"] - merged["f1_base"]).to_numpy(dtype=float)
                delta_cost = (
                    merged["expected_cost_policy_mean"] - merged["cost_base"]
                ).to_numpy(dtype=float)
                ci_f1_low, ci_f1_high = _bootstrap_mean_ci(delta_f1, rng=rng)
                ci_cost_low, ci_cost_high = _bootstrap_mean_ci(delta_cost, rng=rng)

                row = {
                    "scenario": scenario,
                    "corruption": corruption,
                    "variant": variant,
                    "policy": policy,
                    "n_settings": int(len(merged)),
                    "threshold_mean": float(merged["threshold_mean"].mean()),
                    "f1_mean": float(merged["f1_policy_mean"].mean()),
                    "bal_acc_mean": float(merged["bal_acc_policy_mean"].mean()),
                    "expected_cost_mean": float(merged["expected_cost_policy_mean"].mean()),
                    "delta_f1_vs_fixed": float(
                        delta_f1.mean()
                    ),
                    "delta_bal_acc_vs_fixed": float(
                        (merged["bal_acc_policy_mean"] - merged["bal_acc_base"]).mean()
                    ),
                    "delta_cost_vs_fixed": float(
                        delta_cost.mean()
                    ),
                    "delta_f1_vs_fixed_ci95_low_boot": ci_f1_low,
                    "delta_f1_vs_fixed_ci95_high_boot": ci_f1_high,
                    "delta_cost_vs_fixed_ci95_low_boot": ci_cost_low,
                    "delta_cost_vs_fixed_ci95_high_boot": ci_cost_high,
                }
                rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    scenario_order = {
        "clean_0.0": 0,
        "single_0.4": 1,
        "compound_0.4_0.4": 2,
    }
    corruption_order = {"C0_clean": 0, "C2": 1, "C3": 2, "C4": 3, "C2+C4": 4}
    policy_order = {"fixed_0_5": 0, "val_tuned_f1": 1, "cost_sensitive": 2}

    table["scenario_order"] = table["scenario"].map(scenario_order)
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["policy_order"] = table["policy"].map(policy_order)
    table = table.sort_values(["scenario_order", "corruption_order", "variant", "policy_order"])
    return table.drop(columns=["scenario_order", "corruption_order", "policy_order"]).reset_index(drop=True)


def _paired_difference_inference(
    delta: np.ndarray, rng: np.random.Generator
) -> Dict[str, float]:
    vals = np.asarray(delta, dtype=float)
    vals = vals[np.isfinite(vals)]
    n_total = int(len(vals))
    n_nonzero = int((np.abs(vals) > 1e-12).sum())
    mean_delta = float(np.mean(vals)) if n_total else np.nan
    ci_low, ci_high = _bootstrap_mean_ci(vals, rng=rng) if n_total else (np.nan, np.nan)

    t_stat = np.nan
    t_p = np.nan
    if n_total > 1:
        t_res = ttest_1samp(vals, popmean=0.0, nan_policy="omit")
        if np.isfinite(t_res.statistic):
            t_stat = float(t_res.statistic)
        if np.isfinite(t_res.pvalue):
            t_p = float(t_res.pvalue)

    w_stat = np.nan
    w_p = np.nan
    if n_nonzero > 0:
        try:
            w_res = wilcoxon(
                vals[np.abs(vals) > 1e-12],
                alternative="two-sided",
                zero_method="wilcox",
            )
            if np.isfinite(w_res.statistic):
                w_stat = float(w_res.statistic)
            if np.isfinite(w_res.pvalue):
                w_p = float(w_res.pvalue)
        except Exception:
            pass

    return {
        "n_pairs_total": n_total,
        "n_pairs_nonzero": n_nonzero,
        "delta_mean": mean_delta,
        "delta_ci95_low_boot": ci_low,
        "delta_ci95_high_boot": ci_high,
        "paired_t_stat": t_stat,
        "paired_t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
    }


def _dataset_cluster_mean(values: pd.DataFrame, value_col: str) -> np.ndarray:
    if values.empty or value_col not in values.columns:
        return np.asarray([], dtype=float)
    if "dataset" not in values.columns:
        return np.asarray([], dtype=float)
    return (
        values.groupby("dataset", as_index=False)[value_col]
        .mean()[value_col]
        .dropna()
        .to_numpy(dtype=float)
    )


def _severe_policy_subset(
    policy_summary: pd.DataFrame,
    scenario: str,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    if scenario == "single_0.4":
        mask_single = (
            (policy_summary["corruption"].isin(["C2", "C4"]))
            & (policy_summary["severity_single"] == 0.4)
        )
        mask_c3 = (
            (policy_summary["corruption"] == "C3")
            & (policy_summary["severity_single"] == 0.4)
        )
        if c3_eligible_datasets is not None:
            mask_c3 = mask_c3 & policy_summary["dataset"].isin(c3_eligible_datasets)
        return policy_summary[mask_single | mask_c3].copy()
    if scenario == "compound_0.4_0.4":
        return policy_summary[
            (policy_summary["corruption"] == "C2+C4")
            & (policy_summary["severity_c2"] == 0.4)
            & (policy_summary["severity_c4"] == 0.4)
        ].copy()
    return pd.DataFrame()


def _threshold_policy_pairwise_tests_table(
    policy_summary: pd.DataFrame,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2036)
    key_cols = ["dataset", "encoding", "model", "corruption", "severity"]

    for scenario in ["single_0.4", "compound_0.4_0.4"]:
        sub = _severe_policy_subset(
            policy_summary,
            scenario=scenario,
            c3_eligible_datasets=c3_eligible_datasets,
        )
        if sub.empty:
            continue
        sub = sub[sub["variant"] == "uncal"].copy()
        if sub.empty:
            continue

        base = sub[sub["policy"] == "fixed_0_5"][
            key_cols + ["f1_policy_mean", "expected_cost_policy_mean"]
        ].rename(
            columns={
                "f1_policy_mean": "f1_base",
                "expected_cost_policy_mean": "cost_base",
            }
        )
        if base.empty:
            continue

        deltas: Dict[str, pd.DataFrame] = {}
        for policy in ["val_tuned_f1", "cost_sensitive"]:
            pol = sub[sub["policy"] == policy]
            if pol.empty:
                continue
            merged = pol.merge(base, on=key_cols, how="inner")
            if merged.empty:
                continue
            out = merged[key_cols].copy()
            out["delta_f1_vs_fixed"] = merged["f1_policy_mean"] - merged["f1_base"]
            out["delta_cost_vs_fixed"] = (
                merged["expected_cost_policy_mean"] - merged["cost_base"]
            )
            deltas[policy] = out

        if "val_tuned_f1" not in deltas or "cost_sensitive" not in deltas:
            continue

        pair = deltas["val_tuned_f1"].merge(
            deltas["cost_sensitive"],
            on=key_cols,
            how="inner",
            suffixes=("_val", "_cost"),
        )
        if pair.empty:
            continue

        for metric in ["delta_f1_vs_fixed", "delta_cost_vs_fixed"]:
            diff_col = "cluster_diff"
            pair_local = pair[["dataset"]].copy()
            pair_local[diff_col] = (
                pair[f"{metric}_val"].to_numpy(dtype=float)
                - pair[f"{metric}_cost"].to_numpy(dtype=float)
            )
            dataset_diffs = _dataset_cluster_mean(pair_local, diff_col)
            if len(dataset_diffs) == 0:
                continue
            stats = _paired_difference_inference(dataset_diffs, rng=rng)
            if metric == "delta_f1_vs_fixed":
                if stats["delta_mean"] > 0:
                    favored = "val_tuned_f1"
                elif stats["delta_mean"] < 0:
                    favored = "cost_sensitive"
                else:
                    favored = "tie"
            else:
                if stats["delta_mean"] < 0:
                    favored = "val_tuned_f1"
                elif stats["delta_mean"] > 0:
                    favored = "cost_sensitive"
                else:
                    favored = "tie"

            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "comparison": "val_tuned_f1_minus_cost_sensitive",
                    "analysis_unit": "dataset_cluster_mean",
                    "favored_policy_by_mean": favored,
                    **stats,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # One multiplicity family for the full policy pairwise battery
    # (all scenarios and metrics in this table).
    table["paired_t_p_value_bh"] = np.nan
    table["wilcoxon_p_value_bh"] = np.nan
    valid_t = table["paired_t_p_value"].notna()
    if valid_t.any():
        idx = table.index[valid_t]
        table.loc[idx, "paired_t_p_value_bh"] = _benjamini_hochberg(
            table.loc[idx, "paired_t_p_value"].to_numpy()
        )
    valid_w = table["wilcoxon_p_value"].notna()
    if valid_w.any():
        idx = table.index[valid_w]
        table.loc[idx, "wilcoxon_p_value_bh"] = _benjamini_hochberg(
            table.loc[idx, "wilcoxon_p_value"].to_numpy()
        )

    scenario_order = {"single_0.4": 0, "compound_0.4_0.4": 1}
    metric_order = {"delta_f1_vs_fixed": 0, "delta_cost_vs_fixed": 1}
    table["scenario_order"] = table["scenario"].map(scenario_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(["scenario_order", "metric_order"]).drop(
        columns=["scenario_order", "metric_order"]
    )
    return table.reset_index(drop=True)


def _record_calibration_metrics(
    rows: List[Dict],
    dataset: str,
    encoding: str,
    seed: int,
    model: str,
    corruption: str,
    severity,
    y_true,
    proba_by_calibrator: Dict[str, np.ndarray],
):
    for calibrator, proba in proba_by_calibrator.items():
        m = compute_metrics(y_true, proba)
        d = compute_calibration_diagnostics(y_true, proba)
        rows.append(
            {
                "dataset": dataset,
                "encoding": encoding,
                "seed": seed,
                "model": model,
                "corruption": corruption,
                "severity": severity,
                "calibrator": calibrator,
                **m,
                **d,
            }
        )


def _aggregate_calibration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "encoding", "model", "corruption", "severity", "calibrator"]
    agg_dict = {}
    for metric in CALIBRATION_METRICS:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")
    agg_dict["n"] = ("auc", "count")

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()
    for metric in CALIBRATION_METRICS:
        std_col = f"{metric}_std"
        sem_col = f"{metric}_sem"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        agg[std_col] = agg[std_col].fillna(0.0)
        agg[sem_col] = agg[std_col] / np.sqrt(agg["n"].clip(lower=1))
        t_crit = pd.Series(0.0, index=agg.index, dtype=float)
        mask = agg["n"] > 1
        t_crit.loc[mask] = t_dist.ppf(0.975, agg.loc[mask, "n"] - 1)
        half_width = t_crit * agg[sem_col]
        agg[low_col] = agg[f"{metric}_mean"] - half_width
        agg[high_col] = agg[f"{metric}_mean"] + half_width
    return agg


def _calibration_comparison_table(
    calib_summary: pd.DataFrame,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    scenarios = [
        ("C0_clean", "clean_0.0"),
        ("C2", "single_0.4"),
        ("C3", "single_0.4"),
        ("C4", "single_0.4"),
        ("C2+C4", "compound_0.4_0.4"),
    ]
    calibrator_order = ["uncal", "temp_scaled", "platt", "isotonic", "beta"]
    rows = []
    rng = np.random.default_rng(2035)

    for corruption, scenario in scenarios:
        if corruption == "C2+C4":
            sub = calib_summary[
                (calib_summary["corruption"] == "C2+C4")
                & (calib_summary["severity_c2"] == 0.4)
                & (calib_summary["severity_c4"] == 0.4)
            ]
        elif corruption == "C0_clean":
            sub = calib_summary[
                (calib_summary["corruption"] == "C0_clean")
                & (calib_summary["severity_single"] == 0.0)
            ]
        else:
            sub = calib_summary[
                (calib_summary["corruption"] == corruption)
                & (calib_summary["severity_single"] == 0.4)
            ]
            if corruption == "C3" and c3_eligible_datasets is not None:
                sub = sub[sub["dataset"].isin(c3_eligible_datasets)]

        if sub.empty:
            continue

        key_cols = ["dataset", "encoding", "model"]
        base = sub[sub["calibrator"] == "uncal"][
            key_cols
            + [
                "ece_mean",
                "ece_adaptive_mean",
                "brier_mean",
                "logloss_mean",
                "calib_slope_mean",
                "calib_intercept_mean",
                "rel_gap_mean_mean",
            ]
        ]
        base = base.rename(
            columns={
                "ece_mean": "ece_base",
                "ece_adaptive_mean": "ece_adaptive_base",
                "brier_mean": "brier_base",
                "logloss_mean": "logloss_base",
                "calib_slope_mean": "calib_slope_base",
                "calib_intercept_mean": "calib_intercept_base",
                "rel_gap_mean_mean": "rel_gap_mean_base",
            }
        )
        if base.empty:
            continue

        for calibrator in calibrator_order:
            csub = sub[sub["calibrator"] == calibrator]
            if csub.empty:
                continue
            merged = csub.merge(base, on=key_cols, how="inner")
            if merged.empty:
                continue
            delta_ece = (merged["ece_mean"] - merged["ece_base"]).to_numpy(dtype=float)
            delta_logloss = (
                merged["logloss_mean"] - merged["logloss_base"]
            ).to_numpy(dtype=float)
            ci_ece_low, ci_ece_high = _bootstrap_mean_ci(delta_ece, rng=rng)
            ci_ll_low, ci_ll_high = _bootstrap_mean_ci(delta_logloss, rng=rng)
            rows.append(
                {
                    "scenario": scenario,
                    "corruption": corruption,
                    "calibrator": calibrator,
                    "n_settings": int(len(merged)),
                    "auc_mean": float(merged["auc_mean"].mean()),
                    "f1_mean": float(merged["f1_mean"].mean()),
                    "ece_mean": float(merged["ece_mean"].mean()),
                    "ece_adaptive_mean": float(merged["ece_adaptive_mean"].mean()),
                    "brier_mean": float(merged["brier_mean"].mean()),
                    "logloss_mean": float(merged["logloss_mean"].mean()),
                    "calib_slope_mean": float(merged["calib_slope_mean"].mean()),
                    "calib_intercept_mean": float(merged["calib_intercept_mean"].mean()),
                    "rel_gap_mean_mean": float(merged["rel_gap_mean_mean"].mean()),
                    "rel_gap_max_mean": float(merged["rel_gap_max_mean"].mean()),
                    "delta_ece_vs_uncal": float(delta_ece.mean()),
                    "delta_ece_adaptive_vs_uncal": float(
                        (merged["ece_adaptive_mean"] - merged["ece_adaptive_base"]).mean()
                    ),
                    "delta_brier_vs_uncal": float(
                        (merged["brier_mean"] - merged["brier_base"]).mean()
                    ),
                    "delta_logloss_vs_uncal": float(delta_logloss.mean()),
                    "delta_ece_vs_uncal_ci95_low_boot": ci_ece_low,
                    "delta_ece_vs_uncal_ci95_high_boot": ci_ece_high,
                    "delta_logloss_vs_uncal_ci95_low_boot": ci_ll_low,
                    "delta_logloss_vs_uncal_ci95_high_boot": ci_ll_high,
                    "delta_rel_gap_mean_vs_uncal": float(
                        (merged["rel_gap_mean_mean"] - merged["rel_gap_mean_base"]).mean()
                    ),
                    "delta_abs_slope_error_vs_uncal": float(
                        (
                            np.abs(merged["calib_slope_mean"] - 1.0)
                            - np.abs(merged["calib_slope_base"] - 1.0)
                        ).mean()
                    ),
                    "delta_abs_intercept_vs_uncal": float(
                        (
                            np.abs(merged["calib_intercept_mean"])
                            - np.abs(merged["calib_intercept_base"])
                        ).mean()
                    ),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    scenario_order = {"clean_0.0": 0, "single_0.4": 1, "compound_0.4_0.4": 2}
    corruption_order = {"C0_clean": 0, "C2": 1, "C3": 2, "C4": 3, "C2+C4": 4}
    cal_order = {name: i for i, name in enumerate(calibrator_order)}
    table["scenario_order"] = table["scenario"].map(scenario_order)
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["calibrator_order"] = table["calibrator"].map(cal_order)
    table = table.sort_values(["scenario_order", "corruption_order", "calibrator_order"])
    return table.drop(columns=["scenario_order", "corruption_order", "calibrator_order"]).reset_index(drop=True)


def _severe_calibration_subset(
    calib_summary: pd.DataFrame,
    scenario: str,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    if scenario == "single_0.4":
        mask_single = (
            (calib_summary["corruption"].isin(["C2", "C4"]))
            & (calib_summary["severity_single"] == 0.4)
        )
        mask_c3 = (
            (calib_summary["corruption"] == "C3")
            & (calib_summary["severity_single"] == 0.4)
        )
        if c3_eligible_datasets is not None:
            mask_c3 = mask_c3 & calib_summary["dataset"].isin(c3_eligible_datasets)
        return calib_summary[mask_single | mask_c3].copy()
    if scenario == "compound_0.4_0.4":
        return calib_summary[
            (calib_summary["corruption"] == "C2+C4")
            & (calib_summary["severity_c2"] == 0.4)
            & (calib_summary["severity_c4"] == 0.4)
        ].copy()
    return pd.DataFrame()


def _calibration_pairwise_tests_table(
    calib_summary: pd.DataFrame,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    from itertools import combinations

    rows = []
    rng = np.random.default_rng(2037)
    calibrators = ["temp_scaled", "platt", "isotonic", "beta"]
    key_cols = ["dataset", "encoding", "model", "corruption", "severity"]

    for scenario in ["single_0.4", "compound_0.4_0.4"]:
        sub = _severe_calibration_subset(
            calib_summary,
            scenario=scenario,
            c3_eligible_datasets=c3_eligible_datasets,
        )
        if sub.empty:
            continue

        base = sub[sub["calibrator"] == "uncal"][
            key_cols + ["ece_mean", "logloss_mean"]
        ].rename(
            columns={
                "ece_mean": "ece_base",
                "logloss_mean": "logloss_base",
            }
        )
        if base.empty:
            continue

        deltas: Dict[str, pd.DataFrame] = {}
        for calibrator in calibrators:
            csub = sub[sub["calibrator"] == calibrator]
            if csub.empty:
                continue
            merged = csub.merge(base, on=key_cols, how="inner")
            if merged.empty:
                continue
            out = merged[key_cols].copy()
            out["delta_ece_vs_uncal"] = merged["ece_mean"] - merged["ece_base"]
            out["delta_logloss_vs_uncal"] = merged["logloss_mean"] - merged["logloss_base"]
            deltas[calibrator] = out

        for cal_a, cal_b in combinations(calibrators, 2):
            if cal_a not in deltas or cal_b not in deltas:
                continue
            pair = deltas[cal_a].merge(
                deltas[cal_b],
                on=key_cols,
                how="inner",
                suffixes=(f"_{cal_a}", f"_{cal_b}"),
            )
            if pair.empty:
                continue

            for metric in ["delta_ece_vs_uncal", "delta_logloss_vs_uncal"]:
                diff_col = "cluster_diff"
                pair_local = pair[["dataset"]].copy()
                pair_local[diff_col] = (
                    pair[f"{metric}_{cal_a}"].to_numpy(dtype=float)
                    - pair[f"{metric}_{cal_b}"].to_numpy(dtype=float)
                )
                dataset_diffs = _dataset_cluster_mean(pair_local, diff_col)
                if len(dataset_diffs) == 0:
                    continue
                stats = _paired_difference_inference(dataset_diffs, rng=rng)
                if stats["delta_mean"] < 0:
                    favored = cal_a
                elif stats["delta_mean"] > 0:
                    favored = cal_b
                else:
                    favored = "tie"

                rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "calibrator_a": cal_a,
                        "calibrator_b": cal_b,
                        "comparison": f"{cal_a}_minus_{cal_b}",
                        "analysis_unit": "dataset_cluster_mean",
                        "favored_calibrator_by_mean": favored,
                        **stats,
                    }
                )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # One multiplicity family for the full calibration pairwise battery
    # (all scenarios and metrics in this table).
    table["paired_t_p_value_bh"] = np.nan
    table["wilcoxon_p_value_bh"] = np.nan
    valid_t = table["paired_t_p_value"].notna()
    if valid_t.any():
        idx = table.index[valid_t]
        table.loc[idx, "paired_t_p_value_bh"] = _benjamini_hochberg(
            table.loc[idx, "paired_t_p_value"].to_numpy()
        )
    valid_w = table["wilcoxon_p_value"].notna()
    if valid_w.any():
        idx = table.index[valid_w]
        table.loc[idx, "wilcoxon_p_value_bh"] = _benjamini_hochberg(
            table.loc[idx, "wilcoxon_p_value"].to_numpy()
        )

    scenario_order = {"single_0.4": 0, "compound_0.4_0.4": 1}
    metric_order = {"delta_ece_vs_uncal": 0, "delta_logloss_vs_uncal": 1}
    table["scenario_order"] = table["scenario"].map(scenario_order)
    table["metric_order"] = table["metric"].map(metric_order)
    table = table.sort_values(
        ["scenario_order", "metric_order", "calibrator_a", "calibrator_b"]
    ).drop(columns=["scenario_order", "metric_order"])
    return table.reset_index(drop=True)


def _calibration_diagnostics_table(
    calib_summary: pd.DataFrame,
    c3_eligible_datasets: set[str] | None = None,
) -> pd.DataFrame:
    scenarios = [
        ("C0_clean", "clean_0.0"),
        ("C2", "single_0.4"),
        ("C3", "single_0.4"),
        ("C4", "single_0.4"),
        ("C2+C4", "compound_0.4_0.4"),
    ]
    calibrator_order = ["uncal", "temp_scaled", "platt", "isotonic", "beta"]
    rows = []

    for corruption, scenario in scenarios:
        if corruption == "C2+C4":
            sub = calib_summary[
                (calib_summary["corruption"] == "C2+C4")
                & (calib_summary["severity_c2"] == 0.4)
                & (calib_summary["severity_c4"] == 0.4)
            ]
        elif corruption == "C0_clean":
            sub = calib_summary[
                (calib_summary["corruption"] == "C0_clean")
                & (calib_summary["severity_single"] == 0.0)
            ]
        else:
            sub = calib_summary[
                (calib_summary["corruption"] == corruption)
                & (calib_summary["severity_single"] == 0.4)
            ]
            if corruption == "C3" and c3_eligible_datasets is not None:
                sub = sub[sub["dataset"].isin(c3_eligible_datasets)]

        if sub.empty:
            continue

        for calibrator in calibrator_order:
            csub = sub[sub["calibrator"] == calibrator]
            if csub.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "corruption": corruption,
                    "calibrator": calibrator,
                    "n_settings": int(len(csub)),
                    "ece_mean": float(csub["ece_mean"].mean()),
                    "ece_adaptive_mean": float(csub["ece_adaptive_mean"].mean()),
                    "calib_slope_mean": float(csub["calib_slope_mean"].mean()),
                    "calib_intercept_mean": float(csub["calib_intercept_mean"].mean()),
                    "abs_slope_error_mean": float(np.abs(csub["calib_slope_mean"] - 1.0).mean()),
                    "abs_intercept_mean": float(np.abs(csub["calib_intercept_mean"]).mean()),
                    "rel_gap_mean_mean": float(csub["rel_gap_mean_mean"].mean()),
                    "rel_gap_max_mean": float(csub["rel_gap_max_mean"].mean()),
                    "rel_nonempty_bins_mean": float(csub["rel_nonempty_bins_mean"].mean()),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    scenario_order = {"clean_0.0": 0, "single_0.4": 1, "compound_0.4_0.4": 2}
    corruption_order = {"C0_clean": 0, "C2": 1, "C3": 2, "C4": 3, "C2+C4": 4}
    cal_order = {name: i for i, name in enumerate(calibrator_order)}
    table["scenario_order"] = table["scenario"].map(scenario_order)
    table["corruption_order"] = table["corruption"].map(corruption_order)
    table["calibrator_order"] = table["calibrator"].map(cal_order)
    table = table.sort_values(["scenario_order", "corruption_order", "calibrator_order"])
    return table.drop(columns=["scenario_order", "corruption_order", "calibrator_order"]).reset_index(drop=True)


def _write_run_metadata(
    start_ts: float,
    end_ts: float,
    dataset_set: str,
    dataset_source_policy: str,
    selected_datasets: List[str],
    dataset_profiles: Dict[str, Dict[str, Any]],
    expected_raw_rows: int,
    expected_summary_rows: int,
    artifact_rows: Dict[str, int],
    cost_fp: float,
    cost_fn: float,
    run_command: str,
    c1_anchor_mode: str,
    snapshot_info: Dict[str, str],
):
    lock_path = "requirements-lock.txt"
    code_fp = _code_fingerprint(snapshot_info=snapshot_info)
    payload = {
        "utc_started": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "utc_finished": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "runtime_seconds": round(end_ts - start_ts, 6),
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "platform_machine": platform.machine(),
            "platform_processor": platform.processor(),
        },
        "code_fingerprint": code_fp,
        "configuration": {
            "seeds": SEEDS,
            "single_alphas": SINGLE_ALPHAS,
            "c2_levels": C2_LEVELS,
            "c4_levels": C4_LEVELS,
            "encodings": ENCODINGS,
            "dataset_set": dataset_set,
            "dataset_source_policy": dataset_source_policy,
            "datasets": selected_datasets,
            "single_corruptions": SINGLE_CORRUPTIONS,
            "cost_fp": float(cost_fp),
            "cost_fn": float(cost_fn),
            "cost_sensitive_threshold": float(cost_fp / (cost_fp + cost_fn)),
            "c1_anchor_mode": c1_anchor_mode,
            "entrypoint_command": run_command,
            "lockfile_path": lock_path,
            "lockfile_sha256": _file_sha256(lock_path),
            "public_repo_url": (
                os.getenv("PUBLIC_REPO_URL", "").strip() or DEFAULT_PUBLIC_REPO_URL
            ),
            "public_release_tag": (
                os.getenv("PUBLIC_RELEASE_TAG", "").strip() or DEFAULT_PUBLIC_RELEASE_TAG
            ),
            "public_release_commit": (
                os.getenv("PUBLIC_RELEASE_COMMIT", "").strip() or DEFAULT_PUBLIC_RELEASE_COMMIT
            ),
            "public_archive_doi": (
                os.getenv("PUBLIC_ARCHIVE_DOI", "").strip() or DEFAULT_PUBLIC_ARCHIVE_DOI
            ),
        },
        "expected_rows": {
            "metrics_long": int(expected_raw_rows),
            "metrics_summary": int(expected_summary_rows),
        },
        "artifact_rows": {k: int(v) for k, v in artifact_rows.items()},
        "dataset_profiles": dataset_profiles,
    }
    with open("results/run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run():
    start_ts = time.time()
    _ensure_dirs()
    results = []
    policy_results = []
    calibration_rows = []
    corruption_diagnostics = []
    dataset_profiles: Dict[str, Dict[str, Any]] = {}
    datasets, dataset_set = _resolve_datasets()
    dataset_source_policy = os.getenv("DATASET_SOURCE_POLICY", "openml_only").strip().lower()
    cost_fp = float(os.getenv("COST_FP", "1.0"))
    cost_fn = float(os.getenv("COST_FN", "5.0"))
    c1_anchor_mode = os.getenv("C1_ANCHOR_MODE", "label_agnostic").strip().lower()
    if c1_anchor_mode not in {"label_informed", "label_agnostic"}:
        raise ValueError("C1_ANCHOR_MODE must be one of: 'label_informed', 'label_agnostic'.")
    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("COST_FP and COST_FN must be > 0.")
    cost_sensitive_threshold = cost_fp / (cost_fp + cost_fn)
    model_names = list(get_models(SEEDS[0]).keys())
    expected_raw_rows, expected_summary_rows = _expected_row_counts(
        len(model_names), len(datasets)
    )

    for dataset in datasets:
        print(f"[run] loading dataset={dataset}")
        X, y, schema, profile = load_dataset(dataset)
        dataset_profiles[dataset] = profile

        for encoding in ENCODINGS:
            for seed in SEEDS:
                X_train, y_train, X_val, y_val, X_test, y_test = _split_data(X, y, seed)
                preprocessor = build_preprocessor(schema, encoding)
                X_train_proc = preprocessor.fit_transform(X_train)
                X_val_proc = preprocessor.transform(X_val)
                X_test_proc = preprocessor.transform(X_test)

                models = get_models(seed)
                for model_name in model_names:
                    model = models[model_name]
                    model.fit(X_train_proc, y_train)
                    proba_val = model.predict_proba(X_val_proc)
                    T = temperature_scale_fit(proba_val, y_val)
                    platt_model = platt_fit(proba_val, y_val)
                    isotonic_model = isotonic_fit(proba_val, y_val)
                    beta_model = beta_calibration_fit(proba_val, y_val)
                    proba_val_uncal = _to_1d_proba(proba_val)
                    proba_val_cal = _to_1d_proba(temperature_scale_predict(proba_val, T))
                    policy_thresholds = {
                        "uncal": {
                            "fixed_0_5": 0.5,
                            "val_tuned_f1": _optimal_f1_threshold(y_val, proba_val_uncal),
                            "cost_sensitive": cost_sensitive_threshold,
                        },
                        "temp_scaled": {
                            "fixed_0_5": 0.5,
                            "val_tuned_f1": _optimal_f1_threshold(y_val, proba_val_cal),
                            "cost_sensitive": cost_sensitive_threshold,
                        },
                    }

                    def _calibrate_all(proba):
                        return {
                            "uncal": _to_1d_proba(proba),
                            "temp_scaled": temperature_scale_predict(proba, T),
                            "platt": platt_predict(proba, platt_model),
                            "isotonic": isotonic_predict(proba, isotonic_model),
                            "beta": beta_calibration_predict(proba, beta_model),
                        }

                    proba_test = model.predict_proba(X_test_proc)
                    _record(
                        results,
                        dataset,
                        encoding,
                        seed,
                        model_name,
                        "C0_clean",
                        0.0,
                        y_test,
                        proba_test,
                        T,
                        policy_results=policy_results,
                        policy_thresholds=policy_thresholds,
                        cost_fp=cost_fp,
                        cost_fn=cost_fn,
                    )
                    _record_calibration_metrics(
                        calibration_rows,
                        dataset,
                        encoding,
                        seed,
                        model_name,
                        "C0_clean",
                        0.0,
                        y_test,
                        _calibrate_all(proba_test),
                    )

                    for alpha in SINGLE_ALPHAS:
                        X_c1, y_c1, d_c1 = apply_c1_duplication_with_diagnostics(
                            X_test,
                            y_test,
                            alpha,
                            seed + 11,
                            anchor_mode=c1_anchor_mode,
                        )
                        X_c1_proc = preprocessor.transform(X_c1)
                        proba_c1 = model.predict_proba(X_c1_proc)
                        _record(
                            results,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C1",
                            float(alpha),
                            y_c1,
                            proba_c1,
                            T,
                            policy_results=policy_results,
                            policy_thresholds=policy_thresholds,
                            cost_fp=cost_fp,
                            cost_fn=cost_fn,
                        )
                        _record_corruption_diagnostics(
                            corruption_diagnostics,
                            dataset=dataset,
                            encoding=encoding,
                            seed=seed,
                            model=model_name,
                            corruption="C1",
                            severity=float(alpha),
                            diagnostics=d_c1,
                        )
                        _record_calibration_metrics(
                            calibration_rows,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C1",
                            float(alpha),
                            y_c1,
                            _calibrate_all(proba_c1),
                        )

                        X_c2, d_c2 = apply_c2_missingness_with_diagnostics(
                            X_test, schema, alpha, seed + 21
                        )
                        X_c2_proc = preprocessor.transform(X_c2)
                        proba_c2 = model.predict_proba(X_c2_proc)
                        _record(
                            results,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C2",
                            float(alpha),
                            y_test,
                            proba_c2,
                            T,
                            policy_results=policy_results,
                            policy_thresholds=policy_thresholds,
                            cost_fp=cost_fp,
                            cost_fn=cost_fn,
                        )
                        _record_corruption_diagnostics(
                            corruption_diagnostics,
                            dataset=dataset,
                            encoding=encoding,
                            seed=seed,
                            model=model_name,
                            corruption="C2",
                            severity=float(alpha),
                            diagnostics=d_c2,
                        )
                        _record_calibration_metrics(
                            calibration_rows,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C2",
                            float(alpha),
                            y_test,
                            _calibrate_all(proba_c2),
                        )

                        X_c3, d_c3 = apply_c3_categorical_drift_with_diagnostics(
                            X_test, schema, alpha, seed + 31
                        )
                        X_c3_proc = preprocessor.transform(X_c3)
                        proba_c3 = model.predict_proba(X_c3_proc)
                        _record(
                            results,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C3",
                            float(alpha),
                            y_test,
                            proba_c3,
                            T,
                            policy_results=policy_results,
                            policy_thresholds=policy_thresholds,
                            cost_fp=cost_fp,
                            cost_fn=cost_fn,
                        )
                        _record_corruption_diagnostics(
                            corruption_diagnostics,
                            dataset=dataset,
                            encoding=encoding,
                            seed=seed,
                            model=model_name,
                            corruption="C3",
                            severity=float(alpha),
                            diagnostics=d_c3,
                        )
                        _record_calibration_metrics(
                            calibration_rows,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C3",
                            float(alpha),
                            y_test,
                            _calibrate_all(proba_c3),
                        )

                        X_c4, d_c4 = apply_c4_measurement_with_diagnostics(
                            X_test, schema, alpha, seed + 41
                        )
                        X_c4_proc = preprocessor.transform(X_c4)
                        proba_c4 = model.predict_proba(X_c4_proc)
                        _record(
                            results,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C4",
                            float(alpha),
                            y_test,
                            proba_c4,
                            T,
                            policy_results=policy_results,
                            policy_thresholds=policy_thresholds,
                            cost_fp=cost_fp,
                            cost_fn=cost_fn,
                        )
                        _record_corruption_diagnostics(
                            corruption_diagnostics,
                            dataset=dataset,
                            encoding=encoding,
                            seed=seed,
                            model=model_name,
                            corruption="C4",
                            severity=float(alpha),
                            diagnostics=d_c4,
                        )
                        _record_calibration_metrics(
                            calibration_rows,
                            dataset,
                            encoding,
                            seed,
                            model_name,
                            "C4",
                            float(alpha),
                            y_test,
                            _calibrate_all(proba_c4),
                        )

                    for a in C2_LEVELS:
                        for b in C4_LEVELS:
                            X_comp, d_comp = apply_c2_c4_compound_with_diagnostics(
                                X_test, schema, a, b, seed + 51
                            )
                            X_comp_proc = preprocessor.transform(X_comp)
                            proba_comp = model.predict_proba(X_comp_proc)
                            severity = json.dumps({"C2": float(a), "C4": float(b)})
                            _record(
                                results,
                                dataset,
                                encoding,
                                seed,
                                model_name,
                                "C2+C4",
                                severity,
                                y_test,
                                proba_comp,
                                T,
                                policy_results=policy_results,
                                policy_thresholds=policy_thresholds,
                                cost_fp=cost_fp,
                                cost_fn=cost_fn,
                            )
                            _record_corruption_diagnostics(
                                corruption_diagnostics,
                                dataset=dataset,
                                encoding=encoding,
                                seed=seed,
                                model=model_name,
                                corruption="C2+C4",
                                severity=severity,
                                diagnostics=d_comp,
                            )
                            _record_calibration_metrics(
                                calibration_rows,
                                dataset,
                                encoding,
                                seed,
                                model_name,
                                "C2+C4",
                                severity,
                                y_test,
                                _calibrate_all(proba_comp),
                            )

    df = _add_severity_columns(pd.DataFrame(results))
    if len(df) != expected_raw_rows:
        raise RuntimeError(
            f"Unexpected raw row count: observed={len(df)} expected={expected_raw_rows}"
        )
    df.to_csv("results/raw/metrics_long.csv", index=False)

    summary_path = "results/metrics_summary.csv"
    agg = _add_severity_columns(_aggregate_metrics(df))
    if len(agg) != expected_summary_rows:
        raise RuntimeError(
            f"Unexpected summary row count: observed={len(agg)} expected={expected_summary_rows}"
        )
    agg.to_csv(summary_path, index=False)

    effects = _paired_seed_effects(df)
    effects.to_csv("results/tables/effects_seed_paired.csv", index=False)
    primary_inference = _primary_inference_table(effects)
    primary_inference.to_csv("results/tables/table_primary_inference_summary.csv", index=False)
    dataset_effects = _primary_dataset_seed_effects(df)
    dataset_effects.to_csv("results/tables/effects_dataset_paired.csv", index=False)
    primary_dataset_inference = _primary_dataset_inference_summary(dataset_effects)
    primary_dataset_inference.to_csv(
        "results/tables/table_primary_dataset_inference_summary.csv", index=False
    )
    primary_hypothesis_tests = _primary_hypothesis_tests_table(dataset_effects)
    primary_hypothesis_tests.to_csv(
        "results/tables/table_primary_hypothesis_tests.csv", index=False
    )
    primary_hierarchical = _primary_hierarchical_sensitivity_table(df)
    primary_hierarchical.to_csv(
        "results/tables/table_primary_hierarchical_sensitivity.csv", index=False
    )

    single_delta = _single_corruption_delta_table(agg)
    single_delta.to_csv("results/tables/table_single_corruption_delta.csv", index=False)

    compound_delta = _compound_delta_table(agg)
    compound_delta.to_csv("results/tables/table_compound_delta.csv", index=False)

    clean_table = _clean_performance_table(agg)
    clean_table.to_csv("results/tables/table_clean_performance.csv", index=False)

    cross_dataset = _cross_dataset_effect_table(
        single_delta, compound_delta, dataset_profiles=dataset_profiles
    )
    cross_dataset.to_csv("results/tables/table_cross_dataset_effects.csv", index=False)
    cross_dataset_weighted = _cross_dataset_effect_weighted_table(
        single_delta, compound_delta, dataset_profiles=dataset_profiles
    )
    cross_dataset_weighted.to_csv(
        "results/tables/table_cross_dataset_effects_weighted.csv", index=False
    )

    dataset_profiles_table = _dataset_profiles_table(dataset_profiles)
    dataset_profiles_table.to_csv("results/tables/table_dataset_profiles.csv", index=False)
    c3_eligible_datasets = {
        d for d, prof in dataset_profiles.items() if int(prof.get("n_cat_cols", 0)) > 0
    }

    corruption_diag_df = _add_severity_columns(pd.DataFrame(corruption_diagnostics))
    corruption_diag_df.to_csv("results/raw/corruption_diagnostics_long.csv", index=False)
    corruption_diag_table = _corruption_diagnostics_table(corruption_diag_df)
    corruption_diag_table.to_csv("results/tables/table_corruption_diagnostics.csv", index=False)

    policy_df = _add_severity_columns(pd.DataFrame(policy_results))
    policy_df.to_csv("results/raw/policy_metrics_long.csv", index=False)
    policy_summary = _add_severity_columns(_aggregate_policy_metrics(policy_df))
    policy_summary.to_csv("results/tables/policy_metrics_summary.csv", index=False)
    policy_table = _threshold_policy_comparison_table(
        policy_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    policy_table.to_csv("results/tables/table_threshold_policy_comparison.csv", index=False)
    policy_pairwise = _threshold_policy_pairwise_tests_table(
        policy_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    policy_pairwise.to_csv(
        "results/tables/table_threshold_policy_pairwise_tests.csv", index=False
    )

    calibration_df = _add_severity_columns(pd.DataFrame(calibration_rows))
    calibration_df.to_csv("results/raw/calibration_metrics_long.csv", index=False)
    calibration_summary = _add_severity_columns(_aggregate_calibration_metrics(calibration_df))
    calibration_summary.to_csv("results/tables/calibration_metrics_summary.csv", index=False)
    calibration_table = _calibration_comparison_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_table.to_csv("results/tables/table_calibration_comparison.csv", index=False)
    calibration_diag_table = _calibration_diagnostics_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_diag_table.to_csv("results/tables/table_calibration_diagnostics.csv", index=False)
    calibration_pairwise = _calibration_pairwise_tests_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_pairwise.to_csv(
        "results/tables/table_calibration_pairwise_tests.csv", index=False
    )

    generate_all_plots(
        summary_path,
        "results/plots",
        policy_table_csv="results/tables/table_threshold_policy_comparison.csv",
    )

    snapshot_info = _write_code_snapshot_manifest()
    end_ts = time.time()
    artifact_rows = {
        "metrics_long": len(df),
        "metrics_summary": len(agg),
        "effects_seed_paired": len(effects),
        "table_primary_inference_summary": len(primary_inference),
        "effects_dataset_paired": len(dataset_effects),
        "table_primary_dataset_inference_summary": len(primary_dataset_inference),
        "table_primary_hypothesis_tests": len(primary_hypothesis_tests),
        "table_primary_hierarchical_sensitivity": len(primary_hierarchical),
        "table_single_corruption_delta": len(single_delta),
        "table_compound_delta": len(compound_delta),
        "table_clean_performance": len(clean_table),
        "table_cross_dataset_effects": len(cross_dataset),
        "table_cross_dataset_effects_weighted": len(cross_dataset_weighted),
        "table_dataset_profiles": len(dataset_profiles_table),
        "corruption_diagnostics_long": len(corruption_diag_df),
        "table_corruption_diagnostics": len(corruption_diag_table),
        "policy_metrics_long": len(policy_df),
        "policy_metrics_summary": len(policy_summary),
        "table_threshold_policy_comparison": len(policy_table),
        "table_threshold_policy_pairwise_tests": len(policy_pairwise),
        "calibration_metrics_long": len(calibration_df),
        "calibration_metrics_summary": len(calibration_summary),
        "table_calibration_comparison": len(calibration_table),
        "table_calibration_diagnostics": len(calibration_diag_table),
        "table_calibration_pairwise_tests": len(calibration_pairwise),
        "code_snapshot_manifest": int(snapshot_info.get("snapshot_file_count", "0")),
    }
    _write_run_metadata(
        start_ts=start_ts,
        end_ts=end_ts,
        dataset_set=dataset_set,
        dataset_source_policy=dataset_source_policy,
        selected_datasets=datasets,
        dataset_profiles=dataset_profiles,
        expected_raw_rows=expected_raw_rows,
        expected_summary_rows=expected_summary_rows,
        artifact_rows=artifact_rows,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        run_command=(
            f"DATASET_SET={dataset_set};DATASET_SOURCE_POLICY={dataset_source_policy};"
            f"C1_ANCHOR_MODE={c1_anchor_mode};python run_experiments.py"
        ),
        c1_anchor_mode=c1_anchor_mode,
        snapshot_info=snapshot_info,
    )
    print(f"[run] complete in {end_ts - start_ts:.1f}s")


if __name__ == "__main__":
    run()
