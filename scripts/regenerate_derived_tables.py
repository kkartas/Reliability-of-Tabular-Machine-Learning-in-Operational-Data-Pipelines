from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiments import (
    _add_severity_columns,
    _aggregate_calibration_metrics,
    _aggregate_metrics,
    _aggregate_policy_metrics,
    _code_fingerprint,
    _calibration_comparison_table,
    _calibration_diagnostics_table,
    _clean_performance_table,
    _compound_delta_table,
    _cross_dataset_effect_table,
    _cross_dataset_effect_weighted_table,
    _dataset_profiles_table,
    _primary_dataset_inference_summary,
    _primary_dataset_seed_effects,
    _primary_hierarchical_sensitivity_table,
    _primary_hypothesis_tests_table,
    _paired_seed_effects,
    _primary_inference_table,
    _single_corruption_delta_table,
    _threshold_policy_pairwise_tests_table,
    _threshold_policy_comparison_table,
    _calibration_pairwise_tests_table,
    _write_code_snapshot_manifest,
)


def _normalize_dataset_profiles(run_metadata: dict) -> dict:
    config = run_metadata.setdefault("configuration", {})
    default_policy = config.get(
        "dataset_source_policy",
        os.getenv("DATASET_SOURCE_POLICY", "openml_only").strip().lower(),
    )
    profiles = run_metadata.get("dataset_profiles", {})
    for _, profile in profiles.items():
        source = str(profile.get("source", ""))
        source_policy = str(profile.get("source_policy", "")).strip()
        if not source_policy:
            profile["source_policy"] = default_policy
        source_sha = str(profile.get("source_sha256", "")).strip()
        if not source_sha:
            if source.startswith("openml"):
                profile["source_sha256"] = "N/A (OpenML)"
            else:
                profile["source_sha256"] = "N/A"
    return profiles


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = ROOT
    results_dir = root / "results"
    raw_dir = results_dir / "raw"
    tables_dir = results_dir / "tables"

    metrics_long = _add_severity_columns(pd.read_csv(raw_dir / "metrics_long.csv"))
    metrics_summary = _add_severity_columns(_aggregate_metrics(metrics_long))
    metrics_summary.to_csv(results_dir / "metrics_summary.csv", index=False)

    effects = _paired_seed_effects(metrics_long)
    effects.to_csv(tables_dir / "effects_seed_paired.csv", index=False)
    primary = _primary_inference_table(effects)
    primary.to_csv(tables_dir / "table_primary_inference_summary.csv", index=False)
    dataset_effects = _primary_dataset_seed_effects(metrics_long)
    dataset_effects.to_csv(tables_dir / "effects_dataset_paired.csv", index=False)
    dataset_primary = _primary_dataset_inference_summary(dataset_effects)
    dataset_primary.to_csv(
        tables_dir / "table_primary_dataset_inference_summary.csv", index=False
    )
    primary_hypothesis = _primary_hypothesis_tests_table(dataset_effects)
    primary_hypothesis.to_csv(
        tables_dir / "table_primary_hypothesis_tests.csv", index=False
    )
    primary_hierarchical = _primary_hierarchical_sensitivity_table(metrics_long)
    primary_hierarchical.to_csv(
        tables_dir / "table_primary_hierarchical_sensitivity.csv", index=False
    )

    single_delta = _single_corruption_delta_table(metrics_summary)
    single_delta.to_csv(tables_dir / "table_single_corruption_delta.csv", index=False)
    compound_delta = _compound_delta_table(metrics_summary)
    compound_delta.to_csv(tables_dir / "table_compound_delta.csv", index=False)
    clean_table = _clean_performance_table(metrics_summary)
    clean_table.to_csv(tables_dir / "table_clean_performance.csv", index=False)

    with open(results_dir / "run_metadata.json", "r", encoding="utf-8") as f:
        run_metadata = json.load(f)
    dataset_profiles = _normalize_dataset_profiles(run_metadata)
    c3_eligible_datasets = {
        d for d, prof in dataset_profiles.items() if int(prof.get("n_cat_cols", 0)) > 0
    }

    cross = _cross_dataset_effect_table(single_delta, compound_delta, dataset_profiles=dataset_profiles)
    cross.to_csv(tables_dir / "table_cross_dataset_effects.csv", index=False)
    cross_weighted = _cross_dataset_effect_weighted_table(
        single_delta, compound_delta, dataset_profiles=dataset_profiles
    )
    cross_weighted.to_csv(tables_dir / "table_cross_dataset_effects_weighted.csv", index=False)
    dataset_profiles_table = _dataset_profiles_table(dataset_profiles)
    dataset_profiles_table.to_csv(tables_dir / "table_dataset_profiles.csv", index=False)

    policy_long = _add_severity_columns(pd.read_csv(raw_dir / "policy_metrics_long.csv"))
    policy_summary = _add_severity_columns(_aggregate_policy_metrics(policy_long))
    policy_summary.to_csv(tables_dir / "policy_metrics_summary.csv", index=False)
    policy_table = _threshold_policy_comparison_table(
        policy_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    policy_table.to_csv(tables_dir / "table_threshold_policy_comparison.csv", index=False)
    policy_pairwise = _threshold_policy_pairwise_tests_table(
        policy_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    policy_pairwise.to_csv(
        tables_dir / "table_threshold_policy_pairwise_tests.csv", index=False
    )

    calibration_long = _add_severity_columns(pd.read_csv(raw_dir / "calibration_metrics_long.csv"))
    calibration_summary = _add_severity_columns(_aggregate_calibration_metrics(calibration_long))
    calibration_summary.to_csv(tables_dir / "calibration_metrics_summary.csv", index=False)
    calibration_table = _calibration_comparison_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_table.to_csv(tables_dir / "table_calibration_comparison.csv", index=False)
    calibration_diag = _calibration_diagnostics_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_diag.to_csv(tables_dir / "table_calibration_diagnostics.csv", index=False)
    calibration_pairwise = _calibration_pairwise_tests_table(
        calibration_summary, c3_eligible_datasets=c3_eligible_datasets
    )
    calibration_pairwise.to_csv(
        tables_dir / "table_calibration_pairwise_tests.csv", index=False
    )

    artifact_rows = run_metadata.get("artifact_rows", {})
    artifact_rows["table_cross_dataset_effects_weighted"] = int(len(cross_weighted))
    artifact_rows["table_dataset_profiles"] = int(len(dataset_profiles_table))
    artifact_rows["table_primary_inference_summary"] = int(len(primary))
    artifact_rows["effects_dataset_paired"] = int(len(dataset_effects))
    artifact_rows["table_primary_dataset_inference_summary"] = int(len(dataset_primary))
    artifact_rows["table_primary_hypothesis_tests"] = int(len(primary_hypothesis))
    artifact_rows["table_primary_hierarchical_sensitivity"] = int(len(primary_hierarchical))
    artifact_rows["table_threshold_policy_comparison"] = int(len(policy_table))
    artifact_rows["table_threshold_policy_pairwise_tests"] = int(len(policy_pairwise))
    artifact_rows["table_calibration_comparison"] = int(len(calibration_table))
    artifact_rows["table_calibration_pairwise_tests"] = int(len(calibration_pairwise))
    c1_summary = tables_dir / "table_c1_anchor_sensitivity.csv"
    c1_dataset = tables_dir / "table_c1_anchor_sensitivity_dataset.csv"
    c1_long = tables_dir / "c1_anchor_sensitivity_long.csv"
    if c1_summary.exists():
        artifact_rows["table_c1_anchor_sensitivity"] = int(len(pd.read_csv(c1_summary)))
    if c1_dataset.exists():
        artifact_rows["table_c1_anchor_sensitivity_dataset"] = int(len(pd.read_csv(c1_dataset)))
    if c1_long.exists():
        artifact_rows["c1_anchor_sensitivity_long"] = int(len(pd.read_csv(c1_long)))
    dataset_hash_manifest = results_dir / "dataset_hash_manifest.csv"
    if dataset_hash_manifest.exists():
        artifact_rows["dataset_hash_manifest"] = int(len(pd.read_csv(dataset_hash_manifest)))
        config = run_metadata.setdefault("configuration", {})
        config["dataset_hash_manifest_path"] = "results/dataset_hash_manifest.csv"
        config["dataset_hash_manifest_sha256"] = _sha256_file(dataset_hash_manifest)
    snapshot_info = _write_code_snapshot_manifest()
    artifact_rows["code_snapshot_manifest"] = int(snapshot_info.get("snapshot_file_count", "0"))
    run_metadata["code_fingerprint"] = _code_fingerprint(snapshot_info=snapshot_info)
    run_metadata["artifact_rows"] = artifact_rows
    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)


if __name__ == "__main__":
    main()
