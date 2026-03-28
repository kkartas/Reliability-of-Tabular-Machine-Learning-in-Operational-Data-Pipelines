from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corruptions import apply_c1_duplication_with_diagnostics
from datasets import load_dataset
from metrics import compute_metrics
from models import get_models
from preprocess import build_preprocessor
from run_experiments import ENCODINGS, PRIMARY_METRICS, SEEDS, _bootstrap_mean_ci, _resolve_datasets, _split_data


ALPHA_SEVERE = 0.4
ANCHOR_MODES = ["label_informed", "label_agnostic"]


def _fit_and_eval() -> pd.DataFrame:
    rows = []
    datasets, _ = _resolve_datasets()
    for dataset in datasets:
        print(f"[c1-sensitivity] loading dataset={dataset}")
        X, y, schema, _ = load_dataset(dataset)
        for encoding in ENCODINGS:
            for seed in SEEDS:
                X_train, y_train, _, _, X_test, y_test = _split_data(X, y, seed)
                preprocessor = build_preprocessor(schema, encoding)
                X_train_proc = preprocessor.fit_transform(X_train)
                X_test_proc = preprocessor.transform(X_test)

                models = get_models(seed)
                for model_name, model in models.items():
                    model.fit(X_train_proc, y_train)
                    clean_metrics = compute_metrics(y_test, model.predict_proba(X_test_proc))

                    for anchor_mode in ANCHOR_MODES:
                        X_c1, y_c1, _ = apply_c1_duplication_with_diagnostics(
                            X_test,
                            y_test,
                            ALPHA_SEVERE,
                            seed + 11,
                            anchor_mode=anchor_mode,
                        )
                        X_c1_proc = preprocessor.transform(X_c1)
                        c1_metrics = compute_metrics(y_c1, model.predict_proba(X_c1_proc))
                        for metric in PRIMARY_METRICS:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "encoding": encoding,
                                    "model": model_name,
                                    "seed": int(seed),
                                    "anchor_mode": anchor_mode,
                                    "metric": metric,
                                    "delta": float(c1_metrics[metric] - clean_metrics[metric]),
                                }
                            )
    return pd.DataFrame(rows)


def _aggregate_table(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_df = (
        long_df.groupby(["dataset", "encoding", "model", "anchor_mode", "metric"], as_index=False)
        .agg(delta_setting=("delta", "mean"))
    )
    dataset_df = (
        setting_df.groupby(["dataset", "anchor_mode", "metric"], as_index=False)
        .agg(delta_dataset=("delta_setting", "mean"))
    )

    rows = []
    rng = np.random.default_rng(2067)
    for metric in PRIMARY_METRICS:
        informed = dataset_df[
            (dataset_df["metric"] == metric) & (dataset_df["anchor_mode"] == "label_informed")
        ][["dataset", "delta_dataset"]].rename(columns={"delta_dataset": "delta_informed"})
        agnostic = dataset_df[
            (dataset_df["metric"] == metric) & (dataset_df["anchor_mode"] == "label_agnostic")
        ][["dataset", "delta_dataset"]].rename(columns={"delta_dataset": "delta_agnostic"})
        merged = informed.merge(agnostic, on="dataset", how="inner")
        if merged.empty:
            continue

        vals_informed = merged["delta_informed"].to_numpy(dtype=float)
        vals_agnostic = merged["delta_agnostic"].to_numpy(dtype=float)
        vals_gap = vals_agnostic - vals_informed
        inf_low, inf_high = _bootstrap_mean_ci(vals_informed, rng=rng)
        agn_low, agn_high = _bootstrap_mean_ci(vals_agnostic, rng=rng)
        gap_low, gap_high = _bootstrap_mean_ci(vals_gap, rng=rng)

        rows.append(
            {
                "metric": metric,
                "n_datasets": int(len(merged)),
                "label_informed_mean": float(np.mean(vals_informed)),
                "label_informed_ci95_low_boot": inf_low,
                "label_informed_ci95_high_boot": inf_high,
                "label_agnostic_mean": float(np.mean(vals_agnostic)),
                "label_agnostic_ci95_low_boot": agn_low,
                "label_agnostic_ci95_high_boot": agn_high,
                "delta_gap_agnostic_minus_informed_mean": float(np.mean(vals_gap)),
                "delta_gap_agnostic_minus_informed_ci95_low_boot": gap_low,
                "delta_gap_agnostic_minus_informed_ci95_high_boot": gap_high,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        metric_order = {"auc": 0, "f1": 1, "ece": 2}
        out["metric_order"] = out["metric"].map(metric_order)
        out = out.sort_values("metric_order").drop(columns=["metric_order"]).reset_index(drop=True)
    return out, dataset_df


def main() -> None:
    start = time.time()
    root = Path(__file__).resolve().parents[1]
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    long_df = _fit_and_eval()
    if long_df.empty:
        raise RuntimeError("No C1 sensitivity rows were produced.")
    long_df.to_csv(tables_dir / "c1_anchor_sensitivity_long.csv", index=False)

    summary_df, dataset_df = _aggregate_table(long_df)
    summary_df.to_csv(tables_dir / "table_c1_anchor_sensitivity.csv", index=False)
    dataset_df.to_csv(tables_dir / "table_c1_anchor_sensitivity_dataset.csv", index=False)

    elapsed = time.time() - start
    print(
        f"[c1-sensitivity] complete in {elapsed:.1f}s "
        f"(rows={len(long_df)}, table_rows={len(summary_df)})"
    )


if __name__ == "__main__":
    main()
